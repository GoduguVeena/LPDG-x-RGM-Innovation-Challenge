from pathlib import Path
import datetime as dt

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from features import build_training_dataset, load_telemetry
from evaluate import calculate_cost, prepare_actual_data


# ============================================================
# Configuration
# ============================================================

TARGET_THRESHOLD = 0.30

VISITS_PER_WEEK = 15

BASELINE_DAYS = 28
RECENT_DAYS = 7
SIGMA = 3.0

BASELINE_METRICS = [
    "offline_duration_sec",
    "disconnection_cnt",
    "reboot_cnt",
]


# ============================================================
# 3-Sigma Baseline
# ============================================================

def baseline_rank_week(frame, week_start):
    """
    Reproduce the supplied 3-sigma baseline logic
    for an arbitrary historical week.
    """

    end = pd.Timestamp(week_start)

    # Historical 28-day window strictly before prediction week
    window = frame[
        (frame["ts"] >= end - dt.timedelta(days=BASELINE_DAYS))
        & (frame["ts"] < end)
    ].copy()

    if window.empty:
        return pd.DataFrame(
            columns=[
                "gateway_id",
                "flagged_hours",
                "worst_metric",
            ]
        )

    # Calculate gateway-level historical mean and std
    stats = (
        window
        .groupby("gateway_id")[BASELINE_METRICS]
        .agg(["mean", "std"])
    )

    # Recent 7-day period
    recent = window[
        window["ts"] >= end - dt.timedelta(days=RECENT_DAYS)
    ].copy()

    flags = pd.Series(
        0,
        index=recent.index,
        dtype=int,
    )

    worst = pd.Series(
        "",
        index=recent.index,
        dtype=object,
    )

    # Check each baseline metric against 3-sigma threshold
    for metric in BASELINE_METRICS:

        mean = recent["gateway_id"].map(
            stats[(metric, "mean")]
        )

        std = (
            recent["gateway_id"]
            .map(stats[(metric, "std")])
            .replace(0, np.nan)
        )

        exceeded = (
            (recent[metric] - mean)
            > SIGMA * std
        )

        exceeded = exceeded.fillna(False)

        flags = flags + exceeded.astype(int)

        # Store first metric that breached the threshold
        worst = worst.where(
            ~exceeded | (worst != ""),
            metric,
        )

    recent["flagged"] = flags
    recent["worst_metric"] = worst

    # Aggregate flagged hours per gateway
    grouped = (
        recent
        .groupby("gateway_id")
        .agg(
            flagged_hours=("flagged", "sum"),
            worst_metric=(
                "worst_metric",
                lambda s: next(
                    (value for value in s if value),
                    "",
                ),
            ),
        )
        .reset_index()
    )

    # Deterministic ranking:
    # 1. More flagged hours first
    # 2. gateway_id breaks ties
    return (
        grouped
        .sort_values(
            ["flagged_hours", "gateway_id"],
            ascending=[False, True],
        )
        .reset_index(drop=True)
    )


# ============================================================
# Build Baseline Predictions
# ============================================================

def build_baseline_predictions(
    telemetry,
    weeks,
):
    """
    Generate exactly 15 baseline selections per week.
    """

    telemetry = telemetry.copy()

    telemetry["ts"] = pd.to_datetime(
        telemetry["ts_utc"],
        utc=True,
    )

    predictions = []

    for week in weeks:

        ranked = baseline_rank_week(
            telemetry,
            week,
        )

        top = ranked.head(
            VISITS_PER_WEEK
        )

        for rank, row in enumerate(
            top.itertuples(index=False),
            start=1,
        ):

            metric = (
                row.worst_metric
                or "no metric over 3 sigma"
            )

            predictions.append(
                {
                    "week_start": week,
                    "rank": rank,
                    "gateway_id": row.gateway_id,
                    "score": float(
                        row.flagged_hours
                    ),
                    "reason": (
                        f"{row.flagged_hours} flagged hour(s); "
                        f"first breach on {metric}"
                    ),
                }
            )

    return pd.DataFrame(predictions)


# ============================================================
# Build ML Predictions
# ============================================================

def build_ml_predictions(
    training_data,
    train_weeks,
    validation_weeks,
):
    """
    Train Logistic Regression using only training weeks
    and generate Top-15 predictions for future weeks.
    """

    training_data = training_data.copy()

    # Define target
    training_data["target"] = (
        training_data["failure_rate"]
        > TARGET_THRESHOLD
    ).astype(int)

    # Columns that must not be used as model features
    excluded = [
        "gateway_id",
        "week_start",
        "failure_rate",
        "target",
    ]

    feature_columns = [
        column
        for column in training_data.columns
        if column not in excluded
    ]

    # --------------------------------------------------------
    # Training data
    # --------------------------------------------------------

    train_data = training_data[
        training_data["week_start"].isin(
            train_weeks
        )
    ].copy()

    # --------------------------------------------------------
    # Validation data
    # --------------------------------------------------------

    validation_data = training_data[
        training_data["week_start"].isin(
            validation_weeks
        )
    ].copy()

    X_train = train_data[
        feature_columns
    ]

    y_train = train_data[
        "target"
    ]

    X_validation = validation_data[
        feature_columns
    ]

    # --------------------------------------------------------
    # Model pipeline
    # --------------------------------------------------------

    model = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    print("\nTraining ML model...")

    model.fit(
        X_train,
        y_train,
    )

    # --------------------------------------------------------
    # Predict probability of severe failure
    # --------------------------------------------------------

    validation_data["score"] = (
        model.predict_proba(
            X_validation
        )[:, 1]
    )

    predictions = []

    for week in validation_weeks:

        week_data = validation_data[
            validation_data["week_start"] == week
        ].copy()

        # Deterministic ranking:
        # highest probability first,
        # gateway_id breaks ties.
        week_data = week_data.sort_values(
            ["score", "gateway_id"],
            ascending=[False, True],
        )

        top = week_data.head(
            VISITS_PER_WEEK
        )

        for rank, row in enumerate(
            top.itertuples(index=False),
            start=1,
        ):

            predictions.append(
                {
                    "week_start": week,
                    "rank": rank,
                    "gateway_id": row.gateway_id,
                    "score": float(
                        row.score
                    ),
                    "reason": (
                        "Logistic Regression predicted "
                        f"{row.score:.4f} probability "
                        "of severe failure"
                    ),
                }
            )

    return (
        pd.DataFrame(predictions),
        model,
    )


# ============================================================
# Main Comparison Experiment
# ============================================================

def main():

    data_dir = (
        Path(__file__).resolve().parent.parent
        / "data"
    )

    print(
        "Building comparison experiment..."
    )

    # --------------------------------------------------------
    # 1. Build ML dataset
    # --------------------------------------------------------

    dataset = build_training_dataset(
        data_dir
    )

    dataset = (
        dataset
        .sort_values(
            ["week_start", "gateway_id"]
        )
        .reset_index(drop=True)
    )

    weeks = sorted(
        dataset["week_start"].unique()
    )

    validation_count = 8

    train_weeks = weeks[
        :-validation_count
    ]

    validation_weeks = weeks[
        -validation_count:
    ]

    print("\nTraining weeks:")

    print(
        pd.Timestamp(
            train_weeks[0]
        ).strftime("%Y-%m-%d"),
        "to",
        pd.Timestamp(
            train_weeks[-1]
        ).strftime("%Y-%m-%d"),
    )

    print("\nValidation weeks:")

    print(
        pd.Timestamp(
            validation_weeks[0]
        ).strftime("%Y-%m-%d"),
        "to",
        pd.Timestamp(
            validation_weeks[-1]
        ).strftime("%Y-%m-%d"),
    )

    # --------------------------------------------------------
    # 2. ML predictions
    # --------------------------------------------------------

    ml_predictions, model = (
        build_ml_predictions(
            dataset,
            train_weeks,
            validation_weeks,
        )
    )

    # --------------------------------------------------------
    # 3. Baseline predictions
    # --------------------------------------------------------

    print(
        "\nGenerating 3-sigma baseline predictions..."
    )

    telemetry = load_telemetry(
        data_dir
    )

    baseline_predictions = (
        build_baseline_predictions(
            telemetry,
            validation_weeks,
        )
    )

    # --------------------------------------------------------
    # 4. Verify Top-15 constraint
    # --------------------------------------------------------

    ml_counts = (
        ml_predictions
        .groupby("week_start")
        .size()
    )

    baseline_counts = (
        baseline_predictions
        .groupby("week_start")
        .size()
    )

    print("\nML visits per week:")

    print(
        ml_counts.to_string()
    )

    print("\nBaseline visits per week:")

    print(
        baseline_counts.to_string()
    )

    assert (
        ml_counts == VISITS_PER_WEEK
    ).all(), (
        "ML does not select exactly "
        "15 gateways every week."
    )

    assert (
        baseline_counts == VISITS_PER_WEEK
    ).all(), (
        "Baseline does not select exactly "
        "15 gateways every week."
    )

    # --------------------------------------------------------
    # 5. Historical actual outcomes
    # --------------------------------------------------------

    # IMPORTANT:
    # Pass the COMPLETE historical dataset to the cost
    # evaluator so fault episodes are calculated correctly.
    #
    # calculate_cost() will evaluate only the weeks contained
    # in ml_predictions / baseline_predictions.
    actual_data = prepare_actual_data(
        data_dir
    )

    # --------------------------------------------------------
    # 6. Calculate ML cost
    # --------------------------------------------------------

    ml_cost, ml_weekly = calculate_cost(
        actual_data,
        ml_predictions,
    )

    # --------------------------------------------------------
    # 7. Calculate baseline cost
    # --------------------------------------------------------

    baseline_cost, baseline_weekly = (
        calculate_cost(
            actual_data,
            baseline_predictions,
        )
    )

    # --------------------------------------------------------
    # 8. Compare
    # --------------------------------------------------------

    difference = (
        baseline_cost
        - ml_cost
    )

    improvement = (
        difference
        / baseline_cost
        * 100
    )

    print(
        "\n" + "=" * 60
    )

    print(
        "FINAL COST COMPARISON"
    )

    print(
        "=" * 60
    )

    print(
        f"\nML model cost:       "
        f"€{ml_cost:,.0f}"
    )

    print(
        f"3-sigma cost:       "
        f"€{baseline_cost:,.0f}"
    )

    print(
        f"\nCost difference:    "
        f"€{difference:,.0f}"
    )

    print(
        f"ML improvement:     "
        f"{improvement:.2f}%"
    )

    print(
        "\nWeekly comparison:"
    )

    comparison = pd.DataFrame(
        {
            "week_start": ml_weekly[
                "week_start"
            ],
            "ml_cost": ml_weekly[
                "total_cost"
            ].values,
            "baseline_cost": baseline_weekly[
                "total_cost"
            ].values,
        }
    )

    comparison["difference"] = (
        comparison["baseline_cost"]
        - comparison["ml_cost"]
    )

    print(
        comparison.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # 9. Save predictions and comparison
    # --------------------------------------------------------

    ml_predictions.to_csv(
        "predictions_ml_validation.csv",
        index=False,
    )

    baseline_predictions.to_csv(
        "predictions_baseline_validation.csv",
        index=False,
    )

    comparison.to_csv(
        "cost_comparison.csv",
        index=False,
    )

    print(
        "\nSaved:"
        "\n  predictions_ml_validation.csv"
        "\n  predictions_baseline_validation.csv"
        "\n  cost_comparison.csv"
    )


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    main()