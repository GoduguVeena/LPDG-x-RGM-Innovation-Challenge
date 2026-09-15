from pathlib import Path
import argparse

import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from features import (
    build_training_dataset,
    build_prediction_features,
)


TARGET_THRESHOLD = 0.30
VISITS_PER_WEEK = 15

# Default Part 1 submission period.
# Running the script without arguments produces the required
# 8 weeks from 2026-02-02 to 2026-03-23.
DEFAULT_START_WEEK = "2026-02-02"
DEFAULT_NUM_WEEKS = 8


def get_scored_weeks(start_week=DEFAULT_START_WEEK, num_weeks=DEFAULT_NUM_WEEKS):
    """
    Return the weeks for which predictions should be generated.

    Defaults to the exact Part 1 submission period, while allowing
    another prediction period for unseen-month evaluation.
    """

    if num_weeks < 1:
        raise ValueError("num_weeks must be at least 1")

    start_week = pd.Timestamp(start_week)

    if start_week.tzinfo is None:
        start_week = start_week.tz_localize("UTC")
    else:
        start_week = start_week.tz_convert("UTC")

    return pd.date_range(
        start=start_week,
        periods=num_weeks,
        freq="7D",
    )


def train_final_model(dataset):
    """
    Train the final ML model using historical labelled
    gateway-weeks only.
    """

    dataset = dataset.copy()

    dataset["target"] = (
        dataset["failure_rate"] > TARGET_THRESHOLD
    ).astype(int)

    excluded_columns = [
        "gateway_id",
        "week_start",
        "failure_rate",
        "target",
    ]

    feature_columns = [
        column
        for column in dataset.columns
        if column not in excluded_columns
    ]

    if len(feature_columns) != 69:
        raise ValueError(
            f"Expected 69 features, found {len(feature_columns)}"
        )

    X = dataset[feature_columns]
    y = dataset["target"]

    model = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
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

    print("\nTraining final Logistic Regression...")
    print("Training rows:", len(dataset))
    print("Features:", len(feature_columns))
    print(
        "Severe failure rate:",
        f"{y.mean():.2%}",
    )

    model.fit(X, y)

    return model, feature_columns


def build_reason(row):
    """
    Produce a human-readable explanation for the prediction.
    """

    reasons = []

    if pd.notna(row.get("offline_duration_sec_mean_7d")):
        reasons.append("recent offline duration")

    if pd.notna(row.get("disconnection_cnt_mean_7d")):
        reasons.append("recent disconnections")

    if pd.notna(row.get("reboot_cnt_mean_7d")):
        reasons.append("recent reboots")

    if pd.notna(row.get("telemetry_hours_7d")):
        coverage = row["telemetry_hours_7d"]

        if coverage < 100:
            reasons.append(
                "limited recent telemetry coverage"
            )

    if not reasons:
        return "Risk score based on available gateway telemetry"

    return (
        "Elevated predicted severe-failure risk based on "
        + ", ".join(reasons[:3])
    )


def predict_week(
    model,
    feature_columns,
    prediction_features,
    week_start,
):
    """
    Rank gateways for one prediction week.
    """

    prediction_features = prediction_features.copy()

    missing_features = [
        column
        for column in feature_columns
        if column not in prediction_features.columns
    ]

    if missing_features:
        raise ValueError(
            "Missing prediction features: "
            + ", ".join(missing_features)
        )

    X_prediction = prediction_features[
        feature_columns
    ]

    probabilities = model.predict_proba(
        X_prediction
    )[:, 1]

    prediction_features["score"] = probabilities

    # Deterministic ordering:
    # 1. highest predicted risk
    # 2. gateway_id alphabetically for ties
    prediction_features = prediction_features.sort_values(
        ["score", "gateway_id"],
        ascending=[False, True],
    ).reset_index(drop=True)

    selected = prediction_features.head(
        VISITS_PER_WEEK
    ).copy()

    if len(selected) != VISITS_PER_WEEK:
        raise ValueError(
            f"Expected {VISITS_PER_WEEK} gateways for "
            f"{week_start.date()}, found {len(selected)}"
        )

    selected["rank"] = range(
        1,
        VISITS_PER_WEEK + 1,
    )

    selected["week_start"] = (
        week_start.date().isoformat()
    )

    selected["reason"] = selected.apply(
        build_reason,
        axis=1,
    )

    return selected[
        [
            "week_start",
            "rank",
            "gateway_id",
            "score",
            "reason",
        ]
    ]


def build_predictions(data_dir, scored_weeks=None):
    """
    Train on historical labelled data and generate
    predictions for the requested prediction weeks.

    Training labels from the prediction period and any
    later weeks are excluded to prevent target leakage.
    """

    if scored_weeks is None:
        scored_weeks = get_scored_weeks()

    scored_weeks = list(scored_weeks)

    if not scored_weeks:
        raise ValueError(
            "At least one prediction week is required"
        )

    first_prediction_week = pd.Timestamp(
        scored_weeks[0]
    )

    print("==========================================")
    print("FINAL ML PREDICTION PIPELINE")
    print("==========================================")

    print("\nPrediction period:")
    print(
        f"{first_prediction_week.date()} "
        f"to "
        f"{pd.Timestamp(scored_weeks[-1]).date()}"
    )

    print("\nBuilding historical training dataset...")

    full_training_dataset = build_training_dataset(
        data_dir
    )

    # Use only labels strictly before the first
    # prediction week.
    training_dataset = full_training_dataset[
        full_training_dataset["week_start"]
        < first_prediction_week
    ].copy()

    if training_dataset.empty:
        raise ValueError(
            "No historical training rows exist before "
            "the requested prediction period"
        )

    print(
        "Full labelled dataset shape:",
        full_training_dataset.shape,
    )

    print(
        "Leakage-safe training dataset shape:",
        training_dataset.shape,
    )

    model, feature_columns = train_final_model(
        training_dataset
    )

    all_predictions = []

    for week_start in scored_weeks:

        week_start = pd.Timestamp(week_start)

        if week_start.tzinfo is None:
            week_start = week_start.tz_localize("UTC")
        else:
            week_start = week_start.tz_convert("UTC")

        print(
            f"\nGenerating predictions for "
            f"{week_start.date()}..."
        )

        features = build_prediction_features(
            data_dir,
            week_start,
        )

        print(
            "Candidate gateways:",
            features["gateway_id"].nunique(),
        )

        prediction = predict_week(
            model,
            feature_columns,
            features,
            week_start,
        )

        print(
            "Selected:",
            len(prediction),
            "gateways",
        )

        print(
            prediction[
                [
                    "rank",
                    "gateway_id",
                    "score",
                ]
            ].to_string(index=False)
        )

        all_predictions.append(prediction)

    predictions = pd.concat(
        all_predictions,
        ignore_index=True,
    )

    return predictions


def validate_predictions(
    predictions,
    expected_weeks,
):
    """
    Validate the structural requirements of the generated
    prediction file.
    """

    expected_columns = [
        "week_start",
        "rank",
        "gateway_id",
        "score",
        "reason",
    ]

    if list(predictions.columns) != expected_columns:
        raise ValueError(
            "Unexpected prediction columns: "
            f"{list(predictions.columns)}"
        )

    expected_week_count = len(expected_weeks)

    expected_row_count = (
        expected_week_count * VISITS_PER_WEEK
    )

    if len(predictions) != expected_row_count:
        raise ValueError(
            f"Expected {expected_row_count} prediction rows, "
            f"found {len(predictions)}"
        )

    if (
        predictions["week_start"].nunique()
        != expected_week_count
    ):
        raise ValueError(
            f"Expected predictions for exactly "
            f"{expected_week_count} weeks"
        )

    expected_week_strings = {
        pd.Timestamp(week).date().isoformat()
        for week in expected_weeks
    }

    actual_week_strings = set(
        predictions["week_start"].astype(str)
    )

    if actual_week_strings != expected_week_strings:
        raise ValueError(
            "Prediction weeks do not match the "
            "requested prediction period"
        )

    for week, group in predictions.groupby(
        "week_start"
    ):

        if len(group) != VISITS_PER_WEEK:
            raise ValueError(
                f"{week}: expected "
                f"{VISITS_PER_WEEK} rows, "
                f"found {len(group)}"
            )

        if (
            group["gateway_id"].nunique()
            != VISITS_PER_WEEK
        ):
            raise ValueError(
                f"{week}: duplicate gateway selected"
            )

        expected_ranks = list(
            range(1, VISITS_PER_WEEK + 1)
        )

        if sorted(
            group["rank"].tolist()
        ) != expected_ranks:
            raise ValueError(
                f"{week}: ranks are not exactly 1-15"
            )

    if predictions["score"].isna().any():
        raise ValueError(
            "Prediction scores contain NaN values"
        )


def parse_arguments():
    """
    Parse optional prediction-period arguments.

    Running without arguments keeps the required Part 1
    prediction period.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Generate ML gateway rankings for a "
            "specified weekly prediction period."
        )
    )

    parser.add_argument(
        "--start-week",
        default=DEFAULT_START_WEEK,
        help=(
            "First prediction week in YYYY-MM-DD format. "
            "Default: 2026-02-02"
        ),
    )

    parser.add_argument(
        "--weeks",
        type=int,
        default=DEFAULT_NUM_WEEKS,
        help=(
            "Number of weekly prediction periods. "
            "Default: 8"
        ),
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    repo_root = Path(__file__).resolve().parent.parent

    data_dir = repo_root / "data"

    output_path = repo_root / "predictions.csv"

    if not data_dir.exists():
        raise FileNotFoundError(
            f"Data directory not found: {data_dir}"
        )

    scored_weeks = get_scored_weeks(
        start_week=args.start_week,
        num_weeks=args.weeks,
    )

    predictions = build_predictions(
        data_dir,
        scored_weeks=scored_weeks,
    )

    validate_predictions(
        predictions,
        expected_weeks=scored_weeks,
    )

    predictions.to_csv(
        output_path,
        index=False,
    )

    print("\n==========================================")
    print("FINAL PREDICTIONS CREATED")
    print("==========================================")

    print(
        f"\nWrote: {output_path}"
    )

    print(
        "Rows:",
        len(predictions),
    )

    print(
        "Weeks:",
        predictions["week_start"].nunique(),
    )

    print(
        "Gateways per week:",
        predictions.groupby(
            "week_start"
        )["gateway_id"].nunique().tolist(),
    )


if __name__ == "__main__":
    main()