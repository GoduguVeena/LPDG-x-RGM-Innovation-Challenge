from pathlib import Path

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

SCORED_WEEKS = pd.date_range(
    start="2026-02-02",
    periods=8,
    freq="7D",
    tz="UTC",
)


def train_final_model(dataset):
    """
    Train the final ML model using all available historical
    labelled gateway-weeks.
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
            reasons.append("limited recent telemetry coverage")

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
    # 1. highest risk probability
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

    selected["week_start"] = week_start.date().isoformat()

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


def build_predictions(data_dir):
    """
    Train on historical labelled data and generate
    predictions for all eight scored weeks.
    """

    print("==========================================")
    print("FINAL ML PREDICTION PIPELINE")
    print("==========================================")

    print("\nBuilding historical training dataset...")

    training_dataset = build_training_dataset(
        data_dir
    )

    print(
        "Training dataset shape:",
        training_dataset.shape,
    )

    model, feature_columns = train_final_model(
        training_dataset
    )

    all_predictions = []

    for week_start in SCORED_WEEKS:

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


def main():
    repo_root = Path(__file__).resolve().parent.parent

    data_dir = repo_root / "data"

    output_path = repo_root / "predictions.csv"

    if not data_dir.exists():
        raise FileNotFoundError(
            f"Data directory not found: {data_dir}"
        )

    predictions = build_predictions(
        data_dir
    )

    # ---------------------------------------------------------
    # Final structural checks
    # ---------------------------------------------------------

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

    if len(predictions) != 120:
        raise ValueError(
            f"Expected 120 prediction rows, "
            f"found {len(predictions)}"
        )

    if predictions["week_start"].nunique() != 8:
        raise ValueError(
            "Expected predictions for exactly 8 weeks"
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

        if group["gateway_id"].nunique() != VISITS_PER_WEEK:
            raise ValueError(
                f"{week}: duplicate gateway selected"
            )

        expected_ranks = list(
            range(1, VISITS_PER_WEEK + 1)
        )

        if sorted(group["rank"].tolist()) != expected_ranks:
            raise ValueError(
                f"{week}: ranks are not exactly 1-15"
            )

    if predictions["score"].isna().any():
        raise ValueError(
            "Prediction scores contain NaN values"
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