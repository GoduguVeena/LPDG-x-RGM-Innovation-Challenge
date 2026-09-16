import argparse
from pathlib import Path

import joblib
import pandas as pd

from features import build_prediction_features

VISITS_PER_WEEK = 15

# Required Part 1 prediction period
DEFAULT_START_WEEK = "2026-02-02"
DEFAULT_NUM_WEEKS = 8

# The trained model was built with 69 engineered features
EXPECTED_FEATURES = 69


# ============================================================
# Prediction weeks
# ============================================================

def get_scored_weeks(
    start_week=DEFAULT_START_WEEK,
    num_weeks=DEFAULT_NUM_WEEKS,
):
    """
    Return the weekly prediction periods.

    Default:
        2026-02-02
        2026-02-09
        2026-02-16
        2026-02-23
        2026-03-02
        2026-03-09
        2026-03-16
        2026-03-23
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


# ============================================================
# Load trained model
# ============================================================

def load_final_model(repo_root):
    """
    Load the already-trained ML pipeline.

    IMPORTANT:
    This function ONLY loads the saved model.

    There is NO:
        model.fit()
        build_training_dataset()
        joblib.dump()

    during prediction.
    """

    model_path = (
        repo_root
        / "models"
        / "logistic_regression_model.joblib"
    )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found: {model_path}\n"
            "Run train_final.py first to create the model artifact."
        )

    print("\nLoading trained Logistic Regression model...")

    model = joblib.load(model_path)

    print("Model loaded successfully.")

    return model


# ============================================================
# Feature validation
# ============================================================

def get_feature_columns(model, prediction_features):
    """
    Determine the exact feature columns expected by the
    trained model and make sure prediction data contains them.

    The training pipeline was created using 69 engineered
    features.
    """

    # Prefer the feature names stored by scikit-learn when
    # the model was trained using a pandas DataFrame.
    if hasattr(model, "feature_names_in_"):
        feature_columns = list(model.feature_names_in_)

    else:
        excluded_columns = {
            "gateway_id",
            "week_start",
            "failure_rate",
            "target",
            "score",
            "reason",
            "rank",
        }

        feature_columns = [
            column
            for column in prediction_features.columns
            if column not in excluded_columns
        ]

    if len(feature_columns) != EXPECTED_FEATURES:
        raise ValueError(
            f"Expected {EXPECTED_FEATURES} model features, "
            f"found {len(feature_columns)}"
        )

    missing_features = [
        column
        for column in feature_columns
        if column not in prediction_features.columns
    ]

    if missing_features:
        raise ValueError(
            "Prediction data is missing required features: "
            + ", ".join(missing_features)
        )

    return feature_columns


# ============================================================
# Human-readable prediction reason
# ============================================================

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
        return (
            "Risk score based on available gateway telemetry"
        )

    return (
        "Elevated predicted severe-failure risk based on "
        + ", ".join(reasons[:3])
    )


# ============================================================
# Predict one week
# ============================================================

def predict_week(
    model,
    prediction_features,
    week_start,
):
    """
    Generate the ranked top 15 gateways for one prediction week.

    IMPORTANT:
    This function only performs inference.

    It never trains or fits the model.
    """

    prediction_features = prediction_features.copy()

    # Get the exact feature order expected by the trained model.
    feature_columns = get_feature_columns(
        model,
        prediction_features,
    )

    # Arrange prediction columns in exactly the same order
    # used when training the model.
    X_prediction = prediction_features[
        feature_columns
    ]

    # --------------------------------------------------------
    # INFERENCE ONLY
    # --------------------------------------------------------

    probabilities = model.predict_proba(
        X_prediction
    )[:, 1]

    prediction_features["score"] = probabilities

    # --------------------------------------------------------
    # Deterministic ranking
    #
    # 1. Highest predicted severe-failure risk
    # 2. gateway_id alphabetically for ties
    # --------------------------------------------------------

    prediction_features = prediction_features.sort_values(
        ["score", "gateway_id"],
        ascending=[False, True],
    ).reset_index(drop=True)

    # Select top 15 gateways
    selected = prediction_features.head(
        VISITS_PER_WEEK
    ).copy()

    if len(selected) != VISITS_PER_WEEK:
        raise ValueError(
            f"Expected {VISITS_PER_WEEK} gateways for "
            f"{week_start.date()}, "
            f"found {len(selected)}"
        )

    # Assign ranks 1-15
    selected["rank"] = range(
        1,
        VISITS_PER_WEEK + 1,
    )

    # Store prediction week
    selected["week_start"] = (
        week_start.date().isoformat()
    )

    # Generate explanation
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


# ============================================================
# Build predictions for all weeks
# ============================================================

def build_predictions(
    data_dir,
    scored_weeks=None,
):
    """
    Generate predictions for all requested weeks.

    IMPORTANT:
    No training is performed here.

    Workflow:

        Load saved model
              ↓
        Build prediction features
              ↓
        model.predict_proba()
              ↓
        Rank gateways
              ↓
        Select top 15
    """

    if scored_weeks is None:
        scored_weeks = get_scored_weeks()

    scored_weeks = list(scored_weeks)

    if not scored_weeks:
        raise ValueError(
            "At least one prediction week is required"
        )

    repo_root = Path(__file__).resolve().parent.parent

    first_prediction_week = pd.Timestamp(
        scored_weeks[0]
    )

    last_prediction_week = pd.Timestamp(
        scored_weeks[-1]
    )

    print("\n==========================================")
    print("FINAL ML PREDICTION PIPELINE")
    print("==========================================")

    print("\nPrediction period:")
    print(
        f"{first_prediction_week.date()} "
        f"to "
        f"{last_prediction_week.date()}"
    )

    print("\nTraining is NOT performed during prediction.")
    print("Loading pre-trained model...")

    # --------------------------------------------------------
    # Load the already-trained model
    # --------------------------------------------------------

    model = load_final_model(repo_root)

    # --------------------------------------------------------
    # Validate loaded model
    # --------------------------------------------------------

    if not hasattr(model, "n_features_in_"):
        raise ValueError(
            "Loaded model does not expose n_features_in_. "
            "Expected the saved scikit-learn pipeline."
        )

    print(
        "Loaded model feature count:",
        model.n_features_in_,
    )

    if model.n_features_in_ != EXPECTED_FEATURES:
        raise ValueError(
            f"Expected model with "
            f"{EXPECTED_FEATURES} features, "
            f"found {model.n_features_in_}"
        )

    all_predictions = []

    # ========================================================
    # Generate predictions week by week
    # ========================================================

    for week_start in scored_weeks:

        week_start = pd.Timestamp(week_start)

        if week_start.tzinfo is None:
            week_start = week_start.tz_localize("UTC")
        else:
            week_start = week_start.tz_convert("UTC")

        print("\n------------------------------------------")
        print(
            f"Generating predictions for "
            f"{week_start.date()}..."
        )
        print("------------------------------------------")

        # ----------------------------------------------------
        # Build ONLY prediction features
        #
        # No training dataset is created.
        # ----------------------------------------------------

        features = build_prediction_features(
            data_dir,
            week_start,
        )

        if features.empty:
            raise ValueError(
                f"No prediction features generated for "
                f"{week_start.date()}"
            )

        print(
            "Candidate gateways:",
            features["gateway_id"].nunique(),
        )

        # ----------------------------------------------------
        # Generate predictions
        # ----------------------------------------------------

        prediction = predict_week(
            model,
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

    # ========================================================
    # Combine weekly predictions
    # ========================================================

    predictions = pd.concat(
        all_predictions,
        ignore_index=True,
    )

    return predictions


# ============================================================
# Validate final prediction dataframe
# ============================================================

def validate_predictions(
    predictions,
    expected_weeks,
):
    """
    Validate structural requirements of the final
    predictions dataframe.
    """

    expected_columns = [
        "week_start",
        "rank",
        "gateway_id",
        "score",
        "reason",
    ]

    # --------------------------------------------------------
    # Column validation
    # --------------------------------------------------------

    if list(predictions.columns) != expected_columns:
        raise ValueError(
            "Unexpected prediction columns: "
            f"{list(predictions.columns)}"
        )

    # --------------------------------------------------------
    # Row count validation
    # --------------------------------------------------------

    expected_week_count = len(expected_weeks)

    expected_row_count = (
        expected_week_count
        * VISITS_PER_WEEK
    )

    if len(predictions) != expected_row_count:
        raise ValueError(
            f"Expected {expected_row_count} prediction rows, "
            f"found {len(predictions)}"
        )

    # --------------------------------------------------------
    # Week count validation
    # --------------------------------------------------------

    if (
        predictions["week_start"].nunique()
        != expected_week_count
    ):
        raise ValueError(
            f"Expected predictions for exactly "
            f"{expected_week_count} weeks"
        )

    # --------------------------------------------------------
    # Exact week validation
    # --------------------------------------------------------

    expected_week_strings = {
        pd.Timestamp(week)
        .date()
        .isoformat()
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

    # --------------------------------------------------------
    # Validate every week's rankings
    # --------------------------------------------------------

    for week, group in predictions.groupby(
        "week_start"
    ):

        # Exactly 15 gateways
        if len(group) != VISITS_PER_WEEK:
            raise ValueError(
                f"{week}: expected "
                f"{VISITS_PER_WEEK} rows, "
                f"found {len(group)}"
            )

        # No duplicate gateway
        if (
            group["gateway_id"].nunique()
            != VISITS_PER_WEEK
        ):
            raise ValueError(
                f"{week}: duplicate gateway selected"
            )

        # Ranks must be exactly 1-15
        expected_ranks = list(
            range(
                1,
                VISITS_PER_WEEK + 1,
            )
        )

        actual_ranks = sorted(
            group["rank"].tolist()
        )

        if actual_ranks != expected_ranks:
            raise ValueError(
                f"{week}: ranks are not exactly 1-15"
            )

    # --------------------------------------------------------
    # Score validation
    # --------------------------------------------------------

    if predictions["score"].isna().any():
        raise ValueError(
            "Prediction scores contain NaN values"
        )

    print("\nPrediction validation: PASSED")


# ============================================================
# Command-line arguments
# ============================================================

def parse_arguments():
    """
    Parse optional prediction-period arguments.

    Running without arguments generates the required
    Part 1 period.
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


# ============================================================
# Main
# ============================================================

def main():

    args = parse_arguments()

    # Repository root
    repo_root = (
        Path(__file__)
        .resolve()
        .parent
        .parent
    )

    # Input data directory
    data_dir = repo_root / "data"

    # Final output file
    output_path = repo_root / "predictions.csv"

    # --------------------------------------------------------
    # Check data directory
    # --------------------------------------------------------

    if not data_dir.exists():
        raise FileNotFoundError(
            f"Data directory not found: {data_dir}"
        )

    # --------------------------------------------------------
    # Determine prediction weeks
    # --------------------------------------------------------

    scored_weeks = get_scored_weeks(
        start_week=args.start_week,
        num_weeks=args.weeks,
    )

    # --------------------------------------------------------
    # Generate predictions
    # --------------------------------------------------------

    predictions = build_predictions(
        data_dir,
        scored_weeks=scored_weeks,
    )

    # --------------------------------------------------------
    # Validate predictions
    # --------------------------------------------------------

    validate_predictions(
        predictions,
        expected_weeks=scored_weeks,
    )

    # --------------------------------------------------------
    # Save predictions.csv
    # --------------------------------------------------------

    predictions.to_csv(
        output_path,
        index=False,
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

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
        predictions
        .groupby("week_start")
        ["gateway_id"]
        .nunique()
        .tolist(),
    )

    print("\n==========================================")
    print("PIPELINE COMPLETE")
    print("==========================================")


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()