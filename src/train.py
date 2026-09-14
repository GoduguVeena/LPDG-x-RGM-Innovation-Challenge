from pathlib import Path

import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from features import build_training_dataset


TARGET_THRESHOLD = 0.30


def main():
    data_dir = Path(__file__).resolve().parent.parent / "data"

    print("Building ML training dataset...")
    dataset = build_training_dataset(data_dir)

    print("\nDataset shape:", dataset.shape)

    # ---------------------------------------------------------
    # 1. Create classification target
    # ---------------------------------------------------------
    dataset["target"] = (
        dataset["failure_rate"] > TARGET_THRESHOLD
    ).astype(int)

    print(
        f"\nTarget: failure_rate > {TARGET_THRESHOLD:.0%}"
    )

    print("\nTarget distribution:")
    print(dataset["target"].value_counts())
    print(
        dataset["target"]
        .value_counts(normalize=True)
        .sort_index()
        .rename("percentage")
    )

    # ---------------------------------------------------------
    # 2. Sort chronologically
    # ---------------------------------------------------------
    dataset = dataset.sort_values(
        ["week_start", "gateway_id"]
    ).reset_index(drop=True)

    weeks = sorted(dataset["week_start"].unique())

    print("\nWeeks available:", len(weeks))
    print(
        "First week:",
        pd.Timestamp(weeks[0]).strftime("%Y-%m-%d")
    )
    print(
        "Last week:",
        pd.Timestamp(weeks[-1]).strftime("%Y-%m-%d")
    )

    # ---------------------------------------------------------
    # 3. Temporal train/validation split
    # ---------------------------------------------------------
    #
    # Earlier weeks -> training
    # Later weeks  -> validation
    #
    # This prevents future information from entering training.
    # ---------------------------------------------------------

    validation_weeks = 8

    train_weeks = weeks[:-validation_weeks]
    validation_weeks_list = weeks[-validation_weeks:]

    train_data = dataset[
        dataset["week_start"].isin(train_weeks)
    ].copy()

    validation_data = dataset[
        dataset["week_start"].isin(validation_weeks_list)
    ].copy()

    print("\nTemporal split:")
    print(
        "Training:",
        pd.Timestamp(train_weeks[0]).strftime("%Y-%m-%d"),
        "to",
        pd.Timestamp(train_weeks[-1]).strftime("%Y-%m-%d"),
    )

    print(
        "Validation:",
        pd.Timestamp(validation_weeks_list[0]).strftime("%Y-%m-%d"),
        "to",
        pd.Timestamp(validation_weeks_list[-1]).strftime("%Y-%m-%d"),
    )

    print("\nTraining rows:", len(train_data))
    print("Validation rows:", len(validation_data))

    # ---------------------------------------------------------
    # 4. Select features
    # ---------------------------------------------------------

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

    X_train = train_data[feature_columns]
    y_train = train_data["target"]

    X_validation = validation_data[feature_columns]
    y_validation = validation_data["target"]

    print("\nNumber of features:", len(feature_columns))

    # ---------------------------------------------------------
    # 5. Build ML pipeline
    # ---------------------------------------------------------
    #
    # Missing values:
    #     median imputation
    #
    # Scaling:
    #     standardization
    #
    # Model:
    #     Logistic Regression
    #
    # All preprocessing is fitted ONLY on training data.
    # ---------------------------------------------------------

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

    print("\nTraining Logistic Regression...")
    model.fit(X_train, y_train)

    # ---------------------------------------------------------
    # 6. Generate validation probabilities
    # ---------------------------------------------------------

    validation_probability = model.predict_proba(
        X_validation
    )[:, 1]

    validation_prediction = (
        validation_probability >= 0.5
    ).astype(int)

    # ---------------------------------------------------------
    # 7. Evaluate classification performance
    # ---------------------------------------------------------

    print("\n==============================")
    print("VALIDATION RESULTS")
    print("==============================")

    print("\nROC-AUC:")
    print(
        f"{roc_auc_score(y_validation, validation_probability):.4f}"
    )

    print("\nClassification report:")
    print(
        classification_report(
            y_validation,
            validation_prediction,
            digits=4,
            zero_division=0,
        )
    )

    # ---------------------------------------------------------
    # 8. Show most influential features
    # ---------------------------------------------------------

    classifier = model.named_steps["classifier"]

    coefficients = pd.Series(
        classifier.coef_[0],
        index=feature_columns,
    ).sort_values()

    print("\nMost negative features:")
    print(coefficients.head(10))

    print("\nMost positive features:")
    print(coefficients.tail(10))

    print("\nML training experiment complete.")


if __name__ == "__main__":
    main()