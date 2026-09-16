from pathlib import Path

import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from features import build_training_dataset

# Configuration

TARGET_THRESHOLD = 0.30


# Main training pipeline

def main():

    repo_root = Path(__file__).resolve().parent.parent

    data_dir = repo_root / "data"
    models_dir = repo_root / "models"

    print("=" * 60)
    print("FINAL ML TRAINING PIPELINE")
    print("=" * 60)

    # Build training dataset

    print("\nBuilding ML training dataset...")

    dataset = build_training_dataset(data_dir)

    print("\nDataset shape:", dataset.shape)

    # Create classification target

    dataset["target"] = (
        dataset["failure_rate"] > TARGET_THRESHOLD
    ).astype(int)

    print(
        f"\nTarget: failure_rate > {TARGET_THRESHOLD:.0%}"
    )

    print("\nTarget distribution:")
    print(dataset["target"].value_counts())

    print(
        "\nTarget percentage:"
    )

    print(
        dataset["target"]
        .value_counts(normalize=True)
        .sort_index()
        .rename("percentage")
    )

    # Sort chronologically

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

    print("\nWeeks available:", len(weeks))

    print(
        "First week:",
        pd.Timestamp(
            weeks[0]
        ).strftime("%Y-%m-%d")
    )

    print(
        "Last week:",
        pd.Timestamp(
            weeks[-1]
        ).strftime("%Y-%m-%d")
    )

    # Use earlier weeks for training and later weeks for validation.

    validation_weeks_count = 8

    if len(weeks) <= validation_weeks_count:
        raise ValueError(
            "Not enough historical weeks for "
            "an 8-week validation period."
        )

    train_weeks = weeks[
        :-validation_weeks_count
    ]

    validation_weeks_list = weeks[
        -validation_weeks_count:
    ]

    train_data = dataset[
        dataset["week_start"].isin(
            train_weeks
        )
    ].copy()

    validation_data = dataset[
        dataset["week_start"].isin(
            validation_weeks_list
        )
    ].copy()

    print("\nTemporal split:")

    print(
        "Training:",
        pd.Timestamp(
            train_weeks[0]
        ).strftime("%Y-%m-%d"),
        "to",
        pd.Timestamp(
            train_weeks[-1]
        ).strftime("%Y-%m-%d"),
    )

    print(
        "Validation:",
        pd.Timestamp(
            validation_weeks_list[0]
        ).strftime("%Y-%m-%d"),
        "to",
        pd.Timestamp(
            validation_weeks_list[-1]
        ).strftime("%Y-%m-%d"),
    )

    print(
        "\nTraining rows:",
        len(train_data)
    )

    print(
        "Validation rows:",
        len(validation_data)
    )

    # Select model features

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

    X_train = train_data[
        feature_columns
    ]

    y_train = train_data[
        "target"
    ]

    X_validation = validation_data[
        feature_columns
    ]

    y_validation = validation_data[
        "target"
    ]

    print(
        "\nNumber of features:",
        len(feature_columns)
    )

    if len(feature_columns) != 69:
        raise ValueError(
            f"Expected 69 features, "
            f"but found {len(feature_columns)}."
        )

    # Preprocessing + Logistic Regression pipeline

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

    # Train model

    print(
        "\nTraining Logistic Regression..."
    )

    model.fit(
        X_train,
        y_train,
    )

    print(
        "Training complete."
    )

    # Save the complete trained pipeline for inference.

    models_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = (
        models_dir
        / "logistic_regression_model.joblib"
    )

    joblib.dump(
        model,
        model_path,
    )

    print(
        "\nSaved trained model to:"
    )

    print(
        model_path
    )

    # Generate validation probabilities

    validation_probability = (
        model.predict_proba(
            X_validation
        )[:, 1]
    )

    validation_prediction = (
        validation_probability >= 0.5
    ).astype(int)

    # Evaluate model

    print("\n" + "=" * 30)
    print("VALIDATION RESULTS")
    print("=" * 30)

    roc_auc = roc_auc_score(
        y_validation,
        validation_probability,
    )

    print("\nROC-AUC:")
    print(f"{roc_auc:.4f}")

    print("\nClassification report:")

    print(
        classification_report(
            y_validation,
            validation_prediction,
            digits=4,
            zero_division=0,
        )
    )

    # Show influential features

    classifier = (
        model.named_steps[
            "classifier"
        ]
    )

    coefficients = pd.Series(
        classifier.coef_[0],
        index=feature_columns,
    ).sort_values()

    print(
        "\nMost negative features:"
    )

    print(
        coefficients.head(10)
    )

    print(
        "\nMost positive features:"
    )

    print(
        coefficients.tail(10)
    )

    # Final summary

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    print(
        "\nModel:",
        "Logistic Regression"
    )

    print(
        "Features:",
        len(feature_columns)
    )

    print(
        "Training weeks:",
        len(train_weeks)
    )

    print(
        "Validation weeks:",
        len(validation_weeks_list)
    )

    print(
        "Model artifact:",
        model_path
    )

    print(
        "\nThe trained model is ready "
        "for inference by final_predict.py."
    )


# Entry point

if __name__ == "__main__":
    main()