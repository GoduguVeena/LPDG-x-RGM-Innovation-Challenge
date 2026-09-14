from pathlib import Path

from features import build_training_dataset


def main():
    data_dir = Path(__file__).resolve().parent.parent / "data"

    print("Building ML training dataset...")
    
    dataset = build_training_dataset(data_dir)

    print("\nDataset shape:", dataset.shape)

    print("\nTarget summary:")
    print(dataset["failure_rate"].describe())

    print("\nMissing values:")
    missing = dataset.isna().sum()
    print(missing[missing > 0].sort_values(ascending=False).head(20))

    print("\nFailure-rate thresholds:")
    for threshold in [0.10, 0.20, 0.30]:
        percentage = (dataset["failure_rate"] > threshold).mean() * 100
        count = (dataset["failure_rate"] > threshold).sum()

        print(
            f"> {threshold:.0%}: "
            f"{count} rows ({percentage:.2f}%)"
        )

    print("\nWeeks represented:")
    print(
        dataset["week_start"]
        .sort_values()
        .drop_duplicates()
        .dt.strftime("%Y-%m-%d")
        .to_list()
    )

    print("\nTop feature correlations with next-week failure rate:")

    numeric = dataset.select_dtypes(include="number")

    correlations = (
        numeric.corr()["failure_rate"]
        .drop("failure_rate")
        .abs()
        .sort_values(ascending=False)
    )

    for feature, value in correlations.head(20).items():
        signed_corr = numeric[feature].corr(
            numeric["failure_rate"]
        )

        print(
            f"{feature:45s} "
            f"{signed_corr:+.4f}"
        )


if __name__ == "__main__":
    main()