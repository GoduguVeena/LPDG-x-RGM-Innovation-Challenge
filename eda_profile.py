from pathlib import Path
import pandas as pd


DATA_DIR = Path("data")


def profile_csv(path):
    print("\n" + "=" * 70)
    print(f"FILE: {path.name}")
    print("=" * 70)

    df = pd.read_csv(path, encoding="latin1")

    print(f"Shape: {df.shape}")
    print("\nColumns:")
    for col in df.columns:
        print(f"  - {col}: {df[col].dtype}")

    print("\nMissing values:")
    missing = df.isna().sum()
    for col, count in missing.items():
        if count > 0:
            print(f"  - {col}: {count} ({count / len(df) * 100:.2f}%)")

    print("\nFirst 5 rows:")
    print(df.head().to_string(index=False))

    return df


def profile_telemetry():
    print("\n" + "=" * 70)
    print("TELEMETRY")
    print("=" * 70)

    telemetry_path = DATA_DIR / "telemetry"

    files = sorted(telemetry_path.glob("month=*/part-*.parquet"))

    print(f"Monthly partitions found: {len(files)}")

    for file in files:
        df = pd.read_parquet(file)

        print(
            f"{file.parent.name}: "
            f"{len(df):,} rows, "
            f"{len(df.columns)} columns"
        )

        print("  Columns:", ", ".join(df.columns))

        print("  Missing values:")
        missing = df.isna().sum()

        for col, count in missing.items():
            if count > 0:
                print(
                    f"    {col}: {count:,} "
                    f"({count / len(df) * 100:.2f}%)"
                )


def main():
    profile_telemetry()

    profile_csv(DATA_DIR / "meter_read_success.csv")
    profile_csv(DATA_DIR / "field_visits.csv")
    profile_csv(DATA_DIR / "gateway_master.csv")

    print("\n" + "=" * 70)
    print("ENGINEER REVIEW")
    print("=" * 70)

    review_path = DATA_DIR / "engineer_review_2026-02.xlsx"
    review = pd.read_excel(review_path)

    print(f"Shape: {review.shape}")

    print("\nColumns:")
    for col in review.columns:
        print(f"  - {col}: {review[col].dtype}")

    print("\nFirst 5 rows:")
    print(review.head().to_string(index=False))


if __name__ == "__main__":
    main()