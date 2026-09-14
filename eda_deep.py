from pathlib import Path
import pandas as pd

DATA_DIR = Path("data")


def normalize_gateway_id(value):
    return str(value).replace(":", "").replace("-", "").strip().upper()


def load_telemetry():
    files = sorted((DATA_DIR / "telemetry").glob("month=*/part-*.parquet"))

    frames = []

    for file in files:
        df = pd.read_parquet(file)
        df["gateway_id_norm"] = df["gateway_id"].map(normalize_gateway_id)
        df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True)
        frames.append(
            df[
                [
                    "gateway_id_norm",
                    "ts_utc",
                    "offline_duration_sec",
                    "disconnection_cnt",
                    "reboot_cnt",
                    "online_duration_mins",
                ]
            ]
        )

    telemetry = pd.concat(frames, ignore_index=True)

    # Remove exact gateway + timestamp duplicates for analysis only.
    telemetry = telemetry.drop_duplicates(
        subset=["gateway_id_norm", "ts_utc"]
    )

    return telemetry


def main():
    print("=" * 70)
    print("METER READ SUCCESS vs TELEMETRY")
    print("=" * 70)

    # ---------------------------------------------------------
    # 1. Load meter-read data
    # ---------------------------------------------------------
    meter = pd.read_csv(DATA_DIR / "meter_read_success.csv")

    meter["gateway_id_norm"] = meter["gateway_id"].map(
        normalize_gateway_id
    )

    meter["week_start"] = pd.to_datetime(
    meter["week_start"],
    utc=True
)

    meter["read_rate"] = (
        meter["meters_read"] / meter["meters_expected"]
    )

    meter["failure_rate"] = 1 - meter["read_rate"]

    print("\nMeter-read data:")
    print(f"Rows: {len(meter):,}")
    print(f"Gateways: {meter['gateway_id_norm'].nunique():,}")
    print(
        f"Weeks: {meter['week_start'].min().date()} "
        f"to {meter['week_start'].max().date()}"
    )

    print("\nRead-rate summary:")
    print(
        meter["read_rate"]
        .describe()
        .to_string()
    )

    print("\nFailure-rate summary:")
    print(
        meter["failure_rate"]
        .describe()
        .to_string()
    )

    # ---------------------------------------------------------
    # 2. Load telemetry
    # ---------------------------------------------------------
    print("\nLoading telemetry...")
    telemetry = load_telemetry()

    print(
        f"Telemetry rows after exact deduplication: "
        f"{len(telemetry):,}"
    )

    # ---------------------------------------------------------
    # 3. Convert telemetry into weekly features
    # ---------------------------------------------------------
    telemetry["week_start"] = (
        telemetry["ts_utc"]
        .dt.floor("D")
        - pd.to_timedelta(
            telemetry["ts_utc"].dt.dayofweek,
            unit="D"
        )
    )

    weekly = (
        telemetry
        .groupby(["gateway_id_norm", "week_start"])
        .agg(
            telemetry_hours=(
                "ts_utc",
                "count"
            ),
            avg_offline_duration_sec=(
                "offline_duration_sec",
                "mean"
            ),
            avg_disconnection_cnt=(
                "disconnection_cnt",
                "mean"
            ),
            total_reboot_cnt=(
                "reboot_cnt",
                "sum"
            ),
            avg_online_duration_mins=(
                "online_duration_mins",
                "mean"
            ),
        )
        .reset_index()
    )

    # ---------------------------------------------------------
    # 4. Join weekly telemetry with meter performance
    # ---------------------------------------------------------
    merged = meter.merge(
        weekly,
        on=["gateway_id_norm", "week_start"],
        how="left",
    )

    print("\nJoin results:")
    print(f"Meter rows: {len(meter):,}")
    print(f"Merged rows: {len(merged):,}")

    matched = merged["telemetry_hours"].notna().sum()

    print(
        f"Rows with telemetry: {matched:,} "
        f"({matched / len(merged) * 100:.2f}%)"
    )

    print(
        f"Rows without telemetry: "
        f"{len(merged) - matched:,}"
    )

    # ---------------------------------------------------------
    # 5. Correlation with meter-read failure
    # ---------------------------------------------------------
    analysis = merged.dropna(
        subset=[
            "failure_rate",
            "avg_offline_duration_sec",
            "avg_disconnection_cnt",
            "total_reboot_cnt",
        ]
    )

    features = [
        "avg_offline_duration_sec",
        "avg_disconnection_cnt",
        "total_reboot_cnt",
        "avg_online_duration_mins",
        "telemetry_hours",
    ]

    print("\nCorrelation with meter-read failure rate:")
    print("-" * 70)

    for feature in features:
        corr = analysis[feature].corr(
            analysis["failure_rate"]
        )

        print(
            f"{feature:35s}: {corr: .4f}"
        )

    # ---------------------------------------------------------
    # 6. Compare low vs high meter failure
    # ---------------------------------------------------------
    print("\nFailure-rate groups:")
    print("-" * 70)

    analysis["failure_group"] = pd.cut(
        analysis["failure_rate"],
        bins=[-0.001, 0.01, 0.10, 1.0],
        labels=[
            "Low failure (<=1%)",
            "Medium failure (1-10%)",
            "High failure (>10%)",
        ],
    )

    group_summary = (
        analysis
        .groupby("failure_group", observed=False)
        [
            [
                "failure_rate",
                "avg_offline_duration_sec",
                "avg_disconnection_cnt",
                "total_reboot_cnt",
                "telemetry_hours",
            ]
        ]
        .mean()
    )

    print(group_summary.to_string())

    # ---------------------------------------------------------
    # 7. Worst gateways by meter failure
    # ---------------------------------------------------------
    print("\nWorst gateway-weeks by meter failure:")
    print("-" * 70)

    worst = (
        analysis[
            [
                "week_start",
                "gateway_id_norm",
                "meters_expected",
                "meters_read",
                "read_rate",
                "failure_rate",
                "avg_offline_duration_sec",
                "avg_disconnection_cnt",
                "total_reboot_cnt",
            ]
        ]
        .sort_values(
            ["failure_rate", "meters_expected"],
            ascending=[False, False],
        )
        .head(20)
    )

    print(worst.to_string(index=False))

    # ---------------------------------------------------------
    # 8. Coverage by week
    # ---------------------------------------------------------
    print("\nTelemetry coverage by meter-read week:")
    print("-" * 70)

    coverage = (
        merged
        .assign(has_telemetry=merged["telemetry_hours"].notna())
        .groupby("week_start")
        .agg(
            meter_rows=("gateway_id_norm", "size"),
            telemetry_matched=("has_telemetry", "sum"),
        )
    )

    coverage["match_pct"] = (
        coverage["telemetry_matched"]
        / coverage["meter_rows"]
        * 100
    )

    print(coverage.to_string())

    print("\n" + "=" * 70)
    print("EDA COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()