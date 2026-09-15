from pathlib import Path
import glob

import numpy as np
import pandas as pd


TELEMETRY_FEATURES = [
    "offline_duration_sec",
    "disconnection_cnt",
    "reboot_cnt",
    "reboot_duration_sec",
    "online_duration_mins",
    "avg_load1",
    "avg_uptime",
    "rssi_bad",
    "rssi_good",
    "no_conn_importance",
    "reboot_importance",
]

def normalize_gateway_id(series):
    """
    Normalize gateway IDs so formats such as
    0639EA5602C1 and 06:39:EA:56:02:C1
    are treated as the same gateway.
    """
    return (
        series
        .astype(str)
        .str.replace(":", "", regex=False)
        .str.replace("-", "", regex=False)
        .str.strip()
        .str.upper()
    )

def load_telemetry(data_dir):
    """Load and combine all monthly telemetry partitions."""
    pattern = str(Path(data_dir) / "telemetry" / "month=*" / "part-0.parquet")
    files = sorted(glob.glob(pattern))

    if not files:
        raise FileNotFoundError(
            f"No telemetry parquet files found under {data_dir}/telemetry"
        )

    frames = [pd.read_parquet(path) for path in files]
    telemetry = pd.concat(frames, ignore_index=True)

    telemetry["ts_utc"] = pd.to_datetime(
        telemetry["ts_utc"], utc=True
    )

    telemetry["gateway_id"] = normalize_gateway_id(
    telemetry["gateway_id"]
)

    # Duplicate telemetry rows exist in the source.
    # Remove exact duplicates for feature calculation.
    telemetry = telemetry.drop_duplicates()

    return telemetry


def load_meter_read_success(data_dir):
    """Load weekly meter-read outcomes and derive failure rate."""
    path = Path(data_dir) / "meter_read_success.csv"

    meter = pd.read_csv(path)

    meter["gateway_id"] = normalize_gateway_id(
    meter["gateway_id"]
)

    meter["week_start"] = pd.to_datetime(
        meter["week_start"], utc=True
    )

    meter["failure_rate"] = 1 - (
        meter["meters_read"] / meter["meters_expected"]
    )

    return meter


def aggregate_window(telemetry, week_start, days):
    """
    Aggregate telemetry from the previous `days` days
    for every gateway.

    Only telemetry strictly before the target week is used.
    """
    start = week_start - pd.Timedelta(days=days)

    window = telemetry[
        (telemetry["ts_utc"] >= start)
        & (telemetry["ts_utc"] < week_start)
    ].copy()

    if window.empty:
        return pd.DataFrame()

    aggregations = {}

    for feature in TELEMETRY_FEATURES:
        aggregations[feature] = ["mean", "max"]

    result = window.groupby("gateway_id").agg(aggregations)

    result.columns = [
        f"{feature}_{stat}_{days}d"
        for feature, stat in result.columns
    ]

    # Number of telemetry observations is itself useful:
    # missing/silent telemetry can carry operational information.
    coverage = (
        window.groupby("gateway_id")
        .size()
        .rename(f"telemetry_hours_{days}d")
    )

    result = result.join(coverage)

    return result.reset_index()


def build_features_for_week(telemetry, week_start):
    """
    Build leakage-safe features for one target week.

    Features only use telemetry before `week_start`.
    """
    week_start = pd.Timestamp(week_start)

    result = None

    for days in (7, 14, 28):
        window_features = aggregate_window(
            telemetry,
            week_start,
            days,
        )

        if window_features.empty:
            continue

        if result is None:
            result = window_features
        else:
            result = result.merge(
                window_features,
                on="gateway_id",
                how="outer",
            )

    if result is None:
        return pd.DataFrame(
            columns=["gateway_id", "week_start"]
        )

    result["week_start"] = week_start

    return result

def build_training_dataset(data_dir):
    """
    Build one row per gateway-week.

    Features:
        telemetry from the past 7/14/28 days

    Target:
        failure rate during the following meter-read week
    """
    telemetry = load_telemetry(data_dir)
    meter = load_meter_read_success(data_dir)

    datasets = []

    for week_start in sorted(meter["week_start"].unique()):
        features = build_features_for_week(
            telemetry,
            week_start,
        )

        targets = meter[
            meter["week_start"] == week_start
        ][
            [
                "gateway_id",
                "failure_rate",
            ]
        ]

        merged = features.merge(
            targets,
            on="gateway_id",
            how="inner",
        )

        datasets.append(merged)

    dataset = pd.concat(
        datasets,
        ignore_index=True,
    )

    return dataset
def build_prediction_features(data_dir, week_start):
    """
    Build ML features for prediction for one target week.

    Only telemetry strictly before `week_start` is used.
    The gateway universe comes from gateway_master.csv so that
    gateways with missing/quiet telemetry do not disappear.
    """
    telemetry = load_telemetry(data_dir)

    week_start = pd.Timestamp(week_start)

    if week_start.tzinfo is None:
        week_start = week_start.tz_localize("UTC")
    else:
        week_start = week_start.tz_convert("UTC")

    # Build the same 69 features used during training.
    features = build_features_for_week(
        telemetry,
        week_start,
    )

    # Use gateway_master as the candidate gateway universe.
    master_path = Path(data_dir) / "gateway_master.csv"
    master = pd.read_csv(
    master_path,
    encoding="latin1",
)

    if "gateway_id" not in master.columns:
        raise ValueError(
            "gateway_master.csv must contain a gateway_id column"
        )

    gateways = master[["gateway_id"]].drop_duplicates().copy()

    gateways["gateway_id"] = normalize_gateway_id(
    gateways["gateway_id"]
)

    # Keep every gateway, including gateways with no telemetry
    # in the prediction window.
    features = gateways.merge(
        features,
        on="gateway_id",
        how="left",
    )

    features["week_start"] = week_start

    return features