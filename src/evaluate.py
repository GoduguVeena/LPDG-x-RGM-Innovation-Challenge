from pathlib import Path

import pandas as pd

from features import load_meter_read_success


VISIT_COST = 380
FAILURE_COST_PER_WEEK = 600
VISITS_PER_WEEK = 15

TARGET_THRESHOLD = 0.30


def prepare_actual_data(data_dir):
    """
    Load historical meter-read outcomes and mark severe failures.
    """

    data = load_meter_read_success(data_dir)

    data = data.sort_values(
        ["gateway_id", "week_start"]
    ).reset_index(drop=True)

    data["faulty"] = (
        data["failure_rate"] > TARGET_THRESHOLD
    ).astype(int)

    return data


def calculate_episodes(data):
    """
    Identify consecutive faulty weeks for each gateway.

    Each continuous sequence of faulty weeks is one episode.
    """

    data = data.copy()

    previous_faulty = (
        data.groupby("gateway_id")["faulty"]
        .shift(1)
        .fillna(0)
    )

    data["new_episode"] = (
        (data["faulty"] == 1)
        & (previous_faulty == 0)
    ).astype(int)

    data["episode_id"] = (
        data.groupby("gateway_id")["new_episode"]
        .cumsum()
    )

    return data


def calculate_cost(actual_data, predictions):
    """
    Calculate operational cost while preserving fault episodes
    across the full historical dataset.

    The episode state is calculated BEFORE restricting evaluation
    to the prediction weeks, so an episode that started earlier
    is not incorrectly treated as a new episode.
    """

    # ---------------------------------------------------------
    # 1. Work on the complete historical actual dataset
    # ---------------------------------------------------------
    data = calculate_episodes(actual_data)

    # ---------------------------------------------------------
    # 2. Prepare predictions
    # ---------------------------------------------------------
    predictions = predictions.copy()

    predictions["week_start"] = pd.to_datetime(
        predictions["week_start"],
        utc=True,
    )

    selected = predictions[
        predictions["rank"] <= VISITS_PER_WEEK
    ].copy()

    selected_by_week = {
        week: set(group["gateway_id"])
        for week, group in selected.groupby("week_start")
    }

    # ---------------------------------------------------------
    # 3. Evaluate only weeks present in predictions
    # ---------------------------------------------------------
    evaluation_weeks = set(
        selected["week_start"].unique()
    )

    visited_episodes = set()

    total_cost = 0
    weekly_results = []

    for week_start in sorted(evaluation_weeks):

        week_data = data[
            data["week_start"] == week_start
        ]

        gateways_selected = selected_by_week.get(
            week_start,
            set(),
        )

        # Every selected gateway costs €380
        visit_cost = (
            len(gateways_selected)
            * VISIT_COST
        )

        unresolved_failure_cost = 0

        for _, row in week_data.iterrows():

            if row["faulty"] != 1:
                continue

            episode_key = (
                row["gateway_id"],
                row["episode_id"],
            )

            # Visiting during this week resolves the episode
            if row["gateway_id"] in gateways_selected:
                visited_episodes.add(
                    episode_key
                )

            # Otherwise the unresolved episode costs €600
            elif episode_key not in visited_episodes:
                unresolved_failure_cost += (
                    FAILURE_COST_PER_WEEK
                )

        week_cost = (
            visit_cost
            + unresolved_failure_cost
        )

        total_cost += week_cost

        weekly_results.append(
            {
                "week_start": week_start,
                "visits": len(gateways_selected),
                "visit_cost": visit_cost,
                "failure_cost": unresolved_failure_cost,
                "total_cost": week_cost,
            }
        )

    return (
        total_cost,
        pd.DataFrame(weekly_results),
    )


def print_cost_summary(name, total_cost, weekly):
    """
    Print a readable cost summary.
    """

    print("\n" + "=" * 50)
    print(name)
    print("=" * 50)

    print(
        f"Total cost: €{total_cost:,.0f}"
    )

    print(
        f"Average weekly cost: "
        f"€{weekly['total_cost'].mean():,.2f}"
    )

    print(
        f"Total visits: "
        f"{weekly['visits'].sum()}"
    )

    print(
        f"Average visits/week: "
        f"{weekly['visits'].mean():.2f}"
    )


def main():
    data_dir = (
        Path(__file__).resolve().parent.parent
        / "data"
    )

    print("Loading historical failure data...")

    actual_data = prepare_actual_data(
        data_dir
    )

    print(
        f"Historical rows: {len(actual_data)}"
    )

    print(
        f"Historical weeks: "
        f"{actual_data['week_start'].nunique()}"
    )

    print(
        f"Fault threshold: "
        f">{TARGET_THRESHOLD:.0%}"
    )

    print(
        "\nCost evaluator is ready."
    )

    print(
        "\nNext:"
        "\n  1. Generate ML Top-15 decisions"
        "\n  2. Generate baseline Top-15 decisions"
        "\n  3. Pass both into calculate_cost()"
        "\n  4. Compare total cost"
    )


if __name__ == "__main__":
    main()