from __future__ import annotations

import json
import re

from pathlib import Path
from typing import Iterator

import pandas as pd

from .features import make_state_features


def overs_to_balls(
    overs_value,
    balls_per_over: int = 6
) -> int:

    """
    Convert cricket over notation into legal balls.

    Examples:
        20   -> 120 balls
        19.5 -> 119 balls
    """

    value = str(overs_value).strip()

    if "." in value:

        completed_overs, extra_balls = value.split(
            ".",
            1
        )

        completed_overs = int(completed_overs)

        extra_balls = int(extra_balls or 0)

        if extra_balls >= balls_per_over:
            raise ValueError(
                "Invalid cricket over notation"
            )

        return (
            completed_overs * balls_per_over
            + extra_balls
        )

    return int(value) * balls_per_over


def iter_deliveries(
    innings: dict
) -> Iterator[dict]:

    """
    Read each delivery from every over.
    """

    for over in innings.get("overs", []):

        for delivery in over.get(
            "deliveries",
            []
        ):

            yield delivery


def get_innings_total(
    innings: dict
) -> int:

    """
    Calculate total innings runs, including
    penalty runs recorded outside deliveries.
    """

    runs = sum(
        int(delivery.get("runs", {}).get("total", 0))
        for delivery in iter_deliveries(innings)
    )

    penalties = innings.get(
        "penalty_runs",
        {}
    )

    runs += int(penalties.get("pre", 0))
    runs += int(penalties.get("post", 0))

    return runs


def get_season(
    info: dict
) -> int | None:

    """
    Extract the season year.
    """

    season_text = str(
        info.get("season", "")
    )

    match = re.search(
        r"(19|20)\d{2}",
        season_text
    )

    if match:
        return int(match.group())

    dates = info.get("dates", [])

    if dates:
        return int(str(dates[0])[:4])

    return None


def build_training_dataset(
    raw_dir: Path
) -> pd.DataFrame:

    """
    Convert historical IPL JSON files into
    second-innings win-probability training data.
    """

    raw_dir = Path(raw_dir)

    json_files = sorted(
        raw_dir.rglob("*.json")
    )

    if not json_files:

        raise FileNotFoundError(
            f"No JSON match files found in {raw_dir}"
        )

    rows = []

    processed_matches = 0

    skipped_matches = 0

    for file_path in json_files:

        try:

            with file_path.open(
                "r",
                encoding="utf-8"
            ) as file:

                match = json.load(file)

        except (OSError, json.JSONDecodeError):

            skipped_matches += 1

            continue

        info = match.get("info", {})

        # We are building a T20 IPL model.
        if info.get("match_type") != "T20":

            continue

        # Avoid accidentally including other
        # competitions if files are mixed.
        event_name = str(
            info.get("event", {}).get("name", "")
        ).lower()

        if (
            event_name
            and "indian premier league" not in event_name
            and "ipl" not in event_name
        ):

            continue

        outcome = info.get("outcome", {})

        winner = outcome.get("winner")

        # Exclude incomplete outcomes.
        if not winner:

            continue

        # Exclude matches decided by an alternative
        # target or result method.
        if outcome.get("method"):

            continue

        innings_list = match.get(
            "innings",
            []
        )

        if len(innings_list) < 2:

            continue

        first_innings = innings_list[0]

        second_innings = innings_list[1]

        if (
            first_innings.get("super_over")
            or second_innings.get("super_over")
        ):

            continue

        batting_team = second_innings.get(
            "team"
        )

        bowling_team = first_innings.get(
            "team"
        )

        if not batting_team or not bowling_team:

            continue

        if winner not in (
            batting_team,
            bowling_team
        ):

            continue

        season = get_season(info)

        if season is None:

            continue

        venue = info.get(
            "venue",
            "Unknown"
        )

        balls_per_over = int(
            info.get("balls_per_over", 6)
        )

        # Calculate first-innings total.
        first_innings_total = get_innings_total(
            first_innings
        )

        # Use the chase target supplied in the data
        # whenever available.
        target_info = second_innings.get(
            "target",
            {}
        )

        target_total = int(
            target_info.get(
                "runs",
                first_innings_total + 1
            )
        )

        target_overs = target_info.get(
            "overs",
            info.get("overs", 20)
        )

        try:

            total_balls = overs_to_balls(
                target_overs,
                balls_per_over
            )

        except (ValueError, TypeError):

            continue

        if total_balls <= 0 or target_total <= 0:

            continue

        # The label describes the chasing team's
        # eventual match outcome.
        label = int(
            winner == batting_team
        )

        runs_scored = int(
            second_innings.get(
                "penalty_runs",
                {}
            ).get("pre", 0)
        )

        balls_bowled = 0

        wickets_lost = 0

        match_id = file_path.stem

        def save_snapshot():

            row = make_state_features(

                batting_team=batting_team,

                bowling_team=bowling_team,

                venue=venue,

                season=season,

                runs_scored=runs_scored,

                target_total=target_total,

                balls_bowled=balls_bowled,

                total_balls=total_balls,

                wickets_lost=wickets_lost,

                balls_per_over=balls_per_over,
            )

            row["label"] = label

            row["match_id"] = match_id

            rows.append(row)

        # Store the initial state before the
        # first legal delivery.
        save_snapshot()

        # Process every second-innings delivery.
        for delivery in iter_deliveries(
            second_innings
        ):

            delivery_runs = int(
                delivery.get(
                    "runs",
                    {}
                ).get(
                    "total",
                    0
                )
            )

            runs_scored += delivery_runs

            extras = delivery.get(
                "extras",
                {}
            )

            # Wides and no-balls do not consume
            # a legal delivery.
            illegal_delivery = (
                int(extras.get("wides", 0)) > 0
                or int(extras.get("noballs", 0)) > 0
            )

            if not illegal_delivery:

                balls_bowled += 1

            # Count dismissed players.
            for wicket in delivery.get(
                "wickets",
                []
            ):

                wicket_kind = str(
                    wicket.get("kind", "")
                ).lower()

                if wicket_kind != "retired hurt":

                    wickets_lost += 1

            wickets_lost = min(
                wickets_lost,
                10
            )

            # Do not store states after the chase
            # is already decided.
            if runs_scored >= target_total:

                break

            if wickets_lost >= 10:

                break

            if balls_bowled >= total_balls:

                break

            save_snapshot()

        processed_matches += 1

    dataset = pd.DataFrame(rows)

    if dataset.empty:

        raise ValueError(
            "No usable training rows were created. "
            "Check the IPL JSON files and their structure."
        )

    if dataset["label"].nunique() != 2:

        raise ValueError(
            "Training dataset must contain both winning "
            "and losing chasing-team examples."
        )

    print("\nDATA PROCESSING COMPLETED")

    print("-------------------------")

    print(
        "Available JSON files:",
        len(json_files)
    )

    print(
        "Processed matches:",
        processed_matches
    )

    print(
        "Skipped unreadable files:",
        skipped_matches
    )

    print(
        "Training snapshots:",
        len(dataset)
    )

    print(
        "Available seasons:",
        sorted(dataset["season"].unique())
    )

    print(
        "\nLabel distribution:"
    )

    print(
        dataset["label"].value_counts()
    )

    return dataset