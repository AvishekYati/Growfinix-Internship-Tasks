from __future__ import annotations

from typing import Any


# All features expected by the ML model.
FEATURE_COLUMNS = [
    "batting_team",
    "bowling_team",
    "venue",
    "season",
    "runs_scored",
    "target_total",
    "runs_remaining",
    "balls_bowled",
    "balls_remaining",
    "wickets_lost",
    "wickets_remaining",
    "current_run_rate",
    "required_run_rate",
    "run_rate_gap",
    "progress",
]


# Features containing text values.
CATEGORICAL_FEATURES = [
    "batting_team",
    "bowling_team",
    "venue",
]


# Remaining features are numerical.
NUMERIC_FEATURES = [
    column
    for column in FEATURE_COLUMNS
    if column not in CATEGORICAL_FEATURES
]


def make_state_features(
    *,
    batting_team: str,
    bowling_team: str,
    venue: str,
    season: int,
    runs_scored: int,
    target_total: int,
    balls_bowled: int,
    total_balls: int,
    wickets_lost: int,
    balls_per_over: int = 6,
) -> dict[str, Any]:

    """
    Convert a cricket scoreboard into ML model features.

    This function is shared by:
    1. Historical training data processing.
    2. The live prediction API.

    Keeping both processes consistent prevents
    training-serving feature mismatches.
    """

    if total_balls <= 0:
        raise ValueError("total_balls must be positive")

    if balls_per_over <= 0:
        raise ValueError("balls_per_over must be positive")

    if balls_bowled < 0 or balls_bowled > total_balls:
        raise ValueError("Invalid number of balls bowled")

    if wickets_lost < 0 or wickets_lost > 10:
        raise ValueError("Invalid wicket count")

    if target_total <= 0:
        raise ValueError("target_total must be positive")

    # Runs still required to win.
    runs_remaining = max(
        target_total - runs_scored,
        0
    )

    # Legal balls still available.
    balls_remaining = max(
        total_balls - balls_bowled,
        0
    )

    # Remaining wickets.
    wickets_remaining = max(
        10 - wickets_lost,
        0
    )

    # Calculate overs already completed.
    overs_bowled = (
        balls_bowled / balls_per_over
    )

    # Current run rate.
    if overs_bowled > 0:
        current_run_rate = (
            runs_scored / overs_bowled
        )
    else:
        current_run_rate = 0.0

    # Required run rate.
    overs_remaining = (
        balls_remaining / balls_per_over
    )

    if overs_remaining > 0:
        required_run_rate = (
            runs_remaining / overs_remaining
        )
    else:
        required_run_rate = 0.0

    # Difference between scoring rates.
    run_rate_gap = (
        current_run_rate - required_run_rate
    )

    # Percentage of innings completed.
    progress = (
        balls_bowled / total_balls
    )

    return {
        "batting_team": batting_team.strip(),
        "bowling_team": bowling_team.strip(),
        "venue": venue.strip() or "Unknown",

        "season": int(season),

        "runs_scored": int(runs_scored),
        "target_total": int(target_total),
        "runs_remaining": int(runs_remaining),

        "balls_bowled": int(balls_bowled),
        "balls_remaining": int(balls_remaining),

        "wickets_lost": int(wickets_lost),
        "wickets_remaining": int(wickets_remaining),

        "current_run_rate": float(current_run_rate),
        "required_run_rate": float(required_run_rate),

        "run_rate_gap": float(run_rate_gap),

        "progress": float(progress),
    }