import pytest

from ipl_mlops.data import overs_to_balls

from ipl_mlops.features import (
    make_state_features
)


def test_overs_to_balls():

    assert overs_to_balls(20) == 120

    assert overs_to_balls(19.5) == 119


def test_basic_match_features():

    features = make_state_features(

        batting_team="Chennai Super Kings",

        bowling_team="Mumbai Indians",

        venue="M Chinnaswamy Stadium",

        season=2026,

        runs_scored=90,

        target_total=180,

        balls_bowled=60,

        total_balls=120,

        wickets_lost=2,

    )

    assert features["runs_remaining"] == 90

    assert features["balls_remaining"] == 60

    assert features["wickets_remaining"] == 8

    assert features["current_run_rate"] == 9.0

    assert features["required_run_rate"] == 9.0

    assert features["progress"] == 0.5


def test_invalid_total_balls():

    with pytest.raises(ValueError):

        make_state_features(

            batting_team="Team A",

            bowling_team="Team B",

            venue="Stadium",

            season=2026,

            runs_scored=50,

            target_total=180,

            balls_bowled=10,

            total_balls=0,

            wickets_lost=1,

        )