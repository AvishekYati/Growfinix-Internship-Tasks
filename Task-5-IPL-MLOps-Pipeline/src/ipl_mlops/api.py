from __future__ import annotations

import json

from datetime import datetime, timezone

from functools import lru_cache

from pathlib import Path

import joblib
import pandas as pd

from fastapi import (
    FastAPI,
    HTTPException,
)

from pydantic import (
    BaseModel,
    Field,
)

from .features import (
    FEATURE_COLUMNS,
    make_state_features,
)


# ---------------------------------------------
# PROJECT PATHS
# ---------------------------------------------

ROOT = Path(
    __file__
).resolve().parents[2]

MODEL_PATH = (
    ROOT / "artifacts" / "model.joblib"
)

METADATA_PATH = (
    ROOT / "artifacts" / "model_metadata.json"
)


# ---------------------------------------------
# FASTAPI APPLICATION
# ---------------------------------------------

app = FastAPI(

    title="Growfinix IPL Edge API",

    description=(
        "IPL second-innings win probability "
        "prediction using machine learning."
    ),

    version="1.0.0",

)


# ---------------------------------------------
# REQUEST VALIDATION
# ---------------------------------------------

class PredictionRequest(BaseModel):

    batting_team: str = Field(
        min_length=2,
        max_length=100
    )

    bowling_team: str = Field(
        min_length=2,
        max_length=100
    )

    venue: str = Field(
        default="Unknown",
        max_length=150
    )

    season: int = Field(
        default=2026,
        ge=2008,
        le=2100
    )

    runs_scored: int = Field(
        ge=0
    )

    target_total: int = Field(
        gt=0
    )

    balls_bowled: int = Field(
        ge=0,
        le=119
    )

    total_balls: int = Field(
        default=120,
        ge=1,
        le=120
    )

    wickets_lost: int = Field(
        ge=0,
        le=9
    )


# ---------------------------------------------
# LOAD TRAINED MODEL
# ---------------------------------------------

@lru_cache(maxsize=1)
def load_model():

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            "Trained model not found. "
            "Run the training pipeline first."
        )

    return joblib.load(
        MODEL_PATH
    )


# ---------------------------------------------
# LOAD MODEL METADATA
# ---------------------------------------------

def load_metadata():

    if not METADATA_PATH.exists():

        return {}

    return json.loads(
        METADATA_PATH.read_text(
            encoding="utf-8"
        )
    )


# ---------------------------------------------
# ROOT ENDPOINT
# ---------------------------------------------

@app.get("/")
def root():

    return {

        "project":
            "Growfinix IPL Edge",

        "description":
            "IPL Live Win Probability API",

        "documentation":
            "/docs",

        "health":
            "/health",

        "prediction":
            "/predict",

    }


# ---------------------------------------------
# HEALTH CHECK
# ---------------------------------------------

@app.get("/health")
def health():

    if not MODEL_PATH.exists():

        raise HTTPException(

            status_code=503,

            detail="Model is not available"

        )

    metadata = load_metadata()

    return {

        "status":
            "healthy",

        "model_loaded":
            True,

        "model_version":
            metadata.get(
                "model_version",
                "unknown"
            ),

    }


# ---------------------------------------------
# MODEL INFORMATION
# ---------------------------------------------

@app.get("/model-info")
def model_info():

    if not MODEL_PATH.exists():

        raise HTTPException(

            status_code=503,

            detail="Model is not available"

        )

    return load_metadata()


# ---------------------------------------------
# WIN PROBABILITY PREDICTION
# ---------------------------------------------

@app.post("/predict")
def predict(
    request: PredictionRequest
):

    # Validate team selection.
    if (
        request.batting_team.strip().casefold()
        == request.bowling_team.strip().casefold()
    ):

        raise HTTPException(

            status_code=422,

            detail="Batting and bowling teams must differ."

        )

    # The chasing team must not already have won.
    if request.runs_scored >= request.target_total:

        raise HTTPException(

            status_code=422,

            detail="The chase has already reached its target."

        )

    # There must be legal balls remaining.
    if request.balls_bowled >= request.total_balls:

        raise HTTPException(

            status_code=422,

            detail="The innings has no balls remaining."

        )

    # Load trained model.
    try:

        model = load_model()

    except FileNotFoundError as error:

        raise HTTPException(

            status_code=503,

            detail=str(error)

        )

    # Calculate features using the same function
    # used during historical model training.
    features = make_state_features(

        batting_team=request.batting_team,

        bowling_team=request.bowling_team,

        venue=request.venue,

        season=request.season,

        runs_scored=request.runs_scored,

        target_total=request.target_total,

        balls_bowled=request.balls_bowled,

        total_balls=request.total_balls,

        wickets_lost=request.wickets_lost,

    )

    # Create a single-row DataFrame.
    input_data = pd.DataFrame(

        [features],

        columns=FEATURE_COLUMNS

    )

    # Obtain probabilities for both classes.
    probabilities = model.predict_proba(
        input_data
    )[0]

    class_list = list(
        model.classes_
    )

    winning_class_index = class_list.index(1)

    batting_win_probability = float(
        probabilities[winning_class_index]
    )

    bowling_win_probability = (
        1.0 - batting_win_probability
    )

    predicted_winner = (

        request.batting_team

        if batting_win_probability >= 0.5

        else request.bowling_team

    )

    metadata = load_metadata()

    return {

        "project":
            "Growfinix IPL Edge",

        "prediction_timestamp_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "batting_team":
            request.batting_team,

        "bowling_team":
            request.bowling_team,

        "win_probability_pct":
            round(
                batting_win_probability * 100,
                2
            ),

        "opposition_win_probability_pct":
            round(
                bowling_win_probability * 100,
                2
            ),

        "predicted_winner":
            predicted_winner,

        "model_version":
            metadata.get(
                "model_version",
                "unknown"
            ),

        "match_state": {

            "runs_remaining":
                features["runs_remaining"],

            "balls_remaining":
                features["balls_remaining"],

            "wickets_remaining":
                features["wickets_remaining"],

            "current_run_rate":
                round(
                    features["current_run_rate"],
                    2
                ),

            "required_run_rate":
                round(
                    features["required_run_rate"],
                    2
                ),

        },

        "disclaimer":
            "Prediction is a model estimate, not a guarantee.",

    }