from __future__ import annotations

import hashlib
import json
import os

from datetime import datetime, timezone
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn

import numpy as np
import pandas as pd

from mlflow.models import infer_signature

from sklearn.compose import ColumnTransformer

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    brier_score_loss,
    log_loss,
)

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)

from .data import build_training_dataset

from .features import (
    FEATURE_COLUMNS,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
)


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

ROOT = Path(
    __file__
).resolve().parents[2]

RAW_DATA_DIR = (
    ROOT / "data" / "raw" / "ipl_json"
)

PROCESSED_DIR = (
    ROOT / "data" / "processed"
)

ARTIFACTS_DIR = (
    ROOT / "artifacts"
)

MODEL_PATH = (
    ARTIFACTS_DIR / "model.joblib"
)

METADATA_PATH = (
    ARTIFACTS_DIR / "model_metadata.json"
)


# --------------------------------------------------
# CREATE PREPROCESSING PIPELINE
# --------------------------------------------------

def create_pipeline(
    model_name: str
) -> Pipeline:

    """
    Create a consistent preprocessing and
    machine learning pipeline.
    """

    preprocessor = ColumnTransformer(

        transformers=[

            (
                "categorical",

                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                ),

                CATEGORICAL_FEATURES,
            ),

            (
                "numeric",

                StandardScaler(),

                NUMERIC_FEATURES,
            ),

        ],

        remainder="drop",
    )

    if model_name == "logistic_regression":

        classifier = LogisticRegression(

            max_iter=1500,

            class_weight="balanced",

            random_state=42,
        )

    elif model_name == "random_forest":

        classifier = RandomForestClassifier(

            n_estimators=250,

            max_depth=16,

            min_samples_leaf=4,

            max_features="sqrt",

            class_weight="balanced_subsample",

            random_state=42,

            n_jobs=-1,
        )

    else:

        raise ValueError(
            f"Unknown model: {model_name}"
        )

    return Pipeline(

        steps=[

            (
                "preprocessor",
                preprocessor
            ),

            (
                "classifier",
                classifier
            ),

        ]

    )


# --------------------------------------------------
# EVALUATION METRICS
# --------------------------------------------------

def calculate_metrics(
    y_true,
    probabilities
) -> dict:

    """
    Evaluate probability predictions and
    classification performance.
    """

    predicted = (
        probabilities >= 0.5
    ).astype(int)

    probability_matrix = np.column_stack(
        [
            1.0 - probabilities,
            probabilities
        ]
    )

    return {

        "accuracy": float(
            accuracy_score(
                y_true,
                predicted
            )
        ),

        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_true,
                predicted
            )
        ),

        "precision": float(
            precision_score(
                y_true,
                predicted,
                zero_division=0
            )
        ),

        "recall": float(
            recall_score(
                y_true,
                predicted,
                zero_division=0
            )
        ),

        "f1_score": float(
            f1_score(
                y_true,
                predicted,
                zero_division=0
            )
        ),

        "roc_auc": float(
            roc_auc_score(
                y_true,
                probabilities
            )
        ),

        "brier_score": float(
            brier_score_loss(
                y_true,
                probabilities
            )
        ),

        "log_loss": float(
            log_loss(
                y_true,
                probability_matrix,
                labels=[0, 1]
            )
        ),

    }


# --------------------------------------------------
# SEASON-BASED DATA SPLIT
# --------------------------------------------------

def split_by_season(
    dataset: pd.DataFrame
):

    seasons = sorted(
        dataset["season"].unique()
    )

    if len(seasons) < 4:

        raise ValueError(
            "At least four distinct IPL seasons "
            "are required for this temporal split."
        )

    if len(seasons) >= 6:

        train_seasons = seasons[:-3]

        validation_seasons = [
            seasons[-3]
        ]

        test_seasons = seasons[-2:]

    else:

        train_seasons = seasons[:-2]

        validation_seasons = [
            seasons[-2]
        ]

        test_seasons = [
            seasons[-1]
        ]

    train_data = dataset[
        dataset["season"].isin(train_seasons)
    ].copy()

    validation_data = dataset[
        dataset["season"].isin(validation_seasons)
    ].copy()

    test_data = dataset[
        dataset["season"].isin(test_seasons)
    ].copy()

    for name, frame in [

        ("training", train_data),

        ("validation", validation_data),

        ("testing", test_data),

    ]:

        if frame["label"].nunique() != 2:

            raise ValueError(
                f"{name} dataset must contain both classes."
            )

    return (

        train_data,

        validation_data,

        test_data,

        train_seasons,

        validation_seasons,

        test_seasons,

    )


# --------------------------------------------------
# MAIN TRAINING FUNCTION
# --------------------------------------------------

def main():

    print("\nSTARTING IPL MLOPS TRAINING")

    print("============================")

    # Create output directories.
    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    ARTIFACTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # 1. Process historical IPL data.
    dataset = build_training_dataset(
        RAW_DATA_DIR
    )

    processed_file = (
        PROCESSED_DIR / "chase_states.csv"
    )

    dataset.to_csv(
        processed_file,
        index=False
    )

    # Generate a fingerprint for the processed data.
    dataset_hash = hashlib.sha256(
        processed_file.read_bytes()
    ).hexdigest()

    # 2. Split data by historical seasons.
    (
        train_data,

        validation_data,

        test_data,

        train_seasons,

        validation_seasons,

        test_seasons,

    ) = split_by_season(dataset)

    X_train = train_data[
        FEATURE_COLUMNS
    ]

    y_train = train_data[
        "label"
    ]

    X_validation = validation_data[
        FEATURE_COLUMNS
    ]

    y_validation = validation_data[
        "label"
    ]

    X_test = test_data[
        FEATURE_COLUMNS
    ]

    y_test = test_data[
        "label"
    ]

    print("\nDATA SPLIT")

    print("-----------")

    print("Training seasons:", train_seasons)

    print("Validation seasons:", validation_seasons)

    print("Testing seasons:", test_seasons)

    print("Training rows:", len(X_train))

    print("Validation rows:", len(X_validation))

    print("Testing rows:", len(X_test))

    # 3. Compare two models.
    candidates = {

        "logistic_regression":
            create_pipeline(
                "logistic_regression"
            ),

        "random_forest":
            create_pipeline(
                "random_forest"
            ),

    }

    validation_scores = {}

    print("\nMODEL COMPARISON")

    print("----------------")

    for model_name, model in candidates.items():

        print(
            f"\nTraining {model_name}..."
        )

        model.fit(
            X_train,
            y_train
        )

        probabilities = model.predict_proba(
            X_validation
        )[:, 1]

        validation_auc = roc_auc_score(
            y_validation,
            probabilities
        )

        validation_scores[model_name] = float(
            validation_auc
        )

        print(
            f"{model_name} validation ROC-AUC: "
            f"{validation_auc:.4f}"
        )

    # 4. Automatically select the best model.
    best_model_name = max(
        validation_scores,
        key=validation_scores.get
    )

    print(
        "\nSELECTED MODEL:",
        best_model_name
    )

    # 5. Retrain selected model on training
    # and validation seasons.
    final_training_data = pd.concat(
        [
            train_data,
            validation_data
        ],
        ignore_index=True
    )

    X_final = final_training_data[
        FEATURE_COLUMNS
    ]

    y_final = final_training_data[
        "label"
    ]

    final_model = create_pipeline(
        best_model_name
    )

    print(
        "\nRetraining selected model..."
    )

    final_model.fit(
        X_final,
        y_final
    )

    # 6. Evaluate only on untouched test seasons.
    test_probabilities = final_model.predict_proba(
        X_test
    )[:, 1]

    test_metrics = calculate_metrics(
        y_test,
        test_probabilities
    )

    print("\nFINAL TEST METRICS")

    print("------------------")

    for metric_name, value in test_metrics.items():

        print(
            f"{metric_name}: {value:.4f}"
        )

    # 7. Save trained model.
    joblib.dump(
        final_model,
        MODEL_PATH
    )

    model_hash = hashlib.sha256(
        MODEL_PATH.read_bytes()
    ).hexdigest()

    model_version = datetime.now(
        timezone.utc
    ).strftime(
        "%Y%m%dT%H%M%SZ"
    )

    # 8. Prepare metadata.
    report = {

        "project_name":
            "Growfinix IPL Edge",

        "problem_type":
            "Second innings win probability",

        "selected_model":
            best_model_name,

        "model_version":
            model_version,

        "trained_at_utc":
            datetime.now(timezone.utc).isoformat(),

        "dataset_sha256":
            dataset_hash,

        "model_sha256":
            model_hash,

        "training_seasons":
            [int(x) for x in train_seasons],

        "validation_seasons":
            [int(x) for x in validation_seasons],

        "test_seasons":
            [int(x) for x in test_seasons],

        "validation_roc_auc":
            validation_scores,

        "test_metrics":
            test_metrics,

        "training_snapshots":
            int(len(dataset)),

        "unique_matches":
            int(dataset["match_id"].nunique()),

        "features":
            FEATURE_COLUMNS,

    }

    # 9. Configure MLflow.
    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI"
    )

    if not tracking_uri:

        tracking_uri = (
            ROOT / "mlruns"
        ).resolve().as_uri()

    mlflow.set_tracking_uri(
        tracking_uri
    )

    mlflow.set_experiment(
        "Growfinix IPL Win Probability"
    )

    # 10. Log parameters, metrics and model.
    with mlflow.start_run(
        run_name=(
            f"{best_model_name}_{model_version}"
        )
    ) as active_run:

        mlflow.log_params({

            "selected_model":
                best_model_name,

            "problem_type":
                "second_innings_win_probability",

            "split_strategy":
                "chronological_season_split",

            "training_seasons":
                ",".join(
                    map(str, train_seasons)
                ),

            "validation_seasons":
                ",".join(
                    map(str, validation_seasons)
                ),

            "test_seasons":
                ",".join(
                    map(str, test_seasons)
                ),

            "training_snapshots":
                len(dataset),

            "unique_matches":
                int(
                    dataset["match_id"].nunique()
                ),

        })

        # Log validation scores for both candidates.
        for name, score in validation_scores.items():

            mlflow.log_metric(
                f"validation_roc_auc_{name}",
                score
            )

        # Log final test metrics.
        for name, value in test_metrics.items():

            mlflow.log_metric(
                f"test_{name}",
                value
            )

        # Log dataset and model identification.
        mlflow.set_tags({

            "project":
                "Growfinix IPL Edge",

            "dataset_sha256":
                dataset_hash,

            "model_sha256":
                model_hash,

            "training_type":
                "automated",

        })

        report["mlflow_run_id"] = (
            active_run.info.run_id
        )

        # Save metadata for the API.
        METADATA_PATH.write_text(
            json.dumps(
                report,
                indent=4
            ),
            encoding="utf-8"
        )

        mlflow.log_artifact(
            str(METADATA_PATH),
            artifact_path="evaluation"
        )

        # Record a model signature.
        input_example = X_test.head(5)

        signature = infer_signature(
            input_example,
            final_model.predict_proba(
                input_example
            )
        )

        # Log the selected model in MLflow.
        mlflow.sklearn.log_model(

            sk_model=final_model,

            name="ipl-chase-win-probability",

            signature=signature,

            input_example=input_example,

            serialization_format="cloudpickle",
        )

    # 11. Final summary.
    print("\nTRAINING COMPLETED")

    print("==================")

    print("Selected model:", best_model_name)

    print("Model file:", MODEL_PATH)

    print("Metadata file:", METADATA_PATH)

    print("MLflow run:", report["mlflow_run_id"])

    print(
        "\nYour model is ready for API deployment."
    )


if __name__ == "__main__":

    main()