"""Training for the Seattle Airbnb price model.

Reads the three processed splits, fits a gradient boosting model and a ridge
baseline, scores both in dollars and records every run in MLflow.

Usage:
    python src/train.py

Functions provided:
- read_splits:              read the train, validation and test parquet files
- prepare_xy:               split a table into features and the log1p target
- train_gradient_boosting:  fit the main model
- train_ridge:              fit the imputed and scaled baseline
- evaluate:                 score a model and convert the errors to dollars
- log_to_mlflow:            record parameters, metrics and the model
- save_model:               write the fitted model to disk with joblib
- main:                     run both models and keep the better one
"""

import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def read_splits(processed_dir):
    """Read the three processed parquet files.

    Args:
        processed_dir (str | Path): Directory holding train, val and test
            parquet files.

    Returns:
        dict[str, pd.DataFrame]: The three splits keyed by name.
    """
    processed_dir = Path(processed_dir)
    return {
        name: pd.read_parquet(processed_dir / f"{name}.parquet")
        for name in ("train", "val", "test")
    }


def prepare_xy(df, target):
    """Split a table into the feature matrix and the log1p target.

    Args:
        df (pd.DataFrame): One of the processed splits.
        target (str): Name of the target column.

    Returns:
        tuple[pd.DataFrame, pd.Series]: X with numeric features only and y as
        log1p of the target.
    """
    drop = [c for c in (target, "split") if c in df.columns]
    X = df.drop(columns=drop)
    y = np.log1p(df[target])
    return X, y


def train_gradient_boosting(X_train, y_train, params):
    """Fit the main gradient boosting model.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target, already log1p transformed.
        params (dict): Hyperparameters taken from config.yaml.

    Returns:
        Pipeline: Single-step pipeline wrapping the fitted boosting model.
    """
    model = Pipeline([("gradient_boosting", HistGradientBoostingRegressor(**params))])
    model.fit(X_train, y_train)
    return model


def train_ridge(X_train, y_train, params):
    """Fit the ridge baseline behind an imputer and a scaler.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target, already log1p transformed.
        params (dict): Hyperparameters passed to Ridge.

    Returns:
        Pipeline: Imputer, scaler and fitted ridge model.
    """
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("ridge", Ridge(**params)),
        ]
    )
    model.fit(X_train, y_train)
    return model


def evaluate(model, X, y_log):
    """Score a model and report the errors in dollars.

    Both the predictions and the target are converted back from log space
    before the metrics are computed.

    Args:
        model: Any fitted regressor with a predict method.
        X (pd.DataFrame): Features to score.
        y_log (pd.Series): Target in log space.

    Returns:
        dict[str, float]: mae, rmse and r2 on the original dollar scale.
    """
    predicted = np.expm1(model.predict(X))
    actual = np.expm1(y_log.to_numpy())
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)),
    }


def log_to_mlflow(experiment, run_name, params, metrics, model, tracking_uri):
    """Record one training run in MLflow.

    Args:
        experiment (str): MLflow experiment name.
        run_name (str): Name shown for this run.
        params (dict): Values recorded as run parameters.
        metrics (dict): Values recorded as run metrics.
        model: Fitted model to store as an MLflow artifact.
        tracking_uri (str): Where the runs are written.

    Returns:
        str: The id of the created run.
    """
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        try:
            mlflow.sklearn.log_model(
                model, name="model", serialization_format="pickle"
            )
        except TypeError:
            mlflow.sklearn.log_model(
                model, artifact_path="model", serialization_format="pickle"
            )
        return mlflow.active_run().info.run_id


def save_model(model, name, models_dir):
    """Write a fitted model to disk.

    Args:
        model: Fitted model to persist.
        name (str): File name without the extension.
        models_dir (str | Path): Destination directory.

    Returns:
        Path: The written file.
    """
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    path = models_dir / f"{name}.joblib"
    joblib.dump(model, path)
    return path


def main(config_path="config.yaml"):
    """Train both models, compare them and keep the better one.

    Args:
        config_path (str): Path to config.yaml. Missing keys fall back to the
            defaults used here.
    """
    config_file = Path(config_path)
    cfg = yaml.safe_load(config_file.read_text()) if config_file.exists() else {}
    cfg = cfg or {}

    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    tracking_cfg = cfg.get("tracking", {})
    split_cfg = cfg.get("split", {})

    processed_dir = data_cfg.get("processed_dir", "data/processed")
    target = data_cfg.get("target", "price")
    models_dir = cfg.get("models_dir", "models")
    tracking_uri = tracking_cfg.get("tracking_uri", "sqlite:///mlflow.db")
    experiment = tracking_cfg.get("experiment_name", "seattle-airbnb-price")
    seed = split_cfg.get("seed", 42)

    hgb_params = {"random_state": seed, **model_cfg.get("gradient_boosting", {})}
    ridge_params = model_cfg.get("ridge", {})

    splits = read_splits(processed_dir)
    X_train, y_train = prepare_xy(splits["train"], target)
    X_val, y_val = prepare_xy(splits["val"], target)
    X_test, y_test = prepare_xy(splits["test"], target)

    print("train:", X_train.shape, "| val:", X_val.shape, "| test:", X_test.shape)

    results = {}
    runs = [
        ("gradient_boosting", train_gradient_boosting, hgb_params),
        ("ridge", train_ridge, ridge_params),
    ]

    for name, trainer, params in runs:
        model = trainer(X_train, y_train, params)
        val_metrics = evaluate(model, X_val, y_val)
        test_metrics = evaluate(model, X_test, y_test)

        metrics = {f"val_{k}": v for k, v in val_metrics.items()}
        metrics.update({f"test_{k}": v for k, v in test_metrics.items()})

        run_params = {
            "model": name,
            "n_features": X_train.shape[1],
            "n_train": len(X_train),
            **{k: str(v) for k, v in params.items()},
        }

        run_id = log_to_mlflow(
            experiment, name, run_params, metrics, model, tracking_uri
        )
        results[name] = {"model": model, "metrics": metrics, "run_id": run_id}

        print(
            f"{name:18s} val_mae={val_metrics['mae']:8.2f} "
            f"test_mae={test_metrics['mae']:8.2f} "
            f"test_r2={test_metrics['r2']:6.3f}"
        )

    best_name = min(results, key=lambda k: results[k]["metrics"]["val_mae"])
    path = save_model(results[best_name]["model"], best_name, models_dir)

    summary = {
        "best_model": best_name,
        "run_id": results[best_name]["run_id"],
        "metrics": results[best_name]["metrics"],
        "model_path": str(path),
    }
    (Path(models_dir) / "model_card.json").write_text(json.dumps(summary, indent=2))

    print(f"best model: {best_name} -> {path}")


if __name__ == "__main__":
    main()
