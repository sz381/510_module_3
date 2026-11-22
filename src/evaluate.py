"""Evaluation for the Seattle Airbnb price model.

Loads the saved model, scores it again on the test split and writes the two
diagnostic plots that go into the report and into MLflow.

Usage:
    python src/evaluate.py

Functions provided:
- plot_predictions: scatter of predicted against actual price
- plot_residuals:   histogram of the residuals in dollars
- main:             reload the saved model and regenerate the plots
"""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import yaml

import train


def plot_predictions(model, X, y_log, path):
    """Draw predicted price against actual price.

    Points close to the diagonal are accurate predictions. A cloud sitting above
    the line means the model overestimates, below means it underestimates.

    Args:
        model: Fitted model with a predict method.
        X (pd.DataFrame): Features to score.
        y_log (pd.Series): Target in log space.
        path (str | Path): Where to write the PNG.

    Returns:
        Path: The written image.
    """
    predicted = np.expm1(model.predict(X))
    actual = np.expm1(y_log.to_numpy())

    limit = float(max(actual.max(), predicted.max())) * 1.05

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(actual, predicted, s=12, alpha=0.45, edgecolors="none")
    ax.plot([0, limit], [0, limit], color="crimson", linewidth=1, linestyle="--")
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    ax.set_xlabel("actual price (USD per night)")
    ax.set_ylabel("predicted price (USD per night)")
    ax.set_title("Predicted vs actual price on the test split")
    fig.tight_layout()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_residuals(model, X, y_log, path):
    """Draw the distribution of the residuals.

    Residuals are signed as actual minus predicted, so a distribution centred
    left of zero means the model tends to overestimate.

    Args:
        model: Fitted model with a predict method.
        X (pd.DataFrame): Features to score.
        y_log (pd.Series): Target in log space.
        path (str | Path): Where to write the PNG.

    Returns:
        Path: The written image.
    """
    predicted = np.expm1(model.predict(X))
    actual = np.expm1(y_log.to_numpy())
    residual = actual - predicted

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(residual, bins=50, color="steelblue", edgecolor="white", linewidth=0.4)
    ax.axvline(0, color="crimson", linewidth=1, linestyle="--")
    ax.axvline(
        float(np.median(residual)),
        color="black",
        linewidth=1,
        linestyle=":",
        label=f"median {np.median(residual):+.1f} USD",
    )
    ax.set_xlabel("residual, actual minus predicted (USD)")
    ax.set_ylabel("number of listings")
    ax.set_title("Residual distribution on the test split")
    ax.legend()
    fig.tight_layout()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main(config_path="config.yaml"):
    """Reload the saved model and regenerate the evaluation outputs.

    Args:
        config_path (str): Path to config.yaml. Missing keys fall back to the
            defaults used here.
    """
    config_file = Path(config_path)
    cfg = yaml.safe_load(config_file.read_text()) if config_file.exists() else {}
    cfg = cfg or {}

    data_cfg = cfg.get("data", {})
    processed_dir = data_cfg.get("processed_dir", "data/processed")
    target = data_cfg.get("target", "price")
    models_dir = Path(cfg.get("models_dir", "models"))
    reports_dir = Path(cfg.get("reports_dir", "reports"))

    card = json.loads((models_dir / "model_card.json").read_text())
    model = joblib.load(models_dir / f"{card['best_model']}.joblib")

    splits = train.read_splits(processed_dir)
    X_test, y_test = train.prepare_xy(splits["test"], target)

    metrics = train.evaluate(model, X_test, y_test)
    print(f"model     : {card['best_model']}")
    print(f"test mae  : {metrics['mae']:.2f} USD")
    print(f"test rmse : {metrics['rmse']:.2f} USD")
    print(f"test r2   : {metrics['r2']:.3f}")

    predictions_path = plot_predictions(
        model, X_test, y_test, reports_dir / "predictions.png"
    )
    residuals_path = plot_residuals(
        model, X_test, y_test, reports_dir / "residuals.png"
    )

    print(f"wrote     : {predictions_path}")
    print(f"wrote     : {residuals_path}")


if __name__ == "__main__":
    main()
