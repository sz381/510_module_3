"""Train the five-field model used by the prediction API."""

from pathlib import Path

import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.train import evaluate, log_to_mlflow, prepare_xy, read_splits, save_model


def decode_category(frame, prefix):
    """Recover a category from the existing model-ready one-hot columns."""
    encoded = frame.loc[:, frame.columns.str.startswith(prefix)]
    if (
        encoded.empty
        or not encoded.isin([0, 1]).all().all()
        or not encoded.sum(axis=1).eq(1).all()
    ):
        raise ValueError(f"Expected exactly one active {prefix} column per listing")
    return encoded.idxmax(axis=1).str.removeprefix(prefix)


def serving_features(frame):
    return pd.DataFrame(
        {
            "room_type": decode_category(frame, "room_type_"),
            "accommodates": frame["accommodates"],
            "bedrooms": frame["bedrooms"],
            "bathrooms": frame["bathrooms"],
            "neighborhood_group": decode_category(frame, "neighbourhood_group_cleansed_"),
        }
    )


def main():
    config = yaml.safe_load(Path("config.yaml").read_text())
    splits = read_splits(config["data"]["processed_dir"])
    examples = {}
    for name, frame in splits.items():
        features, target = prepare_xy(frame, config["data"]["target"])
        examples[name] = serving_features(features), target

    params = {"random_state": config["split"]["seed"], **config["model"]["gradient_boosting"]}
    encoder = ColumnTransformer(
        [
            ("room_type", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["room_type"]),
            ("capacity", "passthrough", ["accommodates", "bedrooms", "bathrooms"]),
            (
                "neighborhood_group",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ["neighborhood_group"],
            ),
        ]
    )
    model = Pipeline(
        [("features", encoder), ("gradient_boosting", HistGradientBoostingRegressor(**params))]
    )
    model.fit(*examples["train"])

    metrics = {}
    for split in ("val", "test"):
        scores = evaluate(model, *examples[split])
        metrics.update({f"{split}_{key}": value for key, value in scores.items()})
        print(f"{split}: MAE ${scores['mae']:.2f} | RMSE ${scores['rmse']:.2f} | R² {scores['r2']:.3f}")

    tracking = config["tracking"]
    run_id = log_to_mlflow(
        tracking["experiment_name"],
        config["serving"]["model_name"],
        {**params, "features": ", ".join(model.feature_names_in_)},
        metrics,
        model,
        tracking["tracking_uri"],
    )
    path = save_model(model, config["serving"]["model_name"], config["models_dir"])
    print(f"Saved {path} (MLflow run {run_id})")


if __name__ == "__main__":
    main()
