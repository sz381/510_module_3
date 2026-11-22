"""Grid search over the gradient boosting parameters.

Fits every combination returned by build_grid, scores each one on the
validation split and records it in a separate MLflow experiment. Only the
combination with the lowest validation MAE is scored on the test split.

Usage:
    python script/tune.py

Functions provided:
- build_grid: list every hyperparameter combination to try
- run_grid:   fit and score the combinations one by one
- main:       run the grid, report the ranking and score the best one on test
"""

import sys
from itertools import product
from pathlib import Path

import mlflow
import pandas as pd
import yaml

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

import train


def build_grid():
    """List every hyperparameter combination to try.

    Returns:
        list[dict]: One dict of gradient boosting parameters per combination.
    """
    search = {
        "learning_rate": [0.03, 0.05, 0.1],
        "max_iter": [200, 400],
        "max_leaf_nodes": [15, 31],
        "min_samples_leaf": [10, 20],
    }

    keys = list(search)
    return [dict(zip(keys, values)) for values in product(*search.values())]


def run_grid(X_train, y_train, X_val, y_val, grid, seed, experiment, tracking_uri):
    """Fit and score every combination in the grid.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target in log space.
        X_val (pd.DataFrame): Validation features.
        y_val (pd.Series): Validation target in log space.
        grid (list[dict]): Output of build_grid.
        seed (int): Random seed passed to the model.
        experiment (str): MLflow experiment name for the search runs.
        tracking_uri (str): Where the runs are written.

    Returns:
        list[dict]: One entry per combination with its parameters and metrics.
    """
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)

    results = []
    total = len(grid)

    for index, params in enumerate(grid, start=1):
        full = {"random_state": seed, **params}
        model = train.train_gradient_boosting(X_train, y_train, full)
        metrics = train.evaluate(model, X_val, y_val)

        with mlflow.start_run(run_name=f"grid_{index:02d}"):
            mlflow.log_params({k: str(v) for k, v in full.items()})
            mlflow.log_metrics({f"val_{k}": v for k, v in metrics.items()})

        results.append({"params": full, **metrics})

        print(
            f"[{index:02d}/{total}] lr={full['learning_rate']:<5} "
            f"iter={full['max_iter']:<4} leaves={full['max_leaf_nodes']:<3} "
            f"min_leaf={full['min_samples_leaf']:<3} val_mae={metrics['mae']:7.2f}"
        )

    return results


def main(config_path="config.yaml"):
    """Run the grid and report the best combination.

    Args:
        config_path (str): Path to config.yaml. Missing keys fall back to the
            defaults used here.
    """
    config_file = Path(config_path)
    cfg = yaml.safe_load(config_file.read_text()) if config_file.exists() else {}
    cfg = cfg or {}

    data_cfg = cfg.get("data", {})
    split_cfg = cfg.get("split", {})
    tracking_cfg = cfg.get("tracking", {})

    processed_dir = data_cfg.get("processed_dir", "data/processed")
    target = data_cfg.get("target", "price")
    seed = split_cfg.get("seed", 42)
    tracking_uri = tracking_cfg.get("tracking_uri", "sqlite:///mlflow.db")
    experiment = f"{tracking_cfg.get('experiment_name', 'seattle-airbnb-price')}-tune"

    splits = train.read_splits(processed_dir)
    X_train, y_train = train.prepare_xy(splits["train"], target)
    X_val, y_val = train.prepare_xy(splits["val"], target)
    X_test, y_test = train.prepare_xy(splits["test"], target)

    grid = build_grid()
    print(f"trying {len(grid)} combinations\n")

    results = run_grid(
        X_train, y_train, X_val, y_val, grid, seed, experiment, tracking_uri
    )

    table = pd.DataFrame(
        [
            {
                "learning_rate": r["params"]["learning_rate"],
                "max_iter": r["params"]["max_iter"],
                "max_leaf_nodes": r["params"]["max_leaf_nodes"],
                "min_samples_leaf": r["params"]["min_samples_leaf"],
                "val_mae": round(r["mae"], 2),
                "val_rmse": round(r["rmse"], 2),
                "val_r2": round(r["r2"], 4),
            }
            for r in results
        ]
    ).sort_values("val_mae")

    print("\ntop 10 by val_mae\n")
    print(table.head(10).to_string(index=False))

    best_params = results[table.index[0]]
    print(f"\nbest params: {best_params['params']}")

    best_model = train.train_gradient_boosting(X_train, y_train, best_params["params"])
    test_metrics = train.evaluate(best_model, X_test, y_test)
    print(
        f"best on test: mae={test_metrics['mae']:.2f} "
        f"rmse={test_metrics['rmse']:.2f} r2={test_metrics['r2']:.4f}"
    )


if __name__ == "__main__":
    main()
