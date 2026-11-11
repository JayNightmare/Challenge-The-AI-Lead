from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any
from urllib.request import urlopen

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from ml.common.artifacts import (
    DATA,
    ROOT,
    REPORTS,
    model_paths,
    save_sklearn_artifact,
    sha256_file,
)
from ml.common.metrics import regression_metrics
from ml.common.splits import chronological_splits

SOURCE_URL = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"
SOURCE_INFO = {
    "name": "Bike Sharing Dataset",
    "repository": "UCI Machine Learning Repository",
    "dataset_id": 275,
    "url": "https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset",
    "license": "CC BY 4.0",
    "attribution": "Fanaee-T, H. (2013). Bike Sharing Dataset. UCI Machine Learning Repository.",
    "target": "cnt (total hourly rentals)",
}
SEED = 42
FEATURES = [
    "season",
    "yr",
    "mnth",
    "hr",
    "holiday",
    "weekday",
    "workingday",
    "weathersit",
    "temp",
    "atemp",
    "hum",
    "windspeed",
]
EXCLUDED = {"instant", "dteday", "casual", "registered", "cnt"}


def download_source(destination: Path | None = None) -> Path:
    destination = destination or DATA / "raw" / "bike-sharing-dataset.zip"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        with urlopen(SOURCE_URL, timeout=60) as response, destination.open(
            "wb"
        ) as output:
            output.write(response.read())
    if not zipfile.is_zipfile(destination):
        destination.unlink(missing_ok=True)
        raise ValueError("Downloaded UCI Bike Sharing archive is not a valid ZIP file")
    return destination


def parse_source(archive: Path) -> pd.DataFrame:
    with zipfile.ZipFile(archive) as source:
        candidates = [name for name in source.namelist() if name.endswith("hour.csv")]
        if len(candidates) != 1:
            raise ValueError(
                "Expected exactly one hour.csv in the UCI Bike Sharing archive"
            )
        with source.open(candidates[0]) as raw:
            frame = pd.read_csv(raw)
    missing = (set(FEATURES) | {"dteday", "hr", "cnt"}) - set(frame.columns)
    if missing:
        raise ValueError(
            f"UCI hourly file is missing expected fields: {sorted(missing)}"
        )
    if set(FEATURES) & EXCLUDED:
        raise ValueError(
            "A target-leaking or row-identifier field was included as a feature"
        )
    if frame[FEATURES + ["cnt"]].isna().any().any():
        raise ValueError("Unexpected missing value in required UCI hourly fields")
    frame["dteday"] = pd.to_datetime(frame["dteday"], errors="raise")
    return frame


def prepare_data() -> dict[str, int]:
    archive = download_source()
    frame = parse_source(archive)
    splits = chronological_splits(frame)
    output = DATA / "processed" / "tabular"
    output.mkdir(parents=True, exist_ok=True)
    for name, split in splits.items():
        split.to_csv(output / f"{name}.csv", index=False)
    manifest = {
        "source": SOURCE_INFO,
        "archive_sha256": sha256_file(archive),
        "seed": SEED,
        "split_method": "chronological 70/15/15 after stable ordering by dteday and hour",
        "split_sha256": {name: sha256_file(output / f"{name}.csv") for name in splits},
        "features": FEATURES,
        "excluded_columns": sorted(EXCLUDED),
        "counts": {name: len(split) for name, split in splits.items()},
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    (output / "SOURCE.txt").write_text(
        json.dumps(SOURCE_INFO, indent=2) + "\n", encoding="utf-8"
    )
    return manifest["counts"]


class GroupedMeanRegressor:
    def fit(self, features: pd.DataFrame, target: pd.Series) -> "GroupedMeanRegressor":
        self.global_mean_ = float(target.mean())
        values = pd.DataFrame(
            {
                "hr": features["hr"],
                "workingday": features["workingday"],
                "target": target,
            }
        )
        self.group_means_ = (
            values.groupby(["hr", "workingday"])["target"].mean().to_dict()
        )
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return np.asarray(
            [
                self.group_means_.get((int(hour), int(working)), self.global_mean_)
                for hour, working in zip(features["hr"], features["workingday"])
            ],
            dtype=float,
        )


def build_model(model_id: str):
    if model_id == "starter":
        return GroupedMeanRegressor()
    if model_id == "ai-lead":
        return HistGradientBoostingRegressor(
            max_iter=200,
            learning_rate=0.08,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=SEED,
        )
    raise ValueError("model-id must be starter or ai-lead")


def train(model_id: str, *, replace: bool = False) -> Path:
    processed = DATA / "processed" / "tabular"
    train_path = processed / "train.csv"
    manifest_path = processed / "manifest.json"
    if not train_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(
            "Run `python -m ml.tabular.cli prepare` before training"
        )
    data = pd.read_csv(train_path)
    validation = pd.read_csv(processed / "validation.csv")
    if set(FEATURES) & EXCLUDED:
        raise ValueError("Forbidden target or identifier leakage in feature list")
    model_path, _ = model_paths("tabular", model_id)
    if model_path.exists() and not replace:
        raise FileExistsError(
            "A fixed artifact already exists; use --replace explicitly to replace it"
        )
    model = build_model(model_id)
    model.fit(data[FEATURES], data["cnt"])
    validation_predictions = np.maximum(model.predict(validation[FEATURES]), 0.0)
    validation_metrics = regression_metrics(
        validation["cnt"].to_numpy(dtype=float), validation_predictions
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return save_sklearn_artifact(
        "tabular",
        model_id,
        model,
        {
            "dataset": manifest["source"],
            "dataset_sha256": manifest["archive_sha256"],
            "split_manifest": str((processed / "manifest.json").relative_to(ROOT)),
            "split_manifest_sha256": sha256_file(manifest_path),
            "seed": SEED,
            "training_rows": len(data),
            "hardware": "CPU",
            "metrics": {"validation": validation_metrics},
            "features": FEATURES,
            "training_feature_ranges": {
                feature: {
                    "min": float(data[feature].min()),
                    "max": float(data[feature].max()),
                }
                for feature in FEATURES
            },
            "excluded_columns": sorted(EXCLUDED),
            "config": {"estimator": str(model), "training_split_only": True},
        },
    )


def evaluate(model_id: str, split_name: str, *, organiser: bool = False) -> dict:
    if split_name not in {"validation", "test"}:
        raise ValueError("split must be validation or test")
    if split_name == "test" and not organiser:
        raise PermissionError("Final test evaluation requires --organiser")
    from ml.common.artifacts import model_paths

    model_path, _ = model_paths("tabular", model_id)
    split_path = DATA / "processed" / "tabular" / f"{split_name}.csv"
    if not model_path.is_file() or not split_path.is_file():
        raise FileNotFoundError(
            "Model or requested split is missing; prepare data and train the model first"
        )
    import joblib

    model = joblib.load(model_path)
    data = pd.read_csv(split_path)
    predictions = model.predict(data[FEATURES])
    predictions = np.maximum(predictions, 0.0)
    metrics: dict[str, Any] = regression_metrics(
        data["cnt"].to_numpy(dtype=float), predictions
    )
    metrics["prediction_unit"] = "rentals per hour"
    metrics["errors_by_hour"] = {
        str(hour): regression_metrics(
            group["cnt"].to_numpy(dtype=float),
            np.maximum(model.predict(group[FEATURES]), 0.0),
        )["mae"]
        for hour, group in data.assign(_pred=predictions).groupby("hr")
    }
    metrics["predicted_actual_sample"] = [
        {"predicted": float(pred), "actual": float(actual)}
        for pred, actual in zip(
            predictions[:500], data["cnt"].to_numpy(dtype=float)[:500]
        )
    ]
    result = {
        "task": "tabular",
        "model_id": model_id,
        "split": split_name,
        "metrics": metrics,
    }
    output_dir = REPORTS / ("private-test" if split_name == "test" else "validation")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"tabular-{model_id}.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def summarize() -> Path:
    from ml.common.summary import generate_frontend_metrics

    return generate_frontend_metrics()


def export_student_data() -> Path:
    processed = DATA / "processed" / "tabular"
    manifest_path = processed / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            "Run `python -m ml.tabular.cli prepare` before exporting student data"
        )
    output = DATA / "student" / "tabular"
    output.mkdir(parents=True, exist_ok=True)
    for split_name in ("train", "validation"):
        (output / f"{split_name}.csv").write_bytes(
            (processed / f"{split_name}.csv").read_bytes()
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["counts"] = {
        key: manifest["counts"][key] for key in ("train", "validation")
    }
    if "split_sha256" in manifest:
        manifest["split_sha256"] = {
            key: manifest["split_sha256"][key] for key in ("train", "validation")
        }
    manifest["test_split_included"] = False
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return output
