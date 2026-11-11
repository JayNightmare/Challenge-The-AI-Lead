from __future__ import annotations

import csv
import json
import zipfile
from pathlib import Path
from typing import Any, cast
from urllib.request import urlopen

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline

from ml.common.artifacts import (
    DATA,
    ROOT,
    REPORTS,
    model_paths,
    save_sklearn_artifact,
    sha256_file,
)
from ml.common.metrics import classification_metrics
from ml.common.row_ids import row_ids_for
from ml.common.splits import grouped_stratified_splits, normalized_text_hash

SOURCE_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
SOURCE_INFO = {
    "name": "SMS Spam Collection",
    "repository": "UCI Machine Learning Repository",
    "dataset_id": 228,
    "url": "https://archive.ics.uci.edu/dataset/228/sms+spam+collection",
    "license": "CC BY 4.0",
    "attribution": "Almeida, T. A., Gómez Hidalgo, J. M., and Yamakami, A. (2011). Contributions to the Study of SMS Spam Filtering: New Collection and Results.",
}
SEED = 42


def download_source(destination: Path | None = None) -> Path:
    destination = destination or DATA / "raw" / "sms+spam+collection.zip"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        with urlopen(SOURCE_URL, timeout=60) as response, destination.open(
            "wb"
        ) as output:
            output.write(response.read())
    if not zipfile.is_zipfile(destination):
        destination.unlink(missing_ok=True)
        raise ValueError("Downloaded UCI SMS archive is not a valid ZIP file")
    return destination


def parse_source(archive: Path) -> pd.DataFrame:
    with zipfile.ZipFile(archive) as source:
        candidates = [
            name for name in source.namelist() if name.endswith("SMSSpamCollection")
        ]
        if len(candidates) != 1:
            raise ValueError(
                "Expected exactly one SMSSpamCollection file in the UCI archive"
            )
        rows: list[tuple[str, str]] = []
        with source.open(candidates[0]) as raw:
            for row_number, row in enumerate(
                csv.reader((line.decode("utf-8") for line in raw), delimiter="\t"), 1
            ):
                if len(row) != 2 or row[0] not in {"ham", "spam"} or not row[1].strip():
                    raise ValueError(
                        f"Unexpected UCI SMS row format at row {row_number}"
                    )
                rows.append((row[0], row[1]))
    if not rows:
        raise ValueError("The UCI SMS file is empty")
    return pd.DataFrame(rows, columns=["label", "text"])


def prepare_data() -> dict[str, int]:
    archive = download_source()
    frame = parse_source(archive)
    frame["group_id"] = frame["text"].map(normalized_text_hash)
    splits = grouped_stratified_splits(
        frame, label_column="label", group_column="group_id", seed=SEED
    )
    output = DATA / "processed" / "text"
    output.mkdir(parents=True, exist_ok=True)
    for name, split in splits.items():
        split[["label", "text", "group_id"]].to_csv(output / f"{name}.csv", index=False)
    manifest = {
        "source": SOURCE_INFO,
        "archive_sha256": sha256_file(archive),
        "seed": SEED,
        "split_method": "StratifiedGroupKFold, 10 folds; duplicate normalized message hashes stay together",
        "split_sha256": {name: sha256_file(output / f"{name}.csv") for name in splits},
        "counts": {name: len(split) for name, split in splits.items()},
        "group_counts": {
            name: int(split["group_id"].nunique()) for name, split in splits.items()
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    (output / "SOURCE.txt").write_text(
        json.dumps(SOURCE_INFO, indent=2) + "\n", encoding="utf-8"
    )
    return manifest["counts"]


def build_model(model_id: str) -> Pipeline:
    if model_id == "starter":
        features = TfidfVectorizer(
            ngram_range=(1, 2), min_df=1, strip_accents="unicode"
        )
        classifier = MultinomialNB(alpha=1.0)
    elif model_id == "ai-lead":
        transformers: list[tuple[str, Any]] = [
            (
                "word",
                TfidfVectorizer(ngram_range=(1, 2), min_df=1, strip_accents="unicode"),
            ),
            (
                "character",
                TfidfVectorizer(
                    analyzer="char", ngram_range=(3, 5), min_df=2, sublinear_tf=True
                ),
            ),
        ]
        features = FeatureUnion(cast(Any, transformers))
        classifier = LogisticRegression(C=2.0, max_iter=2000, random_state=SEED)
    else:
        raise ValueError("model-id must be starter or ai-lead")
    return Pipeline([("features", features), ("classifier", classifier)])


def train(model_id: str, *, replace: bool = False) -> Path:
    processed = DATA / "processed" / "text"
    train_path = processed / "train.csv"
    manifest_path = processed / "manifest.json"
    if not train_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("Run `python -m ml.text.cli prepare` before training")
    model_path, _ = model_paths("text", model_id)
    if model_path.exists() and not replace:
        raise FileExistsError(
            "A fixed artifact already exists; use --replace explicitly to replace it"
        )
    data = pd.read_csv(train_path)
    validation = pd.read_csv(processed / "validation.csv")
    model = build_model(model_id)
    model.fit(data["text"].astype(str), data["label"].astype(str))
    validation_metrics = classification_metrics(
        validation["label"].astype(str).tolist(),
        np.asarray(model.predict(validation["text"].astype(str))).astype(str).tolist(),
        ["ham", "spam"],
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return save_sklearn_artifact(
        "text",
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
            "config": {"pipeline": str(model), "training_split_only": True},
        },
    )


def evaluate(model_id: str, split_name: str, *, organiser: bool = False) -> dict:
    if split_name not in {"validation", "test"}:
        raise ValueError("split must be validation or test")
    if split_name == "test" and not organiser:
        raise PermissionError("Final test evaluation requires --organiser")
    from ml.common.artifacts import model_paths

    model_path, _ = model_paths("text", model_id)
    split_path = DATA / "processed" / "text" / f"{split_name}.csv"
    if not model_path.is_file() or not split_path.is_file():
        raise FileNotFoundError(
            "Model or requested split is missing; prepare data and train the model first"
        )
    import joblib

    model = joblib.load(model_path)
    data = pd.read_csv(split_path)
    training = pd.read_csv(DATA / "processed" / "text" / "train.csv")
    labels = ["ham", "spam"]
    majority_label = str(training["label"].value_counts().idxmax())
    y_true = data["label"].astype(str).tolist()
    result = {
        "task": "text",
        "model_id": model_id,
        "split": split_name,
        "metrics": classification_metrics(
            y_true,
            model.predict(data["text"].astype(str)).tolist(),
            labels,
        ),
        "reference": {
            "name": "training majority-class baseline",
            "predicted_class": majority_label,
            "metrics": classification_metrics(
                y_true, [majority_label] * len(y_true), labels
            ),
        },
    }
    output_dir = REPORTS / ("private-test" if split_name == "test" else "validation")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"text-{model_id}.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def summarize() -> Path:
    from ml.common.summary import generate_frontend_metrics

    return generate_frontend_metrics()


def export_student_data() -> Path:
    processed = DATA / "processed" / "text"
    manifest_path = processed / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            "Run `python -m ml.text.cli prepare` before exporting student data"
        )
    output = DATA / "student" / "text"
    output.mkdir(parents=True, exist_ok=True)
    for split_name in ("train", "validation"):
        split = pd.read_csv(processed / f"{split_name}.csv")
        split.insert(0, "row_id", row_ids_for("text", split, split_name))
        split.to_csv(output / f"{split_name}.csv", index=False)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["counts"] = {
        key: manifest["counts"][key] for key in ("train", "validation")
    }
    manifest["group_counts"] = {
        key: manifest["group_counts"][key] for key in ("train", "validation")
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
