from __future__ import annotations

import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy
import pandas
import sklearn

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts"
DATA = ROOT / "data"
REPORTS = ROOT / "reports"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def runtime_versions() -> dict[str, str]:
    versions = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "scikit_learn": sklearn.__version__,
    }
    for package in ("joblib", "Pillow", "fastapi", "pydantic", "torch", "torchvision"):
        try:
            versions[package.lower().replace("-", "_")] = version(package)
        except PackageNotFoundError:
            continue
    return versions


def model_paths(task: str, model_id: str) -> tuple[Path, Path]:
    folder = ARTIFACTS / task
    return folder / f"{model_id}.joblib", folder / f"{model_id}.json"


def save_sklearn_artifact(
    task: str, model_id: str, estimator: Any, metadata: dict[str, Any]
) -> Path:
    model_path, metadata_path = model_paths(task, model_id)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(estimator, model_path)
    metadata = {
        **metadata,
        "task": task,
        "model_id": model_id,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": runtime_versions(),
        "artifact_sha256": sha256_file(model_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return model_path


def load_metadata(task: str, model_id: str) -> dict[str, Any] | None:
    _, path = model_paths(task, model_id)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
