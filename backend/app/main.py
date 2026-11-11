from __future__ import annotations

import io
import json
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError

from backend.app.schemas import BikeRequest, TextRequest
from ml.common.artifacts import DATA, REPORTS, load_metadata, model_paths
from ml.tabular.pipeline import FEATURES

MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000
MAX_IMAGE_EDGE = 6000
LOCAL_ORIGINS = [
    "http://127.0.0.1:5173",
    "http://localhost:5173",
    "http://127.0.0.1:4173",
    "http://localhost:4173",
]
TASK_MODEL_IDS = {
    "text": ["starter", "ai-lead"],
    "tabular": ["starter", "ai-lead"],
    "vision": ["starter", "ai-lead"],
}


def load_local_models() -> dict[str, dict[str, Any]]:
    loaded: dict[str, dict[str, Any]] = {task: {} for task in TASK_MODEL_IDS}
    errors: dict[str, list[str]] = {task: [] for task in TASK_MODEL_IDS}
    for task, model_ids in TASK_MODEL_IDS.items():
        for model_id in model_ids:
            if task == "vision":
                path = DATA.parent / "artifacts" / "vision" / f"{model_id}.pt"
                if not path.is_file():
                    continue
                try:
                    from ml.vision.pipeline import load_model

                    estimator, metadata = load_model(model_id)
                except Exception:
                    errors[task].append(model_id)
                    continue
            else:
                path, _ = model_paths(task, model_id)
                metadata = load_metadata(task, model_id)
                if not path.is_file() or metadata is None:
                    continue
                try:
                    if metadata.get("artifact_sha256"):
                        from ml.common.artifacts import sha256_file

                        if sha256_file(path) != metadata["artifact_sha256"]:
                            raise ValueError("Local artifact checksum mismatch")
                    estimator = joblib.load(path)
                except Exception:
                    errors[task].append(model_id)
                    continue
            loaded[task][model_id] = {"estimator": estimator, "metadata": metadata}
    loaded["_errors"] = errors
    return loaded


@asynccontextmanager
async def lifespan(app: FastAPI):
    loaded = load_local_models()
    app.state.model_errors = loaded.pop("_errors", {})
    app.state.models = loaded
    yield
    app.state.models.clear()


app = FastAPI(title="Everyday AI local demo", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=LOCAL_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


def _models(request: Request, task: str) -> dict[str, dict[str, Any]]:
    return request.app.state.models.get(task, {})


def _require_model(request: Request, task: str, model_id: str) -> dict[str, Any]:
    model = _models(request, task).get(model_id)
    if model is None:
        raise HTTPException(
            status_code=503,
            detail={
                "status": _readiness(request, task)["status"],
                "task": task,
                "model_id": model_id,
                "next_command": _readiness(request, task)["next_command"],
            },
        )
    return model


def _readiness(request: Request, task: str) -> dict[str, str]:
    if _models(request, task):
        return {"status": "ready", "next_command": ""}
    if request.app.state.model_errors.get(task):
        return {
            "status": "artifact_error",
            "next_command": f"python -m ml.{task}.cli train --model-id starter --replace",
        }
    processed = DATA / "processed" / task
    if not processed.is_dir():
        return {
            "status": "awaiting_data",
            "next_command": f"python -m ml.{task}.cli prepare",
        }
    return {
        "status": "model_not_trained",
        "next_command": f"python -m ml.{task}.cli train --model-id starter",
    }


def _read_validation_metrics(task: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for model_id in TASK_MODEL_IDS[task]:
        path = REPORTS / "validation" / f"{task}-{model_id}.json"
        if path.is_file():
            try:
                result[model_id] = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
    return result


@app.get("/api/health")
def health(request: Request) -> dict[str, Any]:
    models = request.app.state.models
    return {
        "status": "ok",
        "tasks": {
            task: {
                **_readiness(request, task),
                "available_models": sorted(models[task]),
                "artifact_error": bool(request.app.state.model_errors.get(task)),
            }
            for task in TASK_MODEL_IDS
        },
    }


@app.get("/api/metadata")
def metadata(request: Request) -> dict[str, Any]:
    results = {}
    for task in TASK_MODEL_IDS:
        results[task] = {
            "available_models": [
                {"model_id": model_id, "metadata": loaded["metadata"]}
                for model_id, loaded in _models(request, task).items()
            ],
            "validation_metrics": _read_validation_metrics(task),
            "artifact_error": bool(request.app.state.model_errors.get(task)),
        }
    return {"tasks": results}


@app.post("/api/predict/text")
def predict_text(payload: TextRequest, request: Request) -> dict[str, Any]:
    loaded = _require_model(request, "text", payload.model_id)
    model = loaded["estimator"]
    text = payload.text
    prediction = str(model.predict([text])[0])
    scores = {}
    if hasattr(model, "predict_proba"):
        scores = {
            str(label): float(score)
            for label, score in zip(model.classes_, model.predict_proba([text])[0])
        }
    return {
        "prediction": prediction,
        "scores": scores,
        "score_label": "model score; not a calibrated probability",
        "model_id": payload.model_id,
        "model_version": loaded["metadata"].get("trained_at_utc", "local artifact"),
        "caveat": "For human review only. Not a phishing detector or security assurance; may not generalize to modern messages.",
    }


@app.post("/api/predict/tabular")
def predict_tabular(payload: BikeRequest, request: Request) -> dict[str, Any]:
    features = payload.dataset_features()
    row = pd.DataFrame([{name: features[name] for name in FEATURES}], columns=FEATURES)
    selected = [payload.model_id] if payload.model_id else ["starter", "ai-lead"]
    predictions: dict[str, float] = {}
    out_of_range: set[str] = set()
    for model_id in selected:
        loaded = _models(request, "tabular").get(model_id)
        if loaded is None:
            continue
        value = float(loaded["estimator"].predict(row)[0])
        predictions[model_id] = max(0.0, value)
        for feature, limits in (
            loaded["metadata"].get("training_feature_ranges", {}).items()
        ):
            if not limits["min"] <= features[feature] <= limits["max"]:
                out_of_range.add(feature)
    if not predictions:
        _require_model(request, "tabular", selected[0])
    return {
        "predictions": predictions,
        "unit": "rentals per hour",
        "scenario": "historical scenario estimate",
        "warnings": [
            "Observed historical weather is not a weather forecast.",
            "Arbitrary calendar and weather combinations may be outside the training distribution.",
        ]
        + (
            [f"Outside training values for: {', '.join(sorted(out_of_range))}."]
            if out_of_range
            else []
        ),
    }


@app.post("/api/predict/vision")
async def predict_vision(
    request: Request,
    file: UploadFile = File(...),
    model_id: str = Form("ai-lead"),
) -> dict[str, Any]:
    if model_id not in {"starter", "ai-lead"}:
        raise HTTPException(
            status_code=422, detail="model_id must be starter or ai-lead"
        )
    if file.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(status_code=415, detail="Upload a JPEG, PNG, or WebP image")
    contents = await file.read(MAX_IMAGE_BYTES + 1)
    if len(contents) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds the 8 MiB limit")
    try:
        image = Image.open(io.BytesIO(contents))
        if image.format not in {"JPEG", "PNG", "WEBP"}:
            raise HTTPException(
                status_code=415, detail="Image content must be JPEG, PNG, or WebP"
            )
        if (
            image.width > MAX_IMAGE_EDGE
            or image.height > MAX_IMAGE_EDGE
            or image.width * image.height > MAX_IMAGE_PIXELS
        ):
            raise HTTPException(
                status_code=413, detail="Image dimensions exceed the allowed limit"
            )
        image = image.convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(
            status_code=422, detail="The uploaded file is not a readable image"
        ) from None
    loaded = _require_model(request, "vision", model_id)
    import torch
    from torchvision import transforms

    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    tensor = transform(image).unsqueeze(0)
    started = time.perf_counter()
    with torch.inference_mode():
        probability = loaded["estimator"](tensor).softmax(dim=1)[0].tolist()
    elapsed_ms = (time.perf_counter() - started) * 1000
    classes = loaded["metadata"]["classes"]
    scores = {label: float(score) for label, score in zip(classes, probability)}
    threshold = loaded["metadata"].get("low_score_warning_threshold")
    return {
        "prediction": max(scores, key=scores.get),
        "scores": scores,
        "score_label": "class score; not a calibrated probability",
        "model_id": model_id,
        "model_version": loaded["metadata"].get("trained_at_utc", "local artifact"),
        "inference_ms": round(elapsed_ms, 2),
        "low_score_warning": threshold is not None and max(scores.values()) < threshold,
        "caveat": "Scores do not guarantee correctness. Unfamiliar objects still receive a known-class prediction. Confirm category manually.",
        "image_saved": False,
    }


@app.get("/api/vision/validation-examples/{example_index}/image")
def vision_validation_example(example_index: int, model_id: str = "ai-lead"):
    if model_id not in {"starter", "ai-lead"}:
        raise HTTPException(
            status_code=422, detail="model_id must be starter or ai-lead"
        )
    report_path = REPORTS / "validation" / f"vision-{model_id}.json"
    split_path = DATA / "processed" / "vision" / "validation.csv"
    if not report_path.is_file() or not split_path.is_file():
        raise HTTPException(
            status_code=404, detail="Validation examples are unavailable"
        )
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        mistakes = report.get("representative_mistakes", [])
        if report.get("split") != "validation" or not any(
            item.get("row_index") == example_index for item in mistakes
        ):
            raise HTTPException(
                status_code=404, detail="Approved validation example not found"
            )
        rows = pd.read_csv(split_path)
        relative_path = rows.iloc[example_index]["image_path"]
    except (OSError, json.JSONDecodeError, IndexError, KeyError):
        raise HTTPException(
            status_code=404, detail="Approved validation example not found"
        ) from None
    image_root = (DATA / "raw" / "equipment").resolve()
    image_path = (image_root / str(relative_path)).resolve()
    if (
        image_root not in image_path.parents
        or image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}
        or not image_path.is_file()
    ):
        raise HTTPException(
            status_code=404, detail="Approved validation example not found"
        )
    return FileResponse(image_path)
