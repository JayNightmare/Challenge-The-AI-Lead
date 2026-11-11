from __future__ import annotations

from io import BytesIO
import json

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from pydantic import ValidationError

import backend.app.main as api
from backend.app.submissions import parse_prediction_csv
from backend.app.schemas import BikeRequest, TextRequest
from ml.common.row_ids import row_ids_for
from ml.tabular.pipeline import FEATURES


def test_text_request_rejects_empty_and_oversized_input() -> None:
    with pytest.raises(ValidationError):
        TextRequest(text="   ")
    with pytest.raises(ValidationError):
        TextRequest(text="x" * 5001)


def test_bike_request_rejects_invalid_ranges_and_extra_fields() -> None:
    with pytest.raises(ValidationError):
        BikeRequest(
            season=1,
            year=0,
            month=1,
            hour=24,
            holiday=0,
            weekday=0,
            workingday=0,
            weathersit=1,
            temperature_c=20,
            feels_like_c=20,
            humidity_percent=50,
            wind_speed_kmh=10,
        )
    with pytest.raises(ValidationError):
        BikeRequest(
            season=1,
            year=0,
            month=1,
            hour=1,
            holiday=0,
            weekday=0,
            workingday=0,
            weathersit=1,
            temperature_c=20,
            feels_like_c=20,
            humidity_percent=50,
            wind_speed_kmh=10,
            cnt=3,
        )


def test_bike_physical_inputs_convert_to_uci_encodings() -> None:
    payload = BikeRequest(
        season=1,
        year=1,
        month=8,
        hour=12,
        holiday=0,
        weekday=2,
        workingday=1,
        weathersit=2,
        temperature_c=20.5,
        feels_like_c=25,
        humidity_percent=60,
        wind_speed_kmh=33.5,
    )
    features = payload.dataset_features()
    assert features["temp"] == 0.5
    assert features["atemp"] == 0.5
    assert features["hum"] == 0.6
    assert features["windspeed"] == 0.5
    assert "cnt" not in features


def test_health_and_prediction_report_missing_artifacts(monkeypatch) -> None:
    monkeypatch.setattr(
        api,
        "load_local_models",
        lambda: {"text": {}, "tabular": {}, "vision": {}, "_errors": {}},
    )
    with TestClient(api.app) as client:
        health = client.get("/api/health").json()
        assert health["tasks"]["text"]["status"] == "model_not_trained"
        response = client.post(
            "/api/predict/text", json={"text": "hello", "model_id": "starter"}
        )
        assert response.status_code == 503
        assert response.json()["detail"]["status"] == "model_not_trained"
        assert "next_command" in response.json()["detail"]


def test_prediction_contract_and_request_not_persisted(tmp_path, monkeypatch) -> None:
    class LocalTextModel:
        classes_ = ["ham", "spam"]

        def predict(self, rows):
            return ["ham"]

        def predict_proba(self, rows):
            return [[0.8, 0.2]]

    monkeypatch.setattr(
        api,
        "load_local_models",
        lambda: {
            "text": {
                "starter": {
                    "estimator": LocalTextModel(),
                    "metadata": {"trained_at_utc": "test-local"},
                }
            },
            "tabular": {},
            "vision": {},
            "_errors": {},
        },
    )
    monkeypatch.setattr(api, "REPORTS", tmp_path)
    with TestClient(api.app) as client:
        response = client.post(
            "/api/predict/text",
            json={"text": "private test message", "model_id": "starter"},
        )
        assert response.status_code == 200
        assert response.json()["prediction"] == "ham"
        assert (
            response.json()["score_label"]
            == "model score; not a calibrated probability"
        )
    assert list(tmp_path.rglob("*")) == []


def test_corrupt_local_model_is_reported_as_artifact_error(
    tmp_path, monkeypatch
) -> None:
    corrupt = tmp_path / "broken.joblib"
    corrupt.write_bytes(b"not a model")
    monkeypatch.setattr(
        api,
        "model_paths",
        lambda task, model_id: (corrupt, corrupt.with_suffix(".json")),
    )
    monkeypatch.setattr(
        api, "load_metadata", lambda task, model_id: {"artifact_sha256": "bad"}
    )
    loaded = api.load_local_models()
    assert not loaded["text"]
    assert loaded["_errors"]["text"] == ["starter", "ai-lead"]


def test_vision_form_selects_requested_model_and_does_not_save_image(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(
        api,
        "load_local_models",
        lambda: {"text": {}, "tabular": {}, "vision": {}, "_errors": {}},
    )
    image = BytesIO()
    Image.new("RGB", (12, 12), "red").save(image, format="PNG")
    with TestClient(api.app) as client:
        response = client.post(
            "/api/predict/vision",
            files={"file": ("item.png", image.getvalue(), "image/png")},
            data={"model_id": "starter"},
        )
    assert response.status_code == 503
    assert response.json()["detail"]["model_id"] == "starter"
    assert not list(tmp_path.rglob("*.png"))


def test_vision_rejects_bad_type_oversize_and_dimensions(monkeypatch) -> None:
    monkeypatch.setattr(
        api,
        "load_local_models",
        lambda: {"text": {}, "tabular": {}, "vision": {}, "_errors": {}},
    )
    too_wide = BytesIO()
    Image.new("RGB", (api.MAX_IMAGE_EDGE + 1, 1), "blue").save(too_wide, format="PNG")
    with TestClient(api.app) as client:
        bad_type = client.post(
            "/api/predict/vision",
            files={"file": ("item.txt", b"not an image", "text/plain")},
        )
        oversized = client.post(
            "/api/predict/vision",
            files={
                "file": (
                    "large.png",
                    b"x" * (api.MAX_IMAGE_BYTES + 1),
                    "image/png",
                )
            },
        )
        dimensions = client.post(
            "/api/predict/vision",
            files={"file": ("wide.png", too_wide.getvalue(), "image/png")},
        )
    assert bad_type.status_code == 415
    assert oversized.status_code == 413
    assert dimensions.status_code == 413


def test_vision_example_endpoint_serves_only_approved_validation_rows(
    tmp_path, monkeypatch
) -> None:
    data_root = tmp_path / "data"
    image_root = data_root / "raw" / "equipment"
    processed = data_root / "processed" / "vision"
    reports = tmp_path / "reports" / "validation"
    image_root.mkdir(parents=True)
    processed.mkdir(parents=True)
    reports.mkdir(parents=True)
    image_bytes = BytesIO()
    Image.new("RGB", (8, 8), "green").save(image_bytes, format="PNG")
    (image_root / "approved.png").write_bytes(image_bytes.getvalue())
    (processed / "validation.csv").write_text(
        "image_path,label,physical_object_id,collection_session\napproved.png,mouse,mouse-1,session-1\n",
        encoding="utf-8",
    )
    (reports / "vision-ai-lead.json").write_text(
        json.dumps(
            {
                "split": "validation",
                "representative_mistakes": [
                    {"row_index": 0, "expected": "mouse", "predicted": "keyboard"}
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(api, "DATA", data_root)
    monkeypatch.setattr(api, "REPORTS", tmp_path / "reports")
    with TestClient(api.app) as client:
        approved = client.get("/api/vision/validation-examples/0/image")
        unlisted = client.get("/api/vision/validation-examples/1/image")
    assert approved.status_code == 200
    assert approved.headers["content-type"] == "image/png"
    assert approved.content == image_bytes.getvalue()
    assert unlisted.status_code == 404


def test_prediction_csv_parser_enforces_schema_ids_count_and_uniqueness() -> None:
    valid = b"row_id,prediction\na,ham\nb,spam\n"
    parsed = parse_prediction_csv(valid, ["a", "b"])
    assert parsed["prediction"].tolist() == ["ham", "spam"]
    invalid_files = [
        b"row_id,prediction,extra\na,ham,x\nb,spam,x\n",
        b"row_id,prediction\na,ham\na,spam\n",
        b"row_id,prediction\na,ham\nc,spam\n",
        b"row_id,prediction\na,ham\n",
    ]
    for contents in invalid_files:
        with pytest.raises(ValueError):
            parse_prediction_csv(contents, ["a", "b"])


def test_text_submission_scores_only_validation_ids_and_is_not_saved(
    tmp_path, monkeypatch
) -> None:
    data_root = tmp_path / "data"
    processed = data_root / "processed" / "text"
    processed.mkdir(parents=True)
    validation = pd.DataFrame(
        {
            "label": ["ham", "spam"],
            "text": ["ordinary text", "promo text"],
            "group_id": ["group-a", "group-b"],
        }
    )
    validation.to_csv(processed / "validation.csv", index=False)
    row_ids = row_ids_for("text", validation, "validation")
    (processed / "train.csv").write_text("label,text,group_id\nham,x,a\n")
    monkeypatch.setattr(api, "DATA", data_root)

    class LocalTextModel:
        def predict(self, rows):
            return ["ham", "spam"]

    def loaded_models():
        return {
            "text": {
                model_id: {"estimator": LocalTextModel(), "metadata": {}}
                for model_id in ("starter", "ai-lead")
            },
            "tabular": {},
            "vision": {},
            "_errors": {},
        }

    monkeypatch.setattr(api, "load_local_models", loaded_models)
    csv_contents = (
        "row_id,prediction\n" f"{row_ids[0]},ham\n{row_ids[1]},spam\n"
    ).encode()
    with TestClient(api.app) as client:
        template = client.get("/api/submissions/text/template")
        response = client.post(
            "/api/submissions/text",
            files={"file": ("predictions.csv", csv_contents, "text/csv")},
        )
        invalid = client.post(
            "/api/submissions/text",
            files={
                "file": (
                    "predictions.csv",
                    csv_contents.replace(row_ids[0].encode(), b"unknown-id"),
                    "text/csv",
                )
            },
        )
        model_file = client.post(
            "/api/submissions/text",
            files={
                "file": ("model.joblib", b"arbitrary bytes", "application/octet-stream")
            },
        )
    assert template.status_code == 200
    assert template.text.startswith("row_id,prediction")
    assert "ordinary text" not in template.text and "ham" not in template.text
    assert response.status_code == 200
    assert response.json()["split"] == "validation"
    assert response.json()["metrics"]["macro_f1"] == 1.0
    assert response.json()["private_test_used"] is False
    assert response.json()["prediction_file_saved"] is False
    assert invalid.status_code == 422
    assert model_file.status_code == 415
    assert not list(data_root.rglob("predictions.csv"))


def test_bike_submission_rejects_nonfinite_and_clamps_negative_predictions(
    tmp_path, monkeypatch
) -> None:
    data_root = tmp_path / "data"
    processed = data_root / "processed" / "tabular"
    processed.mkdir(parents=True)
    validation_data = {feature: [1.0, 1.0] for feature in FEATURES}
    validation_data.update(
        {
            "hr": [1, 2],
            "workingday": [0, 1],
            "dteday": ["2011-01-01", "2011-01-01"],
            "cnt": [5, 10],
        }
    )
    validation = pd.DataFrame(validation_data)
    validation.to_csv(processed / "validation.csv", index=False)
    row_ids = row_ids_for("tabular", validation, "validation")
    monkeypatch.setattr(api, "DATA", data_root)

    class LocalRegressor:
        def predict(self, features):
            return np.array([-3.0, 8.0])

    def loaded_models():
        return {
            "text": {},
            "tabular": {
                model_id: {"estimator": LocalRegressor(), "metadata": {}}
                for model_id in ("starter", "ai-lead")
            },
            "vision": {},
            "_errors": {},
        }

    monkeypatch.setattr(api, "load_local_models", loaded_models)
    valid_csv = ("row_id,prediction\n" f"{row_ids[0]},-1\n{row_ids[1]},12\n").encode()
    invalid_csv = valid_csv.replace(b",12\n", b",inf\n")
    with TestClient(api.app) as client:
        valid = client.post(
            "/api/submissions/tabular",
            files={"file": ("predictions.csv", valid_csv, "text/csv")},
        )
        invalid = client.post(
            "/api/submissions/tabular",
            files={"file": ("predictions.csv", invalid_csv, "text/csv")},
        )
    assert valid.status_code == 200
    assert valid.json()["metrics"]["mae"] == 3.5
    assert valid.json()["metrics"]["negative_outputs_clamped_to_zero"] is True
    assert invalid.status_code == 422
