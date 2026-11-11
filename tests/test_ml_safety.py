from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from ml.common.artifacts import load_metadata, model_paths, save_sklearn_artifact
from ml.common.row_ids import row_ids_for
from ml.common.splits import chronological_splits, grouped_stratified_splits
from ml.tabular.pipeline import EXCLUDED, FEATURES, GroupedMeanRegressor
from ml.text.pipeline import build_model
from ml.vision.pipeline import CLASSES, split_by_object, validate_manifest


def test_grouped_split_keeps_duplicate_groups_together() -> None:
    rows = []
    for label in ("ham", "spam"):
        for index in range(50):
            text = f"{label} unique message {index}"
            group = f"{label}-{index}"
            rows.extend([{"label": label, "text": text, "group": group}] * 2)
    splits = grouped_stratified_splits(
        pd.DataFrame(rows), label_column="label", group_column="group", seed=42
    )
    groups = {name: set(frame["group"]) for name, frame in splits.items()}
    assert not (groups["train"] & groups["validation"])
    assert not (groups["train"] & groups["test"])
    assert not (groups["validation"] & groups["test"])


def test_bike_features_exclude_target_and_derived_leaks() -> None:
    assert not (set(FEATURES) & EXCLUDED)
    assert {"cnt", "casual", "registered", "instant"} <= EXCLUDED


def test_chronological_split_orders_boundaries() -> None:
    frame = pd.DataFrame(
        {
            "dteday": pd.date_range("2012-01-01", periods=100, freq="h").date,
            "hr": list(range(100)),
            "cnt": np.arange(100),
        }
    )
    splits = chronological_splits(frame)
    assert tuple(splits["train"].iloc[-1][["dteday", "hr"]]) < tuple(
        splits["validation"].iloc[0][["dteday", "hr"]]
    )
    assert tuple(splits["validation"].iloc[-1][["dteday", "hr"]]) < tuple(
        splits["test"].iloc[0][["dteday", "hr"]]
    )


def test_text_models_are_pipelines_fitted_only_by_train_call() -> None:
    starter = build_model("starter")
    lead = build_model("ai-lead")
    assert starter.steps[0][0] == "features"
    assert lead.steps[0][0] == "features"
    assert not hasattr(starter.named_steps["features"], "vocabulary_")
    assert not hasattr(lead.named_steps["features"], "transformer_list") or not any(
        hasattr(transformer, "vocabulary_")
        for _, transformer in lead.named_steps["features"].transformer_list
    )


def test_grouped_mean_falls_back_to_training_global_mean() -> None:
    model = GroupedMeanRegressor().fit(
        pd.DataFrame({"hr": [1, 1], "workingday": [0, 0]}), pd.Series([10, 20])
    )
    predicted = model.predict(pd.DataFrame({"hr": [1, 9], "workingday": [0, 1]}))
    assert predicted.tolist() == [15.0, 15.0]


def test_artifact_metadata_round_trip(tmp_path, monkeypatch) -> None:
    import ml.common.artifacts as artifacts

    monkeypatch.setattr(artifacts, "ARTIFACTS", tmp_path)
    model = build_model("starter")
    model.fit(["hello class", "offer now"], ["ham", "spam"])
    path = save_sklearn_artifact("test", "starter", model, {"seed": 42})
    assert path.is_file()
    assert load_metadata("test", "starter")["seed"] == 42
    loaded_path, _ = model_paths("test", "starter")
    import joblib

    assert joblib.load(loaded_path).predict(["hello class"]).tolist() == ["ham"]


def test_private_test_evaluation_requires_organiser() -> None:
    from ml.text.pipeline import evaluate

    with pytest.raises(PermissionError):
        evaluate("starter", "test")


def test_split_manifest_does_not_store_message_text() -> None:
    source = json.dumps({"seed": 42, "counts": {"train": 1}})
    assert "text" not in source


def test_vision_object_splits_are_disjoint() -> None:
    rows = [
        {
            "image_path": f"{label}-{index}.png",
            "label": label,
            "physical_object_id": f"{label}-{index}",
            "collection_session": "session-a",
        }
        for label in CLASSES
        for index in range(5)
    ]
    splits = split_by_object(pd.DataFrame(rows), seed=42)
    objects = {name: set(split["physical_object_id"]) for name, split in splits.items()}
    assert not (objects["train"] & objects["validation"])
    assert not (objects["train"] & objects["test"])
    assert not (objects["validation"] & objects["test"])


def test_vision_manifest_rejects_duplicate_image_bytes(tmp_path) -> None:
    rows = []
    for label in CLASSES:
        for index in range(5):
            relative = f"{label}-{index}.png"
            Image.new("RGB", (8, 8), (index, CLASSES.index(label), 1)).save(
                tmp_path / relative
            )
            rows.append(
                {
                    "image_path": relative,
                    "label": label,
                    "physical_object_id": f"{label}-{index}",
                    "collection_session": "session-a",
                }
            )
    (tmp_path / "duplicate.png").write_bytes((tmp_path / "keyboard-0.png").read_bytes())
    rows[1]["image_path"] = "duplicate.png"
    with pytest.raises(ValueError, match="Duplicate image"):
        validate_manifest(pd.DataFrame(rows), tmp_path)


def test_student_export_omits_private_test_data_and_checksums(
    tmp_path, monkeypatch
) -> None:
    from ml.text import pipeline as text_pipeline

    monkeypatch.setattr(text_pipeline, "DATA", tmp_path)
    processed = tmp_path / "processed" / "text"
    processed.mkdir(parents=True)
    for split_name in ("train", "validation", "test"):
        (processed / f"{split_name}.csv").write_text(
            "label,text,group_id\nham,example,hash\n", encoding="utf-8"
        )
    (processed / "manifest.json").write_text(
        json.dumps(
            {
                "counts": {"train": 1, "validation": 1, "test": 1},
                "group_counts": {"train": 1, "validation": 1, "test": 1},
                "split_sha256": {"train": "a", "validation": "b", "test": "private"},
            }
        ),
        encoding="utf-8",
    )
    output = text_pipeline.export_student_data()
    exported = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert set(exported["counts"]) == {"train", "validation"}
    assert set(exported["split_sha256"]) == {"train", "validation"}
    assert not (output / "test.csv").exists()
    assert exported["test_split_included"] is False
    exported_validation = pd.read_csv(output / "validation.csv")
    assert exported_validation["row_id"].is_unique


def test_submission_row_ids_are_stable_and_unique() -> None:
    sms = pd.DataFrame({"group_id": ["duplicate", "duplicate", "other"]})
    first = row_ids_for("text", sms, "validation")
    assert first == row_ids_for("text", sms, "validation")
    assert len(set(first)) == len(sms)
    bike = pd.DataFrame({"dteday": ["2011-01-01", "2011-01-01"], "hr": [1, 2]})
    assert len(set(row_ids_for("tabular", bike, "validation"))) == 2
