from __future__ import annotations

import hashlib
import json
import random
import shutil
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from ml.common.artifacts import DATA, ROOT, REPORTS, sha256_file
from ml.common.metrics import classification_metrics

CLASSES = ["keyboard", "mouse", "headphones"]
SEED = 42
MANIFEST_FIELDS = ["image_path", "label", "physical_object_id", "collection_session"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
WARNING_TARGET_ACCURACY = 0.8


def image_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_manifest(frame: pd.DataFrame, image_root: Path) -> pd.DataFrame:
    missing = set(MANIFEST_FIELDS) - set(frame.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
    frame = frame[MANIFEST_FIELDS].copy()
    if (
        frame.isna().any().any()
        or (frame[MANIFEST_FIELDS].astype(str).apply(lambda col: col.str.strip() == ""))
        .any()
        .any()
    ):
        raise ValueError("Manifest fields must not be empty")
    unknown_classes = set(frame["label"]) - set(CLASSES)
    if unknown_classes:
        raise ValueError(f"Unknown equipment classes: {sorted(unknown_classes)}")
    objects_per_class = frame.groupby("label")["physical_object_id"].nunique()
    absent = set(CLASSES) - set(objects_per_class.index)
    if absent:
        raise ValueError(f"Missing required classes: {sorted(absent)}")
    if frame.groupby("physical_object_id")["label"].nunique().gt(1).any():
        raise ValueError("A physical_object_id cannot be assigned to multiple classes")

    resolved_paths: list[str] = []
    hashes: set[str] = set()
    root = image_root.resolve()
    for relative in frame["image_path"].astype(str):
        path = (root / relative).resolve()
        if (
            root not in path.parents
            or path.suffix.lower() not in IMAGE_EXTENSIONS
            or not path.is_file()
        ):
            raise ValueError(f"Invalid or missing image path: {relative}")
        digest = image_hash(path)
        if digest in hashes:
            raise ValueError("Duplicate image file contents detected")
        hashes.add(digest)
        with Image.open(path) as image:
            image.verify()
        resolved_paths.append(path.relative_to(root).as_posix())
    frame["image_path"] = resolved_paths
    if any(int(objects_per_class.get(label, 0)) < 3 for label in CLASSES):
        raise ValueError(
            "At least 3 distinct physical objects per class are needed for object-disjoint splits"
        )
    if any(int(objects_per_class.get(label, 0)) < 5 for label in CLASSES):
        warnings.warn(
            "Fewer than 5 distinct objects in at least one class; generalisation assessment is weak",
            stacklevel=2,
        )
    return frame


def split_by_object(frame: pd.DataFrame, seed: int = SEED) -> dict[str, pd.DataFrame]:
    rng = random.Random(seed)
    assignments: dict[str, str] = {}
    for label in CLASSES:
        object_ids = sorted(
            frame.loc[frame["label"] == label, "physical_object_id"].unique()
        )
        rng.shuffle(object_ids)
        count = len(object_ids)
        if count < 3:
            raise ValueError(f"Class {label} needs at least 3 distinct objects")
        train_count = max(1, int(count * 0.70))
        validation_count = max(1, int(count * 0.15))
        if train_count + validation_count >= count:
            train_count, validation_count = count - 2, 1
        for object_id in object_ids[:train_count]:
            assignments[object_id] = "train"
        for object_id in object_ids[train_count : train_count + validation_count]:
            assignments[object_id] = "validation"
        for object_id in object_ids[train_count + validation_count :]:
            assignments[object_id] = "test"
    frame = frame.copy()
    frame["split"] = frame["physical_object_id"].map(assignments)
    splits = {
        name: frame.loc[frame["split"] == name]
        .drop(columns="split")
        .reset_index(drop=True)
        for name in ("train", "validation", "test")
    }
    object_sets = {
        name: set(split["physical_object_id"]) for name, split in splits.items()
    }
    if any(
        object_sets[left] & object_sets[right]
        for left, right in (
            ("train", "validation"),
            ("train", "test"),
            ("validation", "test"),
        )
    ):
        raise ValueError("Physical objects overlap across splits")
    return splits


def prepare_data() -> dict[str, int]:
    root = DATA / "raw" / "equipment"
    manifest_path = root / "manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            "Add collected images and manifest.csv; see docs/vision-data.md"
        )
    manifest = validate_manifest(pd.read_csv(manifest_path), root)
    splits = split_by_object(manifest)
    output = DATA / "processed" / "vision"
    output.mkdir(parents=True, exist_ok=True)
    for name, split in splits.items():
        split.to_csv(output / f"{name}.csv", index=False)
    hashes = [image_hash(root / str(row.image_path)) for row in manifest.itertuples()]
    summary = {
        "source": "locally collected Tech Society pilot images",
        "manifest_sha256": sha256_file(manifest_path),
        "image_hashes_sha256": hashlib.sha256(
            "\n".join(sorted(hashes)).encode()
        ).hexdigest(),
        "seed": SEED,
        "classes": CLASSES,
        "counts": {name: len(split) for name, split in splits.items()},
        "split_sha256": {name: sha256_file(output / f"{name}.csv") for name in splits},
        "object_counts": {
            name: {
                label: int(
                    split.loc[split.label == label, "physical_object_id"].nunique()
                )
                for label in CLASSES
            }
            for name, split in splits.items()
        },
        "split_method": "physical-object-disjoint, class-stratified deterministic shuffle",
    }
    (output / "manifest.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary["counts"]


def _torch_components(model_id: str, pretrained: bool):
    import torch
    from torch import nn
    from torchvision.models import ResNet18_Weights, resnet18

    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    if model_id == "starter":
        for parameter in model.parameters():
            parameter.requires_grad = False
    elif model_id == "ai-lead":
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.layer4.parameters():
            parameter.requires_grad = True
    else:
        raise ValueError("model-id must be starter or ai-lead")
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    return torch, model


def _dataset(rows: pd.DataFrame, image_root: Path, training: bool, model_id: str):
    import torch
    from PIL import Image
    from torch.utils.data import Dataset
    from torchvision import transforms

    base = [transforms.Resize(256), transforms.CenterCrop(224)]
    if training and model_id == "ai-lead":
        transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.15, 0.15, 0.15, 0.05),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
    else:
        transform = transforms.Compose(
            base
            + [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    class EquipmentDataset(Dataset):
        def __len__(self):
            return len(rows)

        def __getitem__(self, index):
            row = rows.iloc[index]
            with Image.open(image_root / row.image_path) as source:
                image = source.convert("RGB")
            return transform(image), CLASSES.index(row.label)

    return EquipmentDataset()


def train(
    model_id: str, epochs: int = 8, batch_size: int = 16, *, replace: bool = False
) -> Path:
    import torch
    from torch.utils.data import DataLoader
    from torchvision.models import ResNet18_Weights

    processed = DATA / "processed" / "vision"
    train_path = processed / "train.csv"
    manifest_path = processed / "manifest.json"
    if not train_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(
            "Run `python -m ml.vision.cli prepare` after collecting the pilot images"
        )
    model_path = DATA.parent / "artifacts" / "vision" / f"{model_id}.pt"
    if model_path.exists() and not replace:
        raise FileExistsError(
            "A fixed artifact already exists; use --replace explicitly to replace it"
        )
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    image_root = DATA / "raw" / "equipment"
    rows = pd.read_csv(train_path)
    dataset = _dataset(rows, image_root, True, model_id)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _torch_components(model_id, pretrained=True)[1].to(device)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=1e-3 if model_id == "starter" else 1e-4,
    )
    loss_function = torch.nn.CrossEntropyLoss()
    model.train()
    if model_id == "starter":
        model.eval()
        model.fc.train()
    for _ in range(epochs):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = loss_function(model(images), labels)
            loss.backward()
            optimizer.step()
    metadata_path = model_path.with_suffix(".json")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {key: value.detach().cpu() for key, value in model.state_dict().items()},
        model_path,
    )
    metadata = {
        "task": "vision",
        "model_id": model_id,
        "classes": CLASSES,
        "architecture": "torchvision ResNet-18",
        "initial_weights": ResNet18_Weights.DEFAULT.name,
        "dataset": "locally collected Tech Society pilot images",
        "dataset_sha256": json.loads(manifest_path.read_text(encoding="utf-8"))[
            "image_hashes_sha256"
        ],
        "split_manifest": str(manifest_path.relative_to(ROOT)),
        "split_manifest_sha256": sha256_file(manifest_path),
        "seed": SEED,
        "config": {
            "epochs": epochs,
            "batch_size": batch_size,
            "training_augmentation": model_id == "ai-lead",
            "frozen_backbone": model_id == "starter",
        },
        "hardware": str(device),
        "runtime": __import__(
            "ml.common.artifacts", fromlist=["runtime_versions"]
        ).runtime_versions(),
        "device": str(device),
        "torch": torch.__version__,
        "torchvision": __import__("torchvision").__version__,
        "trained_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "artifact_sha256": sha256_file(model_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return model_path


def load_model(model_id: str):
    import torch

    path = DATA.parent / "artifacts" / "vision" / f"{model_id}.pt"
    metadata_path = path.with_suffix(".json")
    if not path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError("Model not trained")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("artifact_sha256") != sha256_file(path):
        raise ValueError("Model artifact checksum does not match its metadata")
    model = _torch_components(model_id, pretrained=False)[1]
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model, metadata


def evaluate(model_id: str, split_name: str, *, organiser: bool = False) -> dict:
    if split_name not in {"validation", "test"}:
        raise ValueError("split must be validation or test")
    if split_name == "test" and not organiser:
        raise PermissionError("Final test evaluation requires --organiser")
    import torch
    from torch.utils.data import DataLoader

    model, metadata = load_model(model_id)
    rows = pd.read_csv(DATA / "processed" / "vision" / f"{split_name}.csv")
    loader = DataLoader(
        _dataset(rows, DATA / "raw" / "equipment", False, model_id),
        batch_size=32,
        shuffle=False,
    )
    targets: list[str] = []
    predictions: list[str] = []
    model.eval()
    with torch.inference_mode():
        for images, labels in loader:
            output = model(images)
            predictions.extend(
                CLASSES[index] for index in output.argmax(dim=1).tolist()
            )
            targets.extend(CLASSES[index] for index in labels.tolist())
    result = {
        "task": "vision",
        "model_id": model_id,
        "split": split_name,
        "metrics": classification_metrics(targets, predictions, CLASSES),
    }
    if split_name == "validation":
        result["representative_mistakes"] = [
            {
                "row_index": index,
                "expected": targets[index],
                "predicted": predictions[index],
            }
            for index in range(len(targets))
            if targets[index] != predictions[index]
        ][:5]
        threshold = select_warning_threshold(rows, predictions, model)
        result["low_score_warning_threshold"] = threshold
        metadata["low_score_warning_threshold"] = threshold
        metadata["warning_threshold_target_accuracy"] = WARNING_TARGET_ACCURACY
        metadata["metrics"] = {"validation": result["metrics"]}
        metadata_path = DATA.parent / "artifacts" / "vision" / f"{model_id}.json"
        metadata_path.write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
    output_dir = REPORTS / ("private-test" if split_name == "test" else "validation")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"vision-{model_id}.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def select_warning_threshold(
    rows: pd.DataFrame,
    predictions: list[str],
    model,
    target_accuracy: float = WARNING_TARGET_ACCURACY,
) -> float | None:
    if not 0 <= target_accuracy <= 1:
        raise ValueError("target_accuracy must be between 0 and 1")
    from torch.utils.data import DataLoader
    import torch

    loader = DataLoader(
        _dataset(rows, DATA / "raw" / "equipment", False, "starter"),
        batch_size=32,
        shuffle=False,
    )
    scores: list[float] = []
    model.eval()
    with torch.inference_mode():
        for images, _ in loader:
            scores.extend(model(images).softmax(dim=1).max(dim=1).values.tolist())
    correct = [
        row.label == prediction
        for row, prediction in zip(rows.itertuples(), predictions)
    ]
    candidates = sorted(set(float(score) for score in scores))
    eligible = [
        threshold
        for threshold in candidates
        if sum(ok for ok, score in zip(correct, scores) if score >= threshold)
        / max(1, sum(score >= threshold for score in scores))
        >= target_accuracy
    ]
    return float(max(eligible)) if eligible else None


def export_student_data() -> Path:
    processed = DATA / "processed" / "vision"
    manifest_path = processed / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            "Run `python -m ml.vision.cli prepare` before exporting student data"
        )
    source_root = DATA / "raw" / "equipment"
    output = DATA / "student" / "vision"
    output.mkdir(parents=True, exist_ok=True)
    rows = pd.concat(
        [pd.read_csv(processed / f"{name}.csv") for name in ("train", "validation")],
        ignore_index=True,
    )
    for image_path in rows["image_path"].unique():
        source = source_root / image_path
        destination = output / image_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    rows.to_csv(output / "manifest.csv", index=False)
    summary = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary["counts"] = {
        name: len(pd.read_csv(processed / f"{name}.csv"))
        for name in ("train", "validation")
    }
    summary["object_counts"] = {
        name: summary["object_counts"][name] for name in ("train", "validation")
    }
    if "split_sha256" in summary:
        summary["split_sha256"] = {
            name: summary["split_sha256"][name] for name in ("train", "validation")
        }
    summary["test_split_included"] = False
    (output / "manifest.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return output
