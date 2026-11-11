from __future__ import annotations

import hashlib
from collections import Counter

import pandas as pd


def row_ids_for(task: str, frame: pd.DataFrame, split_name: str) -> list[str]:
    if task == "text":
        if "group_id" not in frame:
            raise ValueError("SMS rows require group_id for stable row IDs")
        occurrences: Counter[str] = Counter()
        values = []
        for group_id in frame["group_id"].astype(str):
            occurrence = occurrences[group_id]
            occurrences[group_id] += 1
            values.append(f"{split_name}:{group_id}:{occurrence}")
        prefix = "sms"
    elif task == "tabular":
        if not {"dteday", "hr"}.issubset(frame.columns):
            raise ValueError("Bike rows require date and hour for stable row IDs")
        values = [
            f"{split_name}:{str(day)[:10]}:{int(hour)}"
            for day, hour in zip(frame["dteday"], frame["hr"])
        ]
        prefix = "bike"
    elif task == "vision":
        if "image_path" not in frame:
            raise ValueError("Vision rows require image_path for stable row IDs")
        values = [f"{split_name}:{path}" for path in frame["image_path"].astype(str)]
        prefix = "vision"
    else:
        raise ValueError(f"Unsupported prediction task: {task}")
    return [
        f"{prefix}-{hashlib.sha256(value.encode('utf-8')).hexdigest()[:20]}"
        for value in values
    ]
