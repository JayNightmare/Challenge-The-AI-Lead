from __future__ import annotations

import hashlib
import re

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def normalized_text_hash(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text).strip().casefold()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def grouped_stratified_splits(
    frame: pd.DataFrame, *, label_column: str, group_column: str, seed: int
) -> dict[str, pd.DataFrame]:
    if frame[group_column].isna().any():
        raise ValueError("Split groups must not be missing")
    unique_groups = frame[group_column].nunique()
    if unique_groups < 10:
        raise ValueError(
            "At least 10 unique groups are required for grouped 10-fold splitting"
        )

    splitter = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=seed)
    folds = list(splitter.split(frame, frame[label_column], groups=frame[group_column]))
    test_indices = folds[0][1]
    validation_indices = folds[1][1]
    test_groups = set(frame.iloc[test_indices][group_column])
    validation_groups = set(frame.iloc[validation_indices][group_column])
    if test_groups & validation_groups:
        raise ValueError("A group crosses validation and test splits")
    train_mask = ~frame[group_column].isin(test_groups | validation_groups)
    result = {
        "train": frame.loc[train_mask].copy(),
        "validation": frame.iloc[validation_indices].copy(),
        "test": frame.iloc[test_indices].copy(),
    }
    verify_group_disjoint(result, group_column)
    return result


def verify_group_disjoint(splits: dict[str, pd.DataFrame], group_column: str) -> None:
    names = list(splits)
    for index, left_name in enumerate(names):
        for right_name in names[index + 1 :]:
            overlap = set(splits[left_name][group_column]) & set(
                splits[right_name][group_column]
            )
            if overlap:
                raise ValueError(f"Groups overlap between {left_name} and {right_name}")


def chronological_splits(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    ordered = frame.sort_values(["dteday", "hr"], kind="stable").reset_index(drop=True)
    count = len(ordered)
    train_end = int(count * 0.70)
    validation_end = int(count * 0.85)
    if train_end == 0 or validation_end <= train_end or validation_end >= count:
        raise ValueError(
            "Not enough rows to create chronological train/validation/test splits"
        )
    splits = {
        "train": ordered.iloc[:train_end].copy(),
        "validation": ordered.iloc[train_end:validation_end].copy(),
        "test": ordered.iloc[validation_end:].copy(),
    }
    if not (
        splits["train"].iloc[-1][["dteday", "hr"]].tolist()
        < splits["validation"].iloc[0][["dteday", "hr"]].tolist()
        and splits["validation"].iloc[-1][["dteday", "hr"]].tolist()
        < splits["test"].iloc[0][["dteday", "hr"]].tolist()
    ):
        raise ValueError("Chronological split ordering failed")
    return splits
