from __future__ import annotations

from io import BytesIO

import pandas as pd

MAX_SUBMISSION_BYTES = 512 * 1024
SUBMISSION_COLUMNS = ["row_id", "prediction"]


def parse_prediction_csv(contents: bytes, expected_row_ids: list[str]) -> pd.DataFrame:
    if not contents or len(contents) > MAX_SUBMISSION_BYTES:
        raise ValueError("CSV must be non-empty and no larger than 512 KiB")
    try:
        frame = pd.read_csv(
            BytesIO(contents),
            dtype=str,
            keep_default_na=False,
            on_bad_lines="error",
        )
    except (UnicodeDecodeError, pd.errors.ParserError, ValueError):
        raise ValueError("Could not parse prediction CSV") from None
    if list(frame.columns) != SUBMISSION_COLUMNS:
        raise ValueError("CSV columns must be exactly row_id,prediction in that order")
    frame["row_id"] = frame["row_id"].str.strip()
    frame["prediction"] = frame["prediction"].str.strip()
    if frame.empty or frame["row_id"].eq("").any() or frame["prediction"].eq("").any():
        raise ValueError("CSV row IDs and predictions must not be empty")
    if frame["row_id"].duplicated().any():
        raise ValueError("CSV row IDs must be unique")
    if len(frame) != len(expected_row_ids):
        raise ValueError("CSV must contain exactly the expected validation row count")
    if set(frame["row_id"]) != set(expected_row_ids):
        raise ValueError("CSV row IDs must exactly match the validation template")
    return frame
