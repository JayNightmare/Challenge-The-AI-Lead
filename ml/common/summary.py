from __future__ import annotations

import json

from ml.common.artifacts import REPORTS, ROOT


def generate_frontend_metrics():
    result = {}
    for task in ("vision", "text", "tabular"):
        models = {}
        for model_id in ("starter", "ai-lead"):
            report_path = REPORTS / "validation" / f"{task}-{model_id}.json"
            if not report_path.is_file():
                continue
            try:
                report = json.loads(report_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if report.get("split") == "validation":
                models[model_id] = report
        result[task] = {
            "status": "evaluated" if models else "awaiting_data",
            "models": models,
        }
    path = ROOT / "frontend" / "public" / "data" / "metrics.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return path
