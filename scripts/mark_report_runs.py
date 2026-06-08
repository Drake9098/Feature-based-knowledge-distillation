#!/usr/bin/env python3
"""Mark checkpoint run folders referenced in final_report.md."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHECKPOINTS = ROOT / "experiments" / "checkpoints"

# Runs used in final_report.md (Table 1, cosine scheduler unless noted)
REPORT_RUNS = [
    {
        "method": "Teacher (fine-tuned)",
        "temperature": None,
        "top1_accuracy_percent": 82.16,
        "path": "teacher_finetune_cifar100/2026-05-10_19-43-05",
    },
    {
        "method": "Baseline (CE only)",
        "temperature": None,
        "top1_accuracy_percent": 76.75,
        "path": "phase1_baseline/2026-05-10_20-46-07",
    },
    {
        "method": "Vanilla KD",
        "temperature": 4,
        "top1_accuracy_percent": 80.67,
        "path": "phase2_kd/2026-05-11_10-16-46",
    },
    {
        "method": "Vanilla KD",
        "temperature": 8,
        "top1_accuracy_percent": 80.74,
        "path": "phase2_kd/2026-05-13_16-39-49",
    },
    {
        "method": "Vanilla KD",
        "temperature": 20,
        "top1_accuracy_percent": 80.92,
        "path": "phase2_kd_t20/2026-05-23_12-02-47",
    },
    {
        "method": "FitNet Middle S2",
        "temperature": 3,
        "top1_accuracy_percent": 80.71,
        "path": "fitnet_middle_s2/2026-05-11_13-30-48",
    },
    {
        "method": "FitNet Middle S2",
        "temperature": 8,
        "top1_accuracy_percent": 80.78,
        "path": "fitnet_middle_s2/2026-05-13_17-22-51",
    },
    {
        "method": "FitNet Middle S2",
        "temperature": 20,
        "top1_accuracy_percent": 80.95,
        "path": "fitnet_middle_s2_t20/2026-05-23_12-45-47",
    },
    {
        "method": "FitNet Deep S2",
        "temperature": 3,
        "top1_accuracy_percent": 80.55,
        "path": "fitnet_deep_s2/2026-05-11_17-56-27",
    },
    {
        "method": "FitNet Deep S2",
        "temperature": 8,
        "top1_accuracy_percent": 80.95,
        "path": "fitnet_deep_s2/2026-05-13_19-33-56",
    },
    {
        "method": "FitNet Deep S2",
        "temperature": 20,
        "top1_accuracy_percent": 81.11,
        "path": "fitnet_deep_s2_t20/2026-05-23_13-28-50",
    },
    {
        "method": "FitNet Full S2",
        "temperature": 3,
        "top1_accuracy_percent": 80.51,
        "path": "fitnet_full_s2/2026-05-11_14-13-49",
    },
    {
        "method": "FitNet Full S2",
        "temperature": 8,
        "top1_accuracy_percent": 80.76,
        "path": "fitnet_full_s2/2026-05-13_18-06-52",
    },
    {
        "method": "FitNet Full S2",
        "temperature": 20,
        "top1_accuracy_percent": 80.62,
        "path": "fitnet_full_s2_t20/2026-05-23_14-11-51",
    },
    {
        "method": "AT + KD",
        "temperature": 4,
        "top1_accuracy_percent": 79.52,
        "path": "at_kd/2026-05-11_17-12-54",
    },
    {
        "method": "AT + KD",
        "temperature": 8,
        "top1_accuracy_percent": 79.45,
        "path": "at_kd/2026-05-13_18-49-53",
    },
    {
        "method": "AT + KD",
        "temperature": 20,
        "top1_accuracy_percent": 79.62,
        "path": "at_kd_t20/2026-05-23_14-54-54",
    },
]

MARKER_NAME = "report_source.json"


def main() -> None:
    manifest = {
        "report_file": "final_report.md",
        "section": "5.1 Main results (Table 1)",
        "scheduler": "cosine",
        "teacher_checkpoint": "teacher_finetune_cifar100/2026-05-10_19-43-05",
        "runs": [],
    }

    for entry in REPORT_RUNS:
        run_dir = CHECKPOINTS / entry["path"]
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Run folder not found: {run_dir}")

        marker = {
            "used_in_report": "final_report.md",
            "table": "Table 1",
            "method": entry["method"],
            "temperature": entry["temperature"],
            "top1_accuracy_percent": entry["top1_accuracy_percent"],
            "run_path": f"experiments/checkpoints/{entry['path']}",
        }
        marker_path = run_dir / MARKER_NAME
        marker_path.write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")

        manifest["runs"].append({**entry, "marker_file": MARKER_NAME})

    manifest_path = ROOT / "experiments" / "REPORT_SOURCE_RUNS.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote manifest: {manifest_path}")
    print(f"Marked {len(REPORT_RUNS)} run folders with {MARKER_NAME}")


if __name__ == "__main__":
    main()
