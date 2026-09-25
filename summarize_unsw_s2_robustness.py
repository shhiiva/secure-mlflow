#!/usr/bin/env python3
"""Validate and summarize the completed UNSW-NB15 S2 robustness experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


VARIANT_RE = re.compile(
    r"^s2_(?P<direction>0to1|1to0)_(?P<trigger_feature>.+)_"
    r"(?P<trigger_strength>weak|medium|strong)$"
)
METRICS = [
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc",
    "backdoor_success_rate",
    "neural_cleanse_score",
    "strip_score",
    "activation_clustering_score",
    "fusion_score",
    "execution_seconds",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bootstrap_ci(values: pd.Series, rng: np.random.Generator, resamples: int) -> tuple[float, float]:
    array = values.dropna().to_numpy(dtype=float)
    if not len(array):
        return float("nan"), float("nan")
    samples = rng.choice(array, size=(resamples, len(array)), replace=True).mean(axis=1)
    low, high = np.percentile(samples, [2.5, 97.5])
    return float(low), float(high)


def parse_variant(value: str) -> dict[str, object]:
    match = VARIANT_RE.fullmatch(value)
    if not match:
        raise ValueError(f"Unrecognised variant_id: {value}")
    direction = match.group("direction")
    return {
        "direction": direction.replace("to", "->"),
        "source_label": int(direction[0]),
        "target_label": int(direction[-1]),
        "trigger_feature": match.group("trigger_feature"),
        "trigger_strength": match.group("trigger_strength"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20_260_910)
    args = parser.parse_args()

    root = args.input_dir.resolve()
    runs_path = root / "unsw_s2_robustness_runs.csv"
    manifest_path = root / "unsw_s2_robustness_manifest.json"
    if not runs_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("The combined runs CSV or robustness manifest is missing.")

    runs = pd.read_csv(runs_path)
    parsed = pd.DataFrame([parse_variant(str(value)) for value in runs["variant_id"]])
    runs = pd.concat([runs, parsed], axis=1)
    for metric in METRICS + ["poison_rate", "repetition", "seed"]:
        runs[metric] = pd.to_numeric(runs[metric], errors="coerce")
    runs["fusion_detected"] = runs["fusion_decision"].eq("MALICIOUS").astype(int)

    cell_counts = runs.groupby(["variant_id", "poison_rate"]).size()
    errors: list[str] = []
    if len(runs) != 180:
        errors.append(f"Expected 180 runs; found {len(runs)}.")
    if runs["variant_id"].nunique() != 12:
        errors.append(f"Expected 12 variants; found {runs['variant_id'].nunique()}.")
    if len(cell_counts) != 36 or not cell_counts.eq(5).all():
        errors.append("Expected 36 variant/rate cells with exactly five repetitions each.")
    if runs["seed"].isna().any():
        errors.append("One or more seeds are missing or non-numeric.")
    if runs.duplicated(["variant_id", "poison_rate", "repetition"]).any():
        errors.append("Duplicate variant/rate/repetition records were found.")

    group_columns = [
        "variant_id",
        "direction",
        "source_label",
        "target_label",
        "trigger_feature",
        "trigger_strength",
        "poison_rate",
    ]
    rng = np.random.default_rng(args.seed)
    summary_rows: list[dict[str, object]] = []
    for keys, group in runs.groupby(group_columns, sort=True, dropna=False):
        row = dict(zip(group_columns, keys))
        row["n"] = len(group)
        for metric in METRICS + ["fusion_detected"]:
            values = group[metric]
            low, high = bootstrap_ci(values, rng, args.bootstrap_resamples)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_sd"] = float(values.std(ddof=1))
            row[f"{metric}_ci95_low"] = low
            row[f"{metric}_ci95_high"] = high
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary_path = root / "unsw_s2_robustness_summary.csv"
    summary.to_csv(summary_path, index=False)

    detector_rows: list[dict[str, object]] = []
    for variant_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        match = VARIANT_RE.fullmatch(variant_dir.name)
        performance_path = variant_dir / "unsw_detector_performance.csv"
        if not match or not performance_path.is_file():
            continue
        metadata = parse_variant(variant_dir.name)
        performance = pd.read_csv(performance_path)
        for record in performance.to_dict(orient="records"):
            detector_rows.append({"variant_id": variant_dir.name, **metadata, **record})
    detector_path = root / "unsw_s2_detector_performance_by_variant.csv"
    pd.DataFrame(detector_rows).to_csv(detector_path, index=False)

    overall = {
        "status": "PASS" if not errors else "FAIL",
        "validation_errors": errors,
        "design": {
            "attack_runs": int(len(runs)),
            "variants": int(runs["variant_id"].nunique()),
            "poison_rates": sorted(runs["poison_rate"].unique().tolist()),
            "repetitions_per_variant_rate": sorted(cell_counts.unique().tolist()),
            "directions": sorted(runs["direction"].unique().tolist()),
            "trigger_features": sorted(runs["trigger_feature"].unique().tolist()),
            "trigger_strengths": sorted(runs["trigger_strength"].unique().tolist()),
        },
        "overall_means": {
            metric: float(runs[metric].mean())
            for metric in METRICS + ["fusion_detected"]
        },
        "interpretation_note": (
            "Confidence intervals describe variation across five deterministic repetitions "
            "within each experimental cell; they do not establish population-wide performance."
        ),
        "inputs": {
            str(runs_path): sha256(runs_path),
            str(manifest_path): sha256(manifest_path),
        },
        "outputs": {
            str(summary_path): sha256(summary_path),
            str(detector_path): sha256(detector_path),
        },
    }
    overall_path = root / "unsw_s2_robustness_analysis.json"
    overall_path.write_text(json.dumps(overall, indent=2), encoding="utf-8")

    print(f"Validation: {overall['status']}")
    print(f"Runs: {len(runs)}; variants: {runs['variant_id'].nunique()}; cells: {len(cell_counts)}")
    print(f"Summary: {summary_path}")
    print(f"Detector table: {detector_path}")
    print(f"Analysis manifest: {overall_path}")
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
