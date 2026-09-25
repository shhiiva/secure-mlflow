#!/usr/bin/env python3
"""Leakage-safe UNSW-NB15 S1/S2 secondary poisoning validation.

The official training partition is split into model-fit and detector-calibration
subsets. The official test partition is never used for fitting or threshold
selection. Five clean models calibrate detector thresholds. Each attack record
is written immediately so an interrupted experiment can be resumed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


SCHEMA_VERSION = "1.0"
RATES = (0.01, 0.05, 0.10)
DROP_COLUMNS = {"id", "attack_cat", "attack category", "attack_category"}


def json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialise {type(value)!r}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=json_safe)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def select_poison_indices(
    labels: np.ndarray,
    rate: float,
    seed: int,
    source_label: int | None = None,
) -> np.ndarray:
    labels = np.asarray(labels, dtype=int)
    requested = int(round(len(labels) * rate))
    eligible = (
        np.arange(len(labels))
        if source_label is None
        else np.flatnonzero(labels == source_label)
    )
    if requested <= 0 or requested > len(eligible):
        raise ValueError(
            f"Poison rate {rate} requests {requested} rows from {len(eligible)} eligible rows."
        )
    return np.random.default_rng(seed).choice(eligible, requested, replace=False)


def validate_resume_configuration(path: Path, configuration: dict[str, Any], resume: bool) -> None:
    if resume and path.is_file():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous.get("configuration_sha256") != configuration["configuration_sha256"]:
            raise RuntimeError(
                "Resume refused: stored configuration does not match this command. "
                "Use a new --output-dir for a different experiment."
            )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=json_safe), encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for row in rows:
        for name in row:
            if name not in fieldnames:
                fieldnames.append(name)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def label_column(frame: pd.DataFrame) -> str:
    for name in ("label", "Label"):
        if name in frame.columns:
            return name
    raise ValueError("UNSW-NB15 input must contain a binary label column.")


def binary_labels(series: pd.Series) -> np.ndarray:
    if pd.api.types.is_numeric_dtype(series):
        values = pd.to_numeric(series, errors="raise").astype(int).to_numpy()
    else:
        mapping = {
            "normal": 0,
            "benign": 0,
            "0": 0,
            "attack": 1,
            "malicious": 1,
            "1": 1,
        }
        lowered = series.astype(str).str.strip().str.lower()
        unknown = sorted(set(lowered) - set(mapping))
        if unknown:
            raise ValueError(f"Unsupported labels: {unknown[:10]}")
        values = lowered.map(mapping).astype(int).to_numpy()
    if set(np.unique(values)) != {0, 1}:
        raise ValueError(f"Expected both binary classes; found {sorted(np.unique(values))}")
    return values


def stratified_cap(X: pd.DataFrame, y: np.ndarray, maximum: int, seed: int):
    if maximum <= 0 or len(X) <= maximum:
        return X.reset_index(drop=True), np.asarray(y, dtype=int)
    selected, _ = train_test_split(
        np.arange(len(X)), train_size=maximum, stratify=y, random_state=seed
    )
    selected = np.sort(selected)
    return X.iloc[selected].reset_index(drop=True), np.asarray(y, dtype=int)[selected]


def preprocessor_for(X: pd.DataFrame) -> ColumnTransformer:
    numeric = [name for name in X if pd.api.types.is_numeric_dtype(X[name])]
    categorical = [name for name in X if name not in numeric]
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
    return ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("encode", encoder),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
    )


def new_model(seed: int, max_iter: int) -> MLPClassifier:
    return MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        learning_rate_init=0.001,
        max_iter=max_iter,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=8,
        random_state=seed,
    )


def classification_metrics(model: MLPClassifier, X: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    prediction = model.predict(X)
    probability = model.predict_proba(X)[:, int(np.flatnonzero(model.classes_ == 1)[0])]
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "precision": float(precision_score(y, prediction, zero_division=0)),
        "recall": float(recall_score(y, prediction, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "f1": float(f1_score(y, prediction, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probability)),
        "pr_auc": float(average_precision_score(y, probability)),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def final_hidden_activations(model: MLPClassifier, X: np.ndarray) -> np.ndarray:
    activations = np.asarray(X, dtype=float)
    for weights, intercept in zip(model.coefs_[:-1], model.intercepts_[:-1]):
        activations = np.maximum(activations @ weights + intercept, 0.0)
    return activations


def activation_clustering(model: MLPClassifier, X: np.ndarray, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    if len(X) > 5000:
        selected = rng.choice(len(X), 5000, replace=False)
        X = X[selected]
    activations = final_hidden_activations(model, X)
    clusters = KMeans(n_clusters=2, n_init=10, random_state=seed).fit_predict(activations)
    counts = np.bincount(clusters, minlength=2)
    imbalance = float(abs(counts[0] - counts[1]) / len(clusters))
    # Sample within each cluster so silhouette scoring cannot accidentally
    # receive a one-cluster subset. Degenerate KMeans output is recorded
    # explicitly instead of terminating a long experiment.
    selected_parts = []
    for cluster_id in (0, 1):
        members = np.flatnonzero(clusters == cluster_id)
        take = min(len(members), 1000)
        if take:
            selected_parts.append(rng.choice(members, take, replace=False))
    selected = np.concatenate(selected_parts) if selected_parts else np.asarray([], dtype=int)
    selected_labels = clusters[selected] if len(selected) else np.asarray([], dtype=int)
    if len(selected) >= 3 and len(np.unique(selected_labels)) == 2:
        try:
            silhouette = float(silhouette_score(activations[selected], selected_labels))
            silhouette_status = "EXECUTED"
        except ValueError as error:
            silhouette = 0.0
            silhouette_status = f"UNAVAILABLE: {error}"
    else:
        silhouette = 0.0
        silhouette_status = "DEGENERATE_SINGLE_CLUSTER"
    score = (
        0.0
        if len(np.unique(clusters)) < 2
        else float(np.clip(0.55 * max(0.0, silhouette) + 0.45 * imbalance, 0.0, 1.0))
    )
    return {
        "method": "final_hidden_layer_kmeans",
        "score": score,
        "silhouette": silhouette,
        "silhouette_status": silhouette_status,
        "cluster_counts": [int(value) for value in counts],
        "sample_size": int(len(activations)),
    }


class ArtMlpAdapter:
    def __init__(self, model: MLPClassifier):
        self.model = model
        self.nb_classes = len(model.classes_)
        self.layer_names = [f"hidden_{i + 1}" for i in range(len(model.coefs_) - 1)]

    def predict(self, x, batch_size=128, training_mode=False):
        return self.model.predict(x)

    def predict_proba(self, x):
        return self.model.predict_proba(x)

    def get_activations(self, x, layer=0, batch_size=128, framework=False):
        if isinstance(layer, str):
            layer = self.layer_names.index(layer)
        activations = np.asarray(x, dtype=float)
        for index, (weights, intercept) in enumerate(
            zip(self.model.coefs_[:-1], self.model.intercepts_[:-1])
        ):
            activations = np.maximum(activations @ weights + intercept, 0.0)
            if index == int(layer):
                return activations
        raise ValueError(f"Invalid activation layer {layer}")


def art_activation_clustering(
    model: MLPClassifier, X: np.ndarray, y: np.ndarray, seed: int
) -> dict[str, Any]:
    try:
        from art.defences.detector.poison.activation_defence import ActivationDefence
        try:
            from importlib.metadata import version
            art_version = version("adversarial-robustness-toolbox")
        except Exception:
            art_version = "unknown"
    except ImportError:
        return {"available": False, "status": "NOT_INSTALLED"}
    try:
        rng = np.random.default_rng(seed)
        if len(X) > 5000:
            selected = rng.choice(len(X), 5000, replace=False)
            X, y = X[selected], y[selected]
        defence = ActivationDefence(ArtMlpAdapter(model), X.astype(np.float32), y.astype(int))
        report, flags = defence.detect_poison(
            nb_clusters=2,
            nb_dims=min(10, max(2, X.shape[1] - 1)),
            reduce="PCA",
            cluster_analysis="smaller",
        )
        flags = np.asarray(flags, dtype=int)
        return {
            "available": True,
            "status": "EXECUTED",
            "art_version": art_version,
            "poison_fraction": float(np.mean(flags == 0)),
            "sample_size": int(len(flags)),
            "report": report,
        }
    except Exception as error:
        return {
            "available": True,
            "status": "FAILED",
            "art_version": art_version,
            "error": str(error),
        }


def neural_cleanse_approximation(
    model: MLPClassifier,
    source_samples: np.ndarray,
    training_reference: np.ndarray,
    target: int = 1,
) -> dict[str, Any]:
    """Deterministic one-feature reverse-trigger search for sklearn MLPs."""
    started = time.perf_counter()
    if len(source_samples) > 200:
        selected = np.linspace(0, len(source_samples) - 1, 200, dtype=int)
        source_samples = source_samples[selected]
    target_position = int(np.flatnonzero(model.classes_ == target)[0])
    baseline = float(model.predict_proba(source_samples)[:, target_position].mean())
    variances = np.var(training_reference, axis=0)
    feature_indices = np.argsort(variances)[-min(30, training_reference.shape[1]) :]
    best = None
    candidates_evaluated = 0
    improvement_trajectory = []
    for feature in feature_indices:
        median = float(np.median(training_reference[:, feature]))
        scale = float(np.median(np.abs(training_reference[:, feature] - median)) * 1.4826 + 1e-6)
        for percentile in (1, 5, 50, 95, 99):
            value = float(np.percentile(training_reference[:, feature], percentile))
            modified = source_samples.copy()
            modified[:, feature] = value
            mean_probability = float(model.predict_proba(modified)[:, target_position].mean())
            norm = abs(value - median) / scale
            objective = max(0.0, mean_probability - baseline) / (1.0 + norm)
            candidate = (objective, int(feature), value, mean_probability, norm)
            candidates_evaluated += 1
            if best is None or candidate[0] > best[0]:
                best = candidate
                improvement_trajectory.append(
                    {
                        "candidate": candidates_evaluated,
                        "encoded_feature": int(feature),
                        "value": value,
                        "objective": objective,
                        "target_probability": mean_probability,
                    }
                )
    return {
        "method": "deterministic_one_feature_percentile_search_approximation",
        "score": float(best[0]),
        "best_encoded_feature": int(best[1]),
        "best_value": float(best[2]),
        "baseline_target_probability": baseline,
        "modified_target_probability": float(best[3]),
        "perturbation_norm": float(best[4]),
        "candidates_evaluated": candidates_evaluated,
        "improvement_trajectory": improvement_trajectory,
        "runtime_seconds": float(time.perf_counter() - started),
        "claim_boundary": "This is a sklearn-compatible approximation, not gradient-based Neural Cleanse.",
    }


def strip_score(
    model: MLPClassifier,
    candidates: np.ndarray,
    anchors: np.ndarray,
    seed: int,
    perturbations: int = 20,
    mix: float = 0.10,
) -> dict[str, Any]:
    """Tabular STRIP diagnostic using deterministic feature mixing."""
    rng = np.random.default_rng(seed)
    if len(candidates) > 200:
        candidates = candidates[np.linspace(0, len(candidates) - 1, 200, dtype=int)]
    started = time.perf_counter()
    entropies = []
    for candidate in candidates:
        selected = rng.choice(len(anchors), perturbations, replace=len(anchors) < perturbations)
        mixed = (1.0 - mix) * candidate + mix * anchors[selected]
        probabilities = np.clip(model.predict_proba(mixed), 1e-12, 1.0)
        entropy = -(probabilities * np.log(probabilities)).sum(axis=1)
        entropies.append(float(np.mean(entropy / math.log(probabilities.shape[1]))))
    entropy_array = np.asarray(entropies, dtype=float)
    return {
        "method": "deterministic_tabular_feature_mixing_diagnostic",
        "score": float(1.0 - entropy_array.mean()),
        "mean_normalized_entropy": float(entropy_array.mean()),
        "std_normalized_entropy": float(entropy_array.std(ddof=0)),
        "candidate_count": int(len(candidates)),
        "perturbations_per_candidate": int(perturbations),
        "mixing_ratio": float(mix),
        "per_input_normalized_entropy": entropy_array.tolist(),
        "runtime_seconds": float(time.perf_counter() - started),
        "claim_boundary": "This is a tabular adaptation of STRIP.",
    }


def empirical_threshold(values: list[float]) -> float:
    maximum = max(values)
    return float(maximum + max(1e-6, abs(maximum) * 0.01))


def fusion(scores: dict[str, float], thresholds: dict[str, float]) -> dict[str, Any]:
    weights = {"neural_cleanse": 0.40, "strip": 0.30, "activation_clustering": 0.30}
    ratios = {
        name: float(min(2.0, scores[name] / max(thresholds[name], 1e-12)))
        for name in weights
    }
    score = float(sum(weights[name] * ratios[name] for name in weights))
    return {
        "score": score,
        "threshold": 1.0,
        "decision": "MALICIOUS" if score >= 1.0 else "CLEAN",
        "weights": weights,
        "threshold_normalized_component_ratios": ratios,
    }


def ablation_results(scores: dict[str, float], thresholds: dict[str, float]) -> dict[str, Any]:
    ratios = {name: threshold_ratio(scores[name], thresholds[name]) for name in scores}
    definitions = {
        "neural_cleanse_only": ratios["neural_cleanse"],
        "strip_only": ratios["strip"],
        "activation_clustering_only": ratios["activation_clustering"],
        "neural_cleanse_plus_activation": (
            ratios["neural_cleanse"] + ratios["activation_clustering"]
        ) / 2.0,
        "all_three_equal_weight": statistics.fmean(ratios.values()),
    }
    return {
        name: {
            "score": float(score),
            "threshold": 1.0,
            "decision": "MALICIOUS" if score >= 1.0 else "CLEAN",
        }
        for name, score in definitions.items()
    }


def trigger_copy(frame: pd.DataFrame, feature: str, value: float) -> pd.DataFrame:
    output = frame.copy()
    output[feature] = pd.to_numeric(output[feature], errors="raise").astype(float)
    output.loc[:, feature] = value
    return output


def flatten_record(record: dict[str, Any]) -> dict[str, Any]:
    metrics = record["metrics"]
    detectors = record.get("detectors", {})
    return {
        "scenario": record["scenario"],
        "poison_rate": record["poison_rate"],
        "repetition": record["repetition"],
        "seed": record["seed"],
        "poisoned_rows": record["poisoned_rows"],
        "accuracy": metrics["accuracy"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "roc_auc": metrics["roc_auc"],
        "pr_auc": metrics["pr_auc"],
        "backdoor_success_rate": record.get("backdoor_success_rate", ""),
        "s1_observed_flip_fraction": detectors.get("label_integrity", {}).get("score", ""),
        "neural_cleanse_score": detectors.get("neural_cleanse", {}).get("score", ""),
        "strip_score": detectors.get("strip", {}).get("score", ""),
        "activation_clustering_score": detectors.get("activation_clustering", {}).get("score", ""),
        "art_activation_status": detectors.get("art_activation_clustering", {}).get("status", ""),
        "fusion_score": detectors.get("fusion", {}).get("score", ""),
        "fusion_decision": detectors.get("fusion", {}).get("decision", ""),
        "execution_seconds": record["execution_seconds"],
        "evidence_file": record["evidence_file"],
    }


def bootstrap_mean_ci(values: list[float], seed: int, resamples: int = 20000):
    rng = np.random.default_rng(seed)
    array = np.asarray(values, dtype=float)
    means = np.empty(resamples, dtype=float)
    for index in range(resamples):
        means[index] = float(np.mean(rng.choice(array, size=len(array), replace=True)))
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def grouped_summary(rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    groups: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["scenario"], float(row["poison_rate"]))].append(row)
    output = []
    metrics = (
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
    )
    for (scenario, rate), group in sorted(groups.items()):
        summary: dict[str, Any] = {"scenario": scenario, "poison_rate": rate, "n": len(group)}
        for metric in metrics:
            values = [float(row[metric]) for row in group if row.get(metric) not in (None, "")]
            if values:
                summary[f"{metric}_mean"] = statistics.fmean(values)
                summary[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
                label_seed = int(
                    hashlib.sha256(f"{scenario}:{rate}:{metric}".encode()).hexdigest()[:8], 16
                )
                low, high = bootstrap_mean_ci(values, seed + label_seed)
                summary[f"{metric}_ci95_lower"] = low
                summary[f"{metric}_ci95_upper"] = high
        output.append(summary)
    return output


def threshold_ratio(score: float, threshold: float) -> float:
    return float(min(2.0, score / max(threshold, 1e-12)))


def write_curves_and_figures(
    output: Path,
    clean_controls: list[dict[str, Any]],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    from sklearn.metrics import precision_recall_curve, roc_curve

    curve_sets = {}
    s1 = [record for record in records if record["scenario"] == "S1_LABEL_FLIP"]
    if s1:
        curve_sets["s1"] = (
            np.asarray([0] * len(clean_controls) + [1] * len(s1), dtype=int),
            np.asarray(
                [0.0] * len(clean_controls)
                + [record["detectors"]["label_integrity"]["score"] for record in s1],
                dtype=float,
            ),
        )
    s2 = [record for record in records if record["scenario"] == "S2_BACKDOOR"]
    curve_sets["s2"] = (
        np.asarray([0] * len(clean_controls) + [1] * len(s2), dtype=int),
        np.asarray(
            [control["fusion"]["score"] for control in clean_controls]
            + [record["detectors"]["fusion"]["score"] for record in s2],
            dtype=float,
        ),
    )
    figure_data = {}
    for name, (labels, scores) in curve_sets.items():
        precision, recall, pr_thresholds = precision_recall_curve(labels, scores)
        fpr, tpr, roc_thresholds = roc_curve(labels, scores)
        write_csv(
            output / f"unsw_{name}_pr_curve.csv",
            [
                {
                    "precision": precision[index],
                    "recall": recall[index],
                    "threshold": pr_thresholds[index] if index < len(pr_thresholds) else "",
                }
                for index in range(len(precision))
            ],
        )
        write_csv(
            output / f"unsw_{name}_roc_curve.csv",
            [
                {"fpr": fpr[index], "tpr": tpr[index], "threshold": roc_thresholds[index]}
                for index in range(len(fpr))
            ],
        )
        figure_data[name] = {
            "precision": precision,
            "recall": recall,
            "fpr": fpr,
            "tpr": tpr,
            "pr_auc": float(average_precision_score(labels, scores)),
            "roc_auc": float(roc_auc_score(labels, scores)),
        }
    try:
        import matplotlib.pyplot as plt

        figure, axes = plt.subplots(1, 2, figsize=(10, 4.5))
        for name, values in figure_data.items():
            axes[0].plot(values["recall"], values["precision"], label=f"{name.upper()} AP={values['pr_auc']:.3f}")
            axes[1].plot(values["fpr"], values["tpr"], label=f"{name.upper()} AUC={values['roc_auc']:.3f}")
        axes[0].set(xlabel="Recall", ylabel="Precision", title="UNSW-NB15 precision-recall")
        axes[1].set(xlabel="False-positive rate", ylabel="True-positive rate", title="UNSW-NB15 ROC")
        axes[1].plot([0, 1], [0, 1], "--", color="grey", linewidth=1)
        for axis in axes:
            axis.grid(alpha=0.25)
            axis.legend()
        figure.tight_layout()
        figure.savefig(output / "unsw_pr_roc_curves.png", dpi=300)
        plt.close(figure)
        pr_figure, pr_axis = plt.subplots(figsize=(6, 5))
        roc_figure, roc_axis = plt.subplots(figsize=(6, 5))
        for name, values in figure_data.items():
            pr_axis.plot(values["recall"], values["precision"], label=f"{name.upper()} AP={values['pr_auc']:.3f}")
            roc_axis.plot(values["fpr"], values["tpr"], label=f"{name.upper()} AUC={values['roc_auc']:.3f}")
        pr_axis.set(xlabel="Recall", ylabel="Precision", title="UNSW-NB15 precision-recall")
        roc_axis.set(xlabel="False-positive rate", ylabel="True-positive rate", title="UNSW-NB15 ROC")
        roc_axis.plot([0, 1], [0, 1], "--", color="grey", linewidth=1)
        for axis in (pr_axis, roc_axis):
            axis.grid(alpha=0.25)
            axis.legend()
        pr_figure.tight_layout()
        roc_figure.tight_layout()
        pr_figure.savefig(output / "unsw_pr_curves.png", dpi=300)
        roc_figure.savefig(output / "unsw_roc_curves.png", dpi=300)
        plt.close(pr_figure)
        plt.close(roc_figure)
        plot_status = "SAVED"
    except Exception as error:
        plot_status = f"NOT_SAVED: {error}"
    return {
        name: {"pr_auc": values["pr_auc"], "roc_auc": values["roc_auc"]}
        for name, values in figure_data.items()
    } | {"plot_status": plot_status}


def binary_detection_metrics(labels: list[int], predictions: list[int]) -> dict[str, Any]:
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
    }


def run(args: argparse.Namespace) -> None:
    output = Path(args.output_dir).resolve()
    runs_dir = output / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    train_path, test_path = Path(args.train).resolve(), Path(args.test).resolve()
    train_frame, test_frame = pd.read_csv(train_path), pd.read_csv(test_path)
    train_label, test_label = label_column(train_frame), label_column(test_frame)
    y_all, y_test = binary_labels(train_frame[train_label]), binary_labels(test_frame[test_label])
    drop_train = [c for c in train_frame if c.strip().lower() in DROP_COLUMNS or c == train_label]
    drop_test = [c for c in test_frame if c.strip().lower() in DROP_COLUMNS or c == test_label]
    X_all = train_frame.drop(columns=drop_train)
    X_test_raw = test_frame.drop(columns=drop_test)
    if list(X_all.columns) != list(X_test_raw.columns):
        raise ValueError("Official train/test feature schemas differ.")
    X_all, y_all = stratified_cap(X_all, y_all, args.max_train_rows, args.seed)
    X_fit_raw, X_cal_raw, y_fit, y_cal = train_test_split(
        X_all,
        y_all,
        test_size=args.calibration_fraction,
        stratify=y_all,
        random_state=args.seed,
    )
    X_fit_raw, X_cal_raw = X_fit_raw.reset_index(drop=True), X_cal_raw.reset_index(drop=True)

    preprocessor = preprocessor_for(X_fit_raw)
    X_fit = np.asarray(preprocessor.fit_transform(X_fit_raw), dtype=np.float32)
    X_cal = np.asarray(preprocessor.transform(X_cal_raw), dtype=np.float32)
    X_test = np.asarray(preprocessor.transform(X_test_raw), dtype=np.float32)

    numeric = [c for c in X_fit_raw if pd.api.types.is_numeric_dtype(X_fit_raw[c])]
    if args.trigger_feature:
        if args.trigger_feature not in numeric:
            raise ValueError("--trigger-feature must name a numeric UNSW feature.")
        trigger_feature = args.trigger_feature
    elif "sbytes" in numeric:
        trigger_feature = "sbytes"
    else:
        trigger_feature = sorted(numeric)[0]
    clean_feature = pd.to_numeric(X_fit_raw[trigger_feature], errors="coerce")
    q1, q3 = clean_feature.quantile([0.25, 0.75])
    if args.trigger_strength == "weak":
        trigger_value = float(clean_feature.quantile(0.95))
        trigger_definition = "training-only 95th percentile"
    elif args.trigger_strength == "medium":
        trigger_value = float(clean_feature.quantile(0.99))
        trigger_definition = "training-only 99th percentile"
    else:
        trigger_value = float(clean_feature.quantile(0.99) + args.trigger_iqr_multiplier * (q3 - q1))
        trigger_definition = "training-only 99th percentile plus fixed IQR multiplier"

    source_cal_raw = X_cal_raw.iloc[np.flatnonzero(y_cal == args.s2_source_label)].reset_index(drop=True)
    source_test_raw = X_test_raw.iloc[np.flatnonzero(y_test == args.s2_source_label)].reset_index(drop=True)
    triggered_cal = np.asarray(preprocessor.transform(trigger_copy(source_cal_raw, trigger_feature, trigger_value)), dtype=np.float32)
    triggered_test = np.asarray(preprocessor.transform(trigger_copy(source_test_raw, trigger_feature, trigger_value)), dtype=np.float32)
    source_cal = np.asarray(preprocessor.transform(source_cal_raw), dtype=np.float32)

    configuration = {
        "schema_version": SCHEMA_VERSION,
        "dataset": "UNSW-NB15",
        "role": "secondary_poisoning_validation",
        "official_train": str(train_path),
        "official_test": str(test_path),
        "official_train_sha256": sha256_file(train_path),
        "official_test_sha256": sha256_file(test_path),
        "implementation_file": str(Path(__file__).resolve()),
        "implementation_sha256": sha256_file(Path(__file__).resolve()),
        "split_protocol": "official test held out; official training split into model-fit and detector-calibration",
        "holdout_used_for_training_or_threshold_selection": False,
        "fit_rows": int(len(X_fit)),
        "calibration_rows": int(len(X_cal)),
        "test_rows": int(len(X_test)),
        "encoded_features": int(X_fit.shape[1]),
        "rates": list(RATES),
        "repetitions": args.repetitions,
        "seed": args.seed,
        "trigger_feature": trigger_feature,
        "trigger_value": trigger_value,
        "trigger_strength": args.trigger_strength,
        "trigger_definition": trigger_definition,
        "s2_source_label": args.s2_source_label,
        "s2_target_label": args.s2_target_label,
        "model_configuration": {
            "type": "sklearn.neural_network.MLPClassifier",
            "hidden_layer_sizes": [64, 32],
            "max_iter": args.max_iter,
            "early_stopping": True,
        },
        "neural_cleanse_scope": "deterministic sklearn-compatible approximation",
        "strip_scope": "tabular diagnostic adaptation",
    }
    configuration["configuration_sha256"] = canonical_sha256(configuration)
    configuration_path = output / "configuration.json"
    validate_resume_configuration(configuration_path, configuration, args.resume)
    write_json(configuration_path, configuration)

    clean_controls = []
    control_component_values = defaultdict(list)
    for repetition in range(1, args.repetitions + 1):
        seed = args.seed + repetition
        start = time.perf_counter()
        model = new_model(seed, args.max_iter).fit(X_fit, y_fit)
        nc = neural_cleanse_approximation(model, source_cal, X_fit)
        strip = strip_score(model, triggered_cal, X_cal, seed)
        ac = activation_clustering(model, X_fit, seed)
        control = {
            "configuration_sha256": configuration["configuration_sha256"],
            "repetition": repetition,
            "seed": seed,
            "metrics": classification_metrics(model, X_test, y_test),
            "neural_cleanse": nc,
            "strip": strip,
            "activation_clustering": ac,
            "execution_seconds": time.perf_counter() - start,
        }
        clean_controls.append(control)
        control_component_values["neural_cleanse"].append(nc["score"])
        control_component_values["strip"].append(strip["score"])
        control_component_values["activation_clustering"].append(ac["score"])
        write_json(output / "clean_controls" / f"clean_control_run{repetition:02d}.json", control)

    thresholds = {name: empirical_threshold(values) for name, values in control_component_values.items()}
    thresholds["fusion"] = 1.0
    for control in clean_controls:
        component_scores = {
            name: float(control[name]["score"])
            for name in ("neural_cleanse", "strip", "activation_clustering")
        }
        for name, score in component_scores.items():
            control[name]["threshold"] = thresholds[name]
            control[name]["decision"] = "MALICIOUS" if score >= thresholds[name] else "CLEAN"
        control["fusion"] = fusion(component_scores, thresholds)
        control["ablations"] = ablation_results(component_scores, thresholds)
        control["false_response"] = bool(
            control["fusion"]["decision"] == "MALICIOUS"
        )
        write_json(
            output / "clean_controls" / f"clean_control_run{int(control['repetition']):02d}.json",
            control,
        )
    write_json(
        output / "detector_thresholds.json",
        {
            "selection_population": "five clean controls using official-training calibration data only",
            "selection_rule": "maximum clean-control score plus max(1e-6, 1% margin)",
            "thresholds": thresholds,
            "clean_control_scores": dict(control_component_values),
            "configuration_sha256": configuration["configuration_sha256"],
            "calibration_rows": len(X_cal),
            "seed": args.seed,
        },
    )

    records = []
    scenarios = ("S2_BACKDOOR",) if args.skip_s1 else ("S1_LABEL_FLIP", "S2_BACKDOOR")
    for scenario in scenarios:
        for rate in RATES:
            for repetition in range(1, args.repetitions + 1):
                seed = args.seed + (10000 if scenario.startswith("S1") else 20000) + int(rate * 1000) + repetition
                token = f"{int(round(rate * 100)):02d}"
                direction = f"{args.s2_source_label}to{args.s2_target_label}"
                suffix = (
                    f"_{direction}_{trigger_feature}_{args.trigger_strength}"
                    if scenario == "S2_BACKDOOR"
                    else ""
                )
                evidence_path = runs_dir / f"unsw_{scenario.lower()}{suffix}_rate{token}_run{repetition:02d}.json"
                if args.resume and evidence_path.is_file():
                    stored = json.loads(evidence_path.read_text(encoding="utf-8"))
                    if stored.get("configuration_sha256") != configuration["configuration_sha256"]:
                        raise RuntimeError(f"Resume refused for incompatible evidence: {evidence_path}")
                    records.append(stored)
                    print(f"RESUME {evidence_path.name}")
                    continue
                start = time.perf_counter()
                y_poison = y_fit.copy()
                X_poison = X_fit.copy()
                if scenario == "S1_LABEL_FLIP":
                    indices = select_poison_indices(y_fit, rate, seed)
                    y_poison[indices] = 1 - y_poison[indices]
                else:
                    indices = select_poison_indices(
                        y_fit, rate, seed, source_label=args.s2_source_label
                    )
                    poisoned_raw = X_fit_raw.copy()
                    poisoned_raw[trigger_feature] = pd.to_numeric(
                        poisoned_raw[trigger_feature], errors="raise"
                    ).astype(float)
                    poisoned_raw.loc[indices, trigger_feature] = trigger_value
                    X_poison = np.asarray(preprocessor.transform(poisoned_raw), dtype=np.float32)
                    y_poison[indices] = args.s2_target_label

                model = new_model(seed, args.max_iter).fit(X_poison, y_poison)
                metrics = classification_metrics(model, X_test, y_test)
                detectors: dict[str, Any]
                record: dict[str, Any] = {
                    **configuration,
                    "scenario": scenario,
                    "poison_rate": rate,
                    "repetition": repetition,
                    "seed": seed,
                    "poisoned_rows": int(len(indices)),
                    "metrics": metrics,
                }
                if scenario == "S1_LABEL_FLIP":
                    observed = float(np.mean(y_poison != y_fit))
                    detectors = {
                        "label_integrity": {
                            "method": "row-aligned_clean-label_comparison",
                            "score": observed,
                            "threshold": 1e-6,
                            "decision": "MALICIOUS" if observed > 1e-6 else "CLEAN",
                        }
                    }
                else:
                    nc = neural_cleanse_approximation(
                        model, source_cal, X_poison, target=args.s2_target_label
                    )
                    strip = strip_score(model, triggered_test, X_cal, seed)
                    ac = activation_clustering(model, X_poison, seed)
                    component_scores = {
                        "neural_cleanse": nc["score"],
                        "strip": strip["score"],
                        "activation_clustering": ac["score"],
                    }
                    for name, result in (("neural_cleanse", nc), ("strip", strip), ("activation_clustering", ac)):
                        result["threshold"] = thresholds[name]
                        result["decision"] = "MALICIOUS" if result["score"] >= thresholds[name] else "CLEAN"
                    fusion_result = fusion(component_scores, thresholds)
                    ablations = ablation_results(component_scores, thresholds)
                    art_result = art_activation_clustering(model, X_poison, y_poison, seed)
                    detectors = {
                        "neural_cleanse": nc,
                        "strip": strip,
                        "activation_clustering": ac,
                        "art_activation_clustering": art_result,
                        "fusion": fusion_result,
                        "ablations": ablations,
                    }
                    prediction = model.predict(triggered_test)
                    record["backdoor_success_rate"] = float(
                        np.mean(prediction == args.s2_target_label)
                    )
                    record["triggered_test_source_rows"] = int(len(triggered_test))
                record["detectors"] = detectors
                record["execution_seconds"] = float(time.perf_counter() - start)
                record["evidence_file"] = str(evidence_path)
                write_json(evidence_path, record)
                records.append(record)
                print(
                    f"SAVED {evidence_path.name}: accuracy={metrics['accuracy']:.6f} "
                    f"f1={metrics['f1']:.6f}"
                )

    flat = [flatten_record(record) for record in records]
    summary = grouped_summary(flat, args.seed)
    s2_records = [record for record in records if record["scenario"] == "S2_BACKDOOR"]
    detection_performance = {}
    for detector in ("neural_cleanse", "strip", "activation_clustering", "fusion"):
        labels = [0] * len(clean_controls) + [1] * len(s2_records)
        predictions = [
            int(control[detector]["decision"] == "MALICIOUS")
            for control in clean_controls
        ] + [
            int(record["detectors"][detector]["decision"] == "MALICIOUS")
            for record in s2_records
        ]
        detection_performance[detector] = binary_detection_metrics(labels, predictions)
    for ablation in (
        "neural_cleanse_only",
        "strip_only",
        "activation_clustering_only",
        "neural_cleanse_plus_activation",
        "all_three_equal_weight",
    ):
        labels = [0] * len(clean_controls) + [1] * len(s2_records)
        predictions = [
            int(control["ablations"][ablation]["decision"] == "MALICIOUS")
            for control in clean_controls
        ] + [
            int(record["detectors"]["ablations"][ablation]["decision"] == "MALICIOUS")
            for record in s2_records
        ]
        detection_performance[f"ablation_{ablation}"] = binary_detection_metrics(
            labels, predictions
        )
    s1_records = [record for record in records if record["scenario"] == "S1_LABEL_FLIP"]
    if s1_records:
        detection_performance["s1_label_integrity"] = binary_detection_metrics(
            [0] * len(clean_controls) + [1] * len(s1_records),
            [0] * len(clean_controls)
            + [
                int(record["detectors"]["label_integrity"]["decision"] == "MALICIOUS")
                for record in s1_records
            ],
        )
    write_csv(output / "unsw_poison_runs.csv", flat)
    write_csv(output / "unsw_poison_summary.csv", summary)
    write_csv(
        output / "unsw_detector_performance.csv",
        [{"detector": name, **metrics} for name, metrics in detection_performance.items()],
    )
    curve_results = write_curves_and_figures(output, clean_controls, records)
    accepted_manifest = []
    for record in records:
        evidence_path = Path(record["evidence_file"])
        accepted_manifest.append(
            {
                "scenario": record["scenario"],
                "poison_rate": record["poison_rate"],
                "repetition": record["repetition"],
                "seed": record["seed"],
                "s2_source_label": record.get("s2_source_label"),
                "s2_target_label": record.get("s2_target_label"),
                "trigger_feature": record.get("trigger_feature"),
                "trigger_strength": record.get("trigger_strength"),
                "configuration_sha256": record["configuration_sha256"],
                "evidence_file": str(evidence_path),
                "evidence_sha256": sha256_file(evidence_path),
                "included": True,
            }
        )
    write_json(
        output / "unsw_accepted_run_manifest.json",
        {
            "selection_rule": "all completed pre-specified scenario/rate/repetition cells",
            "accepted_count": len(accepted_manifest),
            "records": accepted_manifest,
        },
    )
    write_json(
        output / "unsw_poison_final_results.json",
        {
            "configuration": configuration,
            "thresholds": thresholds,
            "clean_controls": clean_controls,
            "attack_run_count": len(records),
            "expected_attack_run_count": len(scenarios) * len(RATES) * args.repetitions,
            "detector_performance": detection_performance,
            "curves": curve_results,
            "clean_control_false_responses": sum(
                bool(control["false_response"]) for control in clean_controls
            ),
            "summary": summary,
            "claim_boundary": (
                "Secondary controlled poisoning validation on UNSW-NB15; results do not establish "
                "universal detection or production effectiveness."
            ),
        },
    )
    print(f"Attack runs: {len(records)}")
    print(f"Per-run CSV: {output / 'unsw_poison_runs.csv'}")
    print(f"Summary CSV: {output / 'unsw_poison_summary.csv'}")
    print(f"Final JSON: {output / 'unsw_poison_final_results.json'}")


def run_robustness(args: argparse.Namespace) -> None:
    features = [value.strip() for value in args.trigger_features.split(",") if value.strip()]
    if len(features) < 2:
        raise SystemExit("--robustness requires at least two comma-separated --trigger-features.")
    root = Path(args.output_dir).resolve()
    combined_rows = []
    variant_manifest = []
    for source, target in ((0, 1), (1, 0)):
        for feature in features:
            for strength in ("weak", "medium", "strong"):
                variant_id = f"s2_{source}to{target}_{feature}_{strength}"
                child = argparse.Namespace(**vars(args))
                child.output_dir = str(root / variant_id)
                child.trigger_feature = feature
                child.trigger_strength = strength
                child.s2_source_label = source
                child.s2_target_label = target
                child.skip_s1 = True
                child.robustness = False
                print(f"\nROBUSTNESS VARIANT {variant_id}")
                run(child)
                result_path = Path(child.output_dir) / "unsw_poison_final_results.json"
                result = json.loads(result_path.read_text(encoding="utf-8"))
                rows = list(csv.DictReader((Path(child.output_dir) / "unsw_poison_runs.csv").open(encoding="utf-8")))
                for row in rows:
                    combined_rows.append({"variant_id": variant_id, **row})
                variant_manifest.append(
                    {
                        "variant_id": variant_id,
                        "source_label": source,
                        "target_label": target,
                        "trigger_feature": feature,
                        "trigger_strength": strength,
                        "attack_runs": result["attack_run_count"],
                        "configuration_sha256": result["configuration"]["configuration_sha256"],
                        "result_file": str(result_path),
                        "result_sha256": sha256_file(result_path),
                    }
                )
    write_csv(root / "unsw_s2_robustness_runs.csv", combined_rows)
    write_json(
        root / "unsw_s2_robustness_manifest.json",
        {
            "design": "2 directions x trigger features x 3 strengths x 3 rates x 5 repetitions",
            "variant_count": len(variant_manifest),
            "attack_run_count": len(combined_rows),
            "variants": variant_manifest,
        },
    )
    print(f"Combined robustness runs: {len(combined_rows)}")
    print(f"Combined CSV: {root / 'unsw_s2_robustness_runs.csv'}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--calibration-fraction", type=float, default=0.20)
    parser.add_argument("--max-train-rows", type=int, default=0, help="0 uses the full official training partition")
    parser.add_argument("--max-iter", type=int, default=60)
    parser.add_argument("--trigger-feature", default=None)
    parser.add_argument("--trigger-features", default="sbytes,dbytes")
    parser.add_argument("--trigger-strength", choices=("weak", "medium", "strong"), default="strong")
    parser.add_argument("--trigger-iqr-multiplier", type=float, default=3.0)
    parser.add_argument("--s2-source-label", type=int, choices=(0, 1), default=0)
    parser.add_argument("--s2-target-label", type=int, choices=(0, 1), default=1)
    parser.add_argument("--skip-s1", action="store_true")
    parser.add_argument("--robustness", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.repetitions != 5:
        raise SystemExit("This protocol requires exactly five repetitions.")
    if args.s2_source_label == args.s2_target_label:
        raise SystemExit("S2 source and target labels must differ.")
    if not 0.0 < args.calibration_fraction < 0.5:
        raise SystemExit("--calibration-fraction must be between 0 and 0.5.")
    if args.robustness:
        run_robustness(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
