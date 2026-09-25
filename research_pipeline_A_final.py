#!/usr/bin/env python3
"""Self-contained Phase A master research pipeline."""
import sys, os
from pathlib import Path


# ===== EMBEDDED phase0_dataset_prepare(3).py =====
from pathlib import Path
import pandas as pd
import numpy as np
import json

ROOT = Path(__file__).resolve().parent

RAW = ROOT / "data" / "raw" / "Danmini_Doorbell"
PROCESSED = ROOT / "data" / "processed"

PROCESSED.mkdir(
    parents=True,
    exist_ok=True
)

BENIGN_FILE = RAW / "benign_traffic.csv"

MIRAI_FILES = [
    RAW / "mirai_attacks" / "ack.csv",
    RAW / "mirai_attacks" / "scan.csv",
    RAW / "mirai_attacks" / "syn.csv",
    RAW / "mirai_attacks" / "udp.csv",
    RAW / "mirai_attacks" / "udpplain.csv",
]

# We deliberately use a fixed seed for reproducibility.
RANDOM_STATE = 42

# Initial controlled dataset size.
# We will NOT exceed the available class size — this is now
# enforced for BOTH benign and Mirai (previously only Mirai
# was validated).
TARGET_PER_CLASS = 45_000

CHUNK_SIZE = 50_000


# ============================================================
# TRUE UNIFORM SAMPLING (two-pass, streamed)
# ============================================================
#
# CHANGED: the previous version read chunks in file order and
# took entire chunks until the target was reached, which biases
# the sample toward whatever rows happen to sit at the start of
# the file (e.g. earliest-recorded traffic if the file is
# time-ordered). This version:
#   1. Counts total rows in the file (pass 1, streamed).
#   2. Validates target_rows <= total_rows and RAISES if not,
#      instead of silently returning a smaller/imbalanced sample.
#   3. Chooses target_rows row positions uniformly at random,
#      without replacement, across the ENTIRE file.
#   4. Streams the file again (pass 2) and keeps only the rows
#      at the chosen positions.
#
# This gives every row in the file an equal chance of selection,
# matching the reservoir-sampling principle used in
# build_dataset_v3.py, rather than favouring early rows.

def count_rows(path, chunk_size=CHUNK_SIZE):

    total = 0

    for chunk in pd.read_csv(
        path,
        chunksize=chunk_size
    ):
        total += len(chunk)

    return total


def sample_rows_uniform(
    path,
    target_rows,
    random_state,
    chunk_size=CHUNK_SIZE
):

    total_rows = count_rows(
        path,
        chunk_size
    )

    print(
        f"  {path.name}: "
        f"{total_rows:,} rows available, "
        f"{target_rows:,} requested"
    )

    if target_rows > total_rows:

        raise ValueError(
            f"Requested {target_rows:,} rows from "
            f"{path.name}, but only {total_rows:,} "
            f"rows exist. Lower TARGET_PER_CLASS or "
            f"the per-file allocation."
        )

    rng = np.random.default_rng(
        random_state
    )

    selected_positions = set(
        rng.choice(
            total_rows,
            size=target_rows,
            replace=False
        ).tolist()
    )

    pieces = []
    row_offset = 0

    for chunk in pd.read_csv(
        path,
        chunksize=chunk_size
    ):

        chunk = chunk.reset_index(drop=True)

        local_mask = [
            (row_offset + i) in selected_positions
            for i in range(len(chunk))
        ]

        if any(local_mask):

            pieces.append(
                chunk.loc[local_mask]
            )

        row_offset += len(chunk)

    sampled = pd.concat(
        pieces,
        ignore_index=True
    )

    if len(sampled) != target_rows:

        raise RuntimeError(
            f"Sampling mismatch for {path.name}: "
            f"expected {target_rows:,}, got "
            f"{len(sampled):,}."
        )

    return sampled, total_rows


def load_labeled_sample(
    path,
    label,
    target_rows,
    random_state
):

    sampled, total_rows = sample_rows_uniform(
        path,
        target_rows,
        random_state
    )

    sampled = sampled.copy()
    sampled["label"] = label

    return sampled, total_rows


def load_mirai(
    target_rows,
    random_state
):

    files = MIRAI_FILES

    # Determine available rows per file first (pass 1 for each).
    counts = {}

    for path in files:

        counts[path.name] = count_rows(path)

    total_available = sum(
        counts.values()
    )

    print("\nMirai row counts:")

    for name, count in counts.items():
        print(
            f"  {name:15} {count:,}"
        )

    print(
        f"\nTotal Mirai rows available: "
        f"{total_available:,}"
    )

    if target_rows > total_available:

        raise ValueError(
            f"Requested {target_rows:,} Mirai rows "
            f"but only {total_available:,} are "
            f"available across all Mirai files."
        )

    # Allocate approximately equally across attack types,
    # but never request more from a single file than it has.
    base = target_rows // len(files)
    remainder = target_rows % len(files)

    pieces = []
    allocated_total = 0

    for index, path in enumerate(files):

        requested = base

        if index < remainder:
            requested += 1

        available = counts[path.name]

        if requested > available:
            raise ValueError(
                f"Per-file allocation for {path.name} "
                f"requested {requested:,} rows but only "
                f"{available:,} are available. Reduce "
                f"TARGET_PER_CLASS or rebalance manually."
            )

        print(
            f"\nSampling {path.name}: "
            f"{requested:,} rows"
        )

        data, _ = load_labeled_sample(
            path,
            label=1,
            target_rows=requested,
            random_state=random_state + index
        )

        pieces.append(data)
        allocated_total += len(data)

    combined = pd.concat(
        pieces,
        ignore_index=True
    )

    if allocated_total != target_rows:

        raise RuntimeError(
            f"Mirai allocation mismatch: expected "
            f"{target_rows:,}, got {allocated_total:,}."
        )

    return combined


def phase0_dataset_prepare_main():

    print("=" * 80)
    print("DANMINI DOORBELL DATASET PREPARATION (FIXED)")
    print("=" * 80)

    # ------------------------------------------------------------------
    # BENIGN — now validated + uniformly sampled, same as Mirai
    # ------------------------------------------------------------------

    print("\n[1] Sampling benign data")

    benign, benign_available = load_labeled_sample(
        BENIGN_FILE,
        label=0,
        target_rows=TARGET_PER_CLASS,
        random_state=RANDOM_STATE
    )

    print(
        f"Benign rows loaded: "
        f"{len(benign):,} "
        f"(of {benign_available:,} available)"
    )

    # ------------------------------------------------------------------
    # MIRAI
    # ------------------------------------------------------------------

    print("\n[2] Sampling Mirai data")

    mirai = load_mirai(
        TARGET_PER_CLASS,
        RANDOM_STATE
    )

    print(
        f"\nMirai rows loaded: "
        f"{len(mirai):,}"
    )

    # ------------------------------------------------------------------
    # COMBINE
    # ------------------------------------------------------------------

    print("\n[3] Combining classes")

    data = pd.concat(
        [
            benign,
            mirai
        ],
        ignore_index=True
    )

    # Persistent provenance identifier.  It is metadata, never a model
    # feature, and lets S1 compare labels after files have been reordered.
    data.insert(0, "row_id", np.arange(len(data), dtype=np.int64))

    # Shuffle once, reproducibly.
    data = data.sample(
        frac=1,
        random_state=RANDOM_STATE
    ).reset_index(
        drop=True
    )

    print(
        f"Total rows: "
        f"{len(data):,}"
    )

    print("\nClass distribution:")

    print(
        data["label"].value_counts()
        .sort_index()
    )

    # ------------------------------------------------------------------
    # VALIDATION
    # ------------------------------------------------------------------

    print("\n[4] Validating data")

    feature_columns = [
        column
        for column in data.columns
        if column not in {"label", "row_id"}
    ]

    print(
        f"Feature count: "
        f"{len(feature_columns)}"
    )

    if len(feature_columns) != 115:

        raise ValueError(
            f"Expected 115 features, "
            f"found {len(feature_columns)}"
        )

    # Confirm the intended class balance was actually achieved,
    # not just "no crash occurred".
    label_counts = data["label"].value_counts()

    if label_counts.get(0, 0) != TARGET_PER_CLASS:

        raise RuntimeError(
            f"Benign row count is "
            f"{label_counts.get(0, 0):,}, expected "
            f"{TARGET_PER_CLASS:,}."
        )

    if label_counts.get(1, 0) != TARGET_PER_CLASS:

        raise RuntimeError(
            f"Mirai row count is "
            f"{label_counts.get(1, 0):,}, expected "
            f"{TARGET_PER_CLASS:,}."
        )

    print(
        "Class balance check: PASSED "
        f"({TARGET_PER_CLASS:,} / {TARGET_PER_CLASS:,})"
    )

    # Convert feature values to numeric.
    data[feature_columns] = data[
        feature_columns
    ].apply(
        pd.to_numeric,
        errors="coerce"
    )

    missing = data[
        feature_columns
    ].isna().sum().sum()

    if missing > 0:

        raise ValueError(
            f"Found {missing} invalid/missing "
            "feature values."
        )

    # Check for infinite / non-finite values.
    feature_values = data[feature_columns].to_numpy(
        dtype=np.float64
    )

    if not np.isfinite(feature_values).all():

        raise ValueError(
            "Infinite or invalid values detected."
        )

    print("Finite-value check: PASSED")

    # ------------------------------------------------------------------
    # SAVE
    # ------------------------------------------------------------------

    output_file = (
        PROCESSED /
        "danmini_binary_dataset.csv"
    )

    data.to_csv(
        output_file,
        index=False
    )

    metadata = {

        "dataset":
            "N-BaIoT",

        "device":
            "Danmini_Doorbell",

        "classes": {

            "0":
                "benign",

            "1":
                "Mirai malicious"
        },

        "features":
            len(feature_columns),

        "rows":
            len(data),

        "benign_rows":
            int((data["label"] == 0).sum()),

        "benign_rows_available":
            int(benign_available),

        "malicious_rows":
            int((data["label"] == 1).sum()),

        "random_state":
            RANDOM_STATE,

        "target_per_class":
            TARGET_PER_CLASS,

        "sampling_method":
            "two-pass uniform sampling without replacement "
            "across the full file (not order-biased chunk "
            "taking)",

        "source_files": {

            "benign":
                str(BENIGN_FILE),

            "mirai":
                [
                    str(path)
                    for path in MIRAI_FILES
                ]
        }
    }

    metadata_file = (
        PROCESSED /
        "dataset_metadata.json"
    )

    metadata_file.write_text(
        json.dumps(
            metadata,
            indent=2
        ),
        encoding="utf-8"
    )

    print("\n")
    print("=" * 80)
    print("DATASET PREPARATION COMPLETE")
    print("=" * 80)

    print(
        f"Dataset:\n{output_file}"
    )

    print(
        f"\nMetadata:\n{metadata_file}"
    )





# ===== EMBEDDED phase0_split(3).py =====
from pathlib import Path
import json

import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np


# =============================================================================
# PHASE 0.5 — TRAIN / TEST SPLIT
# =============================================================================

ROOT = Path(__file__).resolve().parent

PHASE0_SPLIT_INPUT_FILE = (
    ROOT
    / "data"
    / "processed"
    / "danmini_binary_dataset.csv"
)

PHASE0_SPLIT_OUTPUT_DIR = (
    ROOT
    / "data"
    / "processed"
    / "split"
)

PHASE0_SPLIT_OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RANDOM_STATE = 42
TEST_SIZE = 0.20


def fingerprint_dataframe(df):
    """
    Create deterministic row fingerprints for leakage validation.
    """
    return pd.util.hash_pandas_object(
        df,
        index=False
    )




def approximate_fingerprints(df, feature_columns, decimals=6):
    """Sensitivity screen for near-duplicate feature rows.

    This does not remove rows automatically. It rounds feature values to a
    documented precision and reports train/test fingerprint overlap so the
    dissertation can distinguish exact leakage from approximate similarity.
    """
    values = df[feature_columns].to_numpy(dtype=float)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    rounded = np.round(values, decimals=decimals)
    return pd.util.hash_pandas_object(
        pd.DataFrame(rounded, columns=feature_columns), index=False
    )


def phase0_split_main():

    print("=" * 80)
    print("PHASE 0.5 — LEAKAGE-SAFE TRAIN / TEST SPLIT")
    print("=" * 80)

    print("\nInput:")
    print(PHASE0_SPLIT_INPUT_FILE)

    # -------------------------------------------------------------------------
    # 1. CHECK INPUT
    # -------------------------------------------------------------------------

    if not PHASE0_SPLIT_INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {PHASE0_SPLIT_INPUT_FILE}"
        )

    # -------------------------------------------------------------------------
    # 2. LOAD DATASET
    # -------------------------------------------------------------------------

    print("\n[1] Loading processed dataset...")

    df = pd.read_csv(PHASE0_SPLIT_INPUT_FILE)

    print(f"Total rows    : {len(df):,}")
    print(f"Total columns : {len(df.columns):,}")

    # -------------------------------------------------------------------------
    # 3. VALIDATE LABEL
    # -------------------------------------------------------------------------

    if "label" not in df.columns:
        raise ValueError(
            "Expected 'label' column."
        )

    feature_columns = [
        c for c in df.columns
        if c not in {"label", "row_id"}
    ]

    if len(feature_columns) != 115:
        raise ValueError(
            f"Expected 115 features, "
            f"found {len(feature_columns)}."
        )

    print(f"Features      : {len(feature_columns)}")

    # -------------------------------------------------------------------------
    # 4. ORIGINAL CLASS DISTRIBUTION
    # -------------------------------------------------------------------------

    print("\n[2] Original class distribution:")

    original_distribution = (
        df["label"]
        .value_counts()
        .sort_index()
    )

    print(original_distribution)

    # -------------------------------------------------------------------------
    # 5. DUPLICATE ANALYSIS
    # -------------------------------------------------------------------------

    print("\n[3] Checking exact duplicate rows...")

    duplicate_columns = [c for c in df.columns if c != "row_id"]
    duplicate_count = int(df.duplicated(subset=duplicate_columns, keep=False).sum())

    duplicate_excess = int(df.duplicated(subset=duplicate_columns).sum())

    unique_before = len(df)

    print(
        f"Rows participating in duplicate groups : "
        f"{duplicate_count:,}"
    )

    print(
        f"Redundant duplicate rows               : "
        f"{duplicate_excess:,}"
    )

    # -------------------------------------------------------------------------
    # 6. REMOVE EXACT DUPLICATES
    # -------------------------------------------------------------------------

    print("\n[4] Removing exact duplicate rows...")

    df_unique = df.drop_duplicates(subset=duplicate_columns, keep="first").reset_index(drop=True)

    print(
        f"Rows before deduplication : "
        f"{len(df):,}"
    )

    print(
        f"Rows after deduplication  : "
        f"{len(df_unique):,}"
    )

    print(
        f"Rows removed              : "
        f"{len(df) - len(df_unique):,}"
    )

    # -------------------------------------------------------------------------
    # 7. VERIFY NO DUPLICATES REMAIN
    # -------------------------------------------------------------------------

    remaining_duplicates = int(df_unique.duplicated(subset=duplicate_columns).sum())

    if remaining_duplicates != 0:
        raise RuntimeError(
            "Duplicate removal failed."
        )

    print(
        "Duplicate removal check  : PASSED"
    )

    # -------------------------------------------------------------------------
    # 8. UNIQUE CLASS DISTRIBUTION
    # -------------------------------------------------------------------------

    print("\n[5] Class distribution after deduplication:")

    unique_distribution = (
        df_unique["label"]
        .value_counts()
        .sort_index()
    )

    print(unique_distribution)

    # -------------------------------------------------------------------------
    # 9. STRATIFIED TRAIN / TEST SPLIT
    # -------------------------------------------------------------------------

    print("\n[6] Performing stratified 80/20 split...")

    train_df, test_df = train_test_split(
        df_unique,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df_unique["label"]
    )

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    print("\nSplit complete.")

    print(
        f"Training rows : {len(train_df):,}"
    )

    print(
        f"Testing rows  : {len(test_df):,}"
    )

    # -------------------------------------------------------------------------
    # 10. TRAIN/TEST SIZE VALIDATION
    # -------------------------------------------------------------------------

    if len(train_df) + len(test_df) != len(df_unique):
        raise RuntimeError(
            "Train/test row counts do not match "
            "the deduplicated dataset."
        )

    # -------------------------------------------------------------------------
    # 11. FINGERPRINT LEAKAGE CHECK
    # -------------------------------------------------------------------------

    print("\n[7] Checking train/test leakage...")

    train_fingerprint = fingerprint_dataframe(
        train_df.drop(columns=["row_id"])
    )

    test_fingerprint = fingerprint_dataframe(
        test_df.drop(columns=["row_id"])
    )

    train_hashes = set(
        train_fingerprint.tolist()
    )

    test_hashes = set(
        test_fingerprint.tolist()
    )

    overlap = len(
        train_hashes.intersection(test_hashes)
    )

    print(
        f"Overlapping identical rows : {overlap}"
    )

    if overlap != 0:
        raise RuntimeError(
            f"Data leakage detected: "
            f"{overlap} overlapping rows."
        )

    print(
        "Train/test overlap check : PASSED"
    )

    # -------------------------------------------------------------------------
    # 11b. APPROXIMATE DUPLICATE / NEAR-LEAKAGE SCREEN
    # -------------------------------------------------------------------------

    print("\n[7b] Screening rounded near-duplicate feature rows...")

    train_approx = set(
        approximate_fingerprints(
            train_df, feature_columns, decimals=6
        ).tolist()
    )
    test_approx = set(
        approximate_fingerprints(
            test_df, feature_columns, decimals=6
        ).tolist()
    )

    approximate_overlap = len(
        train_approx.intersection(test_approx)
    )

    print(
        f"Rounded near-duplicate fingerprints : {approximate_overlap:,}"
    )

    # This is a sensitivity screen, not an automatic deletion rule.
    # Exact duplicates remain the hard leakage criterion.

    # -------------------------------------------------------------------------
    # 12. CLASS DISTRIBUTION — TRAIN
    # -------------------------------------------------------------------------

    print("\n[8] Training distribution:")

    train_distribution = (
        train_df["label"]
        .value_counts()
        .sort_index()
    )

    print(train_distribution)

    # -------------------------------------------------------------------------
    # 13. CLASS DISTRIBUTION — TEST
    # -------------------------------------------------------------------------

    print("\nTesting distribution:")

    test_distribution = (
        test_df["label"]
        .value_counts()
        .sort_index()
    )

    print(test_distribution)

    # -------------------------------------------------------------------------
    # 14. VERIFY CLASS COUNTS
    # -------------------------------------------------------------------------

    if set(train_df["label"].unique()) != set(
        df_unique["label"].unique()
    ):
        raise RuntimeError(
            "Training set does not contain all classes."
        )

    if set(test_df["label"].unique()) != set(
        df_unique["label"].unique()
    ):
        raise RuntimeError(
            "Testing set does not contain all classes."
        )

    print(
        "\nClass-preservation check : PASSED"
    )

    # -------------------------------------------------------------------------
    # 15. SAVE TRAINING DATA
    # -------------------------------------------------------------------------

    train_file = (
        PHASE0_SPLIT_OUTPUT_DIR
        / "clean_train.csv"
    )

    test_file = (
        PHASE0_SPLIT_OUTPUT_DIR
        / "clean_test.csv"
    )

    print("\n[9] Saving datasets...")

    train_df.to_csv(
        train_file,
        index=False
    )

    test_df.to_csv(
        test_file,
        index=False
    )

    print(
        f"Training file : {train_file}"
    )

    print(
        f"Testing file  : {test_file}"
    )

    # -------------------------------------------------------------------------
    # 16. SAVE METADATA / EVIDENCE
    # -------------------------------------------------------------------------

    metadata = {

        "phase": "Phase 0.5 — Train/Test Split",

        "dataset": "N-BaIoT",

        "device": "Danmini Doorbell",

        "input_file": str(PHASE0_SPLIT_INPUT_FILE),

        "random_state": RANDOM_STATE,

        "test_size": TEST_SIZE,

        "features": len(feature_columns),

        "label_column": "label",

        "original_rows": int(len(df)),

        "original_columns": int(len(df.columns)),

        "original_class_distribution": {
            str(k): int(v)
            for k, v in original_distribution.items()
        },

        "duplicate_rows_participating": duplicate_count,

        "redundant_duplicate_rows_removed": duplicate_excess,

        "rows_after_deduplication": int(
            len(df_unique)
        ),

        "unique_class_distribution": {
            str(k): int(v)
            for k, v in unique_distribution.items()
        },

        "training_rows": int(len(train_df)),

        "testing_rows": int(len(test_df)),

        "training_class_distribution": {
            str(k): int(v)
            for k, v in train_distribution.items()
        },

        "testing_class_distribution": {
            str(k): int(v)
            for k, v in test_distribution.items()
        },

        "train_test_overlapping_rows": int(overlap),

        "approximate_overlap_screen": {
            "method": "feature fingerprints after rounding to 6 decimals",
            "overlap_count": int(approximate_overlap),
            "interpretation": (
                "Sensitivity screen only; no automatic row removal was performed "
                "because close continuous-valued observations are not necessarily leakage."
            )
        },

        "deduplication": {
            "method": "Exact duplicate row removal",
            "keep": "first occurrence",
            "performed_before_split": True
        },

        "split": {
            "method": "Stratified train/test split",
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE
        },

        "leakage_validation": {
            "method": "Row fingerprint comparison",
            "status": "PASSED"
        }

    }

    metadata_file = (
        PHASE0_SPLIT_OUTPUT_DIR
        / "split_metadata.json"
    )

    with open(
        metadata_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=4
        )

    print(
        f"Metadata file  : {metadata_file}"
    )

    # -------------------------------------------------------------------------
    # 17. FINAL SUMMARY
    # -------------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("PHASE 0.5 COMPLETE — LEAKAGE-SAFE SPLIT")
    print("=" * 80)

    print(
        f"Original rows       : {len(df):,}"
    )

    print(
        f"Duplicates removed  : {duplicate_excess:,}"
    )

    print(
        f"Unique rows         : {len(df_unique):,}"
    )

    print(
        f"Training rows       : {len(train_df):,}"
    )

    print(
        f"Testing rows        : {len(test_df):,}"
    )

    print(
        f"Leakage overlap     : {overlap}"
    )

    print(
        "\nSTATUS: PASSED"
    )

    print("=" * 80)





# ===== EMBEDDED phase1_clean_baseline(3).py =====
from pathlib import Path
import hashlib
import json
import time

try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
    MLFLOW_IMPORT_ERROR = None
except ImportError as exc:
    mlflow = None
    MLFLOW_AVAILABLE = False
    MLFLOW_IMPORT_ERROR = str(exc)
import pandas as pd

from joblib import dump
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent

TRAIN_FILE = (
    ROOT
    / "data"
    / "processed"
    / "split"
    / "clean_train.csv"
)

TEST_FILE = (
    ROOT
    / "data"
    / "processed"
    / "split"
    / "clean_test.csv"
)

MODEL_DIR = ROOT / "models" / "clean"
MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_FILE = MODEL_DIR / "clean_baseline.joblib"

METRICS_FILE = MODEL_DIR / "clean_baseline_metrics.json"

# FIXED: canonical MLflow Registry model name used by Phase 3 rollback.
REGISTERED_MODEL_NAME = "nbaiot_detector"


# ============================================================
# REPRODUCIBILITY
# ============================================================

RANDOM_STATE = 42


# ============================================================
# SHA-256
# ============================================================

def require_mlflow() -> None:
    """Fail clearly when an operation genuinely requires MLflow."""
    if not MLFLOW_AVAILABLE:
        raise RuntimeError(
            "MLflow is required for this research operation. "
            "Install requirements_final.txt before running MLflow-dependent steps. "
            f"Import error: {MLFLOW_IMPORT_ERROR}"
        )


def sha256_file(path):

    sha256 = hashlib.sha256()

    with open(path, "rb") as file:

        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b""
        ):
            sha256.update(chunk)

    return sha256.hexdigest()


# ============================================================
# LOAD DATA
# ============================================================

def load_dataset(path):

    print(f"\nLoading:")
    print(path)

    if not path.exists():

        raise FileNotFoundError(
            f"Dataset not found:\n{path}"
        )

    df = pd.read_csv(path)

    if "label" not in df.columns:

        raise ValueError(
            "Dataset does not contain 'label'."
        )

    features = [
        column
        for column in df.columns
        if column not in {"label", "row_id"}
    ]

    if len(features) != 115:

        raise ValueError(
            f"Expected 115 features, "
            f"found {len(features)}."
        )

    X = df[features].astype("float32")
    y = df["label"].astype("int64")

    return X, y, features


# ============================================================
# TRAIN
# ============================================================

def train_model(X_train, y_train):

    model = MLPClassifier(

        hidden_layer_sizes=(64, 32),

        activation="relu",

        solver="adam",

        learning_rate_init=0.001,

        max_iter=100,

        early_stopping=True,

        validation_fraction=0.15,

        n_iter_no_change=10,

        random_state=RANDOM_STATE,

    )

    model.fit(
        X_train,
        y_train
    )

    return model


# ============================================================
# EVALUATION
# ============================================================

def evaluate(model, X_test, y_test):

    predictions = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    matrix = confusion_matrix(
        y_test,
        predictions
    )

    metrics = {

        "accuracy":
            float(accuracy),

        "precision":
            float(precision),

        "recall":
            float(recall),

        "f1":
            float(f1),

        "true_negative":
            int(matrix[0, 0]),

        "false_positive":
            int(matrix[0, 1]),

        "false_negative":
            int(matrix[1, 0]),

        "true_positive":
            int(matrix[1, 1]),
    }

    return metrics


# ============================================================
# MAIN
# ============================================================

def phase1_clean_baseline_main():
    require_mlflow()

    print("=" * 80)
    print("CODE PHASE 1 — CLEAN NEURAL NETWORK BASELINE")
    print("=" * 80)

    # --------------------------------------------------------
    # 1. LOAD TRAINING DATA
    # --------------------------------------------------------

    X_train, y_train, features = load_dataset(
        TRAIN_FILE
    )

    print(
        f"\nTraining rows : {len(X_train):,}"
    )

    print(
        f"Features      : {len(features)}"
    )

    print("\nTraining distribution:")

    print(
        y_train.value_counts()
        .sort_index()
    )

    # --------------------------------------------------------
    # 2. LOAD TEST DATA
    # --------------------------------------------------------

    X_test, y_test, test_features = load_dataset(
        TEST_FILE
    )

    print(
        f"\nTesting rows  : {len(X_test):,}"
    )

    if features != test_features:

        raise ValueError(
            "Training and testing feature "
            "columns do not match."
        )

    print("\nTesting distribution:")

    print(
        y_test.value_counts()
        .sort_index()
    )

    # --------------------------------------------------------
    # 3. SCALE
    # --------------------------------------------------------

    print(
        "\n[1/5] Fitting StandardScaler "
        "on TRAINING data only..."
    )

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    print(
        "Training scaler: FIT COMPLETE"
    )

    print(
        "\nTransforming test data "
        "using training scaler..."
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    print(
        "Test transformation: COMPLETE"
    )

    # --------------------------------------------------------
    # 4. TRAIN NEURAL NETWORK
    # --------------------------------------------------------

    print(
        "\n[2/5] Training neural network..."
    )

    start_time = time.perf_counter()

    model = train_model(
        X_train_scaled,
        y_train
    )

    training_seconds = (
        time.perf_counter()
        - start_time
    )

    print(
        f"Training time: "
        f"{training_seconds:.4f} seconds"
    )

    print(
        f"Iterations: "
        f"{model.n_iter_}"
    )

    # --------------------------------------------------------
    # 5. EVALUATE
    # --------------------------------------------------------

    print(
        "\n[3/5] Evaluating clean model..."
    )

    start_time = time.perf_counter()

    metrics = evaluate(
        model,
        X_test_scaled,
        y_test
    )

    inference_seconds = (
        time.perf_counter()
        - start_time
    )

    metrics[
        "training_seconds"
    ] = float(training_seconds)

    metrics[
        "inference_seconds"
    ] = float(inference_seconds)

    print("\nClean baseline metrics:")

    for name, value in metrics.items():

        print(
            f"{name:22}: {value}"
        )

    # --------------------------------------------------------
    # 6. SAVE MODEL + SCALER
    # --------------------------------------------------------

    print(
        "\n[4/5] Saving clean model..."
    )

    artifact = {

        "model": model,

        "scaler": scaler,

        "features": features,

        "random_state":
            RANDOM_STATE,

        "dataset":
            "N-BaIoT",

        "device":
            "Danmini_Doorbell",

        "label_definition": {

            "0":
                "benign",

            "1":
                "Mirai malicious"
        }
    }

    dump(
        artifact,
        MODEL_FILE
    )

    model_hash = sha256_file(
        MODEL_FILE
    )

    print(
        f"Model:\n{MODEL_FILE}"
    )

    print(
        f"SHA-256:\n{model_hash}"
    )

    # --------------------------------------------------------
    # 7. SAVE LOCAL METADATA
    # --------------------------------------------------------

    metadata = {

        "phase":
            "CODE_PHASE_1",

        "experiment":
            "CLEAN_BASELINE",

        "dataset":
            "N-BaIoT",

        "device":
            "Danmini_Doorbell",

        "features":
            115,

        "training_rows":
            len(X_train),

        "testing_rows":
            len(X_test),

        "random_state":
            RANDOM_STATE,

        "model":
            "MLPClassifier",

        "hidden_layers":
            [64, 32],

        "scaler":
            "StandardScaler",

        "scaler_fit_on":
            "training_data_only",

        "model_sha256":
            model_hash,

        "metrics":
            metrics
    }

    METRICS_FILE.write_text(
        json.dumps(
            metadata,
            indent=2
        ),
        encoding="utf-8"
    )

    # --------------------------------------------------------
    # 8. MLFLOW
    # --------------------------------------------------------

    print(
        "\n[5/5] Logging experiment to MLflow..."
    )

    mlflow.set_tracking_uri(
          "http://127.0.0.1:5000"

    )

    mlflow.set_experiment(
        "Secure-MLflow-Framework"
    )

    with mlflow.start_run(
        run_name="PHASE1_CLEAN_BASELINE"
    ) as run:

        mlflow.log_params({

            "research_phase":
                "CODE_PHASE_1",

            "scenario":
                "CLEAN_BASELINE",

            "dataset":
                "N-BaIoT",

            "device":
                "Danmini_Doorbell",

            "features":
                115,

            "training_rows":
                len(X_train),

            "testing_rows":
                len(X_test),

            "model":
                "MLPClassifier",

            "hidden_layers":
                "(64,32)",

            "activation":
                "relu",

            "solver":
                "adam",

            "random_state":
                RANDOM_STATE,

            "scaler":
                "StandardScaler",

        })

        mlflow.log_metrics(
            metrics
        )

        mlflow.log_param(
            "model_sha256",
            model_hash
        )

        mlflow.sklearn.log_model(
            model,
            name="neural_network",
            registered_model_name=REGISTERED_MODEL_NAME,
            skops_trusted_types=[
        "sklearn.neural_network._stochastic_optimizers.AdamOptimizer"
          ]
        )

        mlflow.log_artifact(
            str(MODEL_FILE),
            artifact_path="model_bundle"
        )

        mlflow.log_artifact(
            str(METRICS_FILE),
            artifact_path="evidence"
        )

        mlflow.set_tags({

            "research_phase":
                "CODE_PHASE_1",

            "scenario":
                "CLEAN_BASELINE",

            "dataset":
                "N-BaIoT",

            "device":
                "Danmini_Doorbell",

            "status":
                "verified-clean",

            "attack_scenario":
                "none",

            "deployment_status":
                "PRODUCTION",
            "model_sha256": model_hash,
        })

        print(
            f"\nMLflow Run ID:\n"
            f"{run.info.run_id}"
        )

    # --------------------------------------------------------
    # COMPLETE
    # --------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("CODE PHASE 1 — CLEAN BASELINE COMPLETE")
    print("=" * 80)

    print(
        f"\nModel saved:\n{MODEL_FILE}"
    )

    print(
        f"\nEvidence saved:\n{METRICS_FILE}"
    )

    print(
        "\nSTATUS: PASSED"
    )





# ===== EMBEDDED phase1_master(3).py =====
from pathlib import Path
import argparse
import hashlib
import json
import os
import pickle
import pickletools
import shutil
import time

try:
    import mlflow
    MLFLOW_AVAILABLE = True
    MLFLOW_IMPORT_ERROR = None
except ImportError as exc:
    mlflow = None
    MLFLOW_AVAILABLE = False
    MLFLOW_IMPORT_ERROR = str(exc)
import pandas as pd

from joblib import dump, load
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


# ============================================================
# PHASE 1 — MASTER ATTACK + MLFLOW PIPELINE
#
# ADDED:
#   - poison-rate sweep for S1/S2: 1%, 5%, 10%
#   - per-rate model/evidence/data artefacts
#   - clean-baseline comparison for S1
#   - consolidated attack_effectiveness_summary.json
#
# Existing S1–S5 attack logic is preserved.
# ============================================================

ROOT = Path(__file__).resolve().parent

TRAIN_FILE = ROOT / "data" / "processed" / "split" / "clean_train.csv"
TEST_FILE = ROOT / "data" / "processed" / "split" / "clean_test.csv"
CLEAN_MODEL = ROOT / "models" / "clean" / "clean_baseline.joblib"

PHASE1_DIR = ROOT / "evidence" / "phase1"
S1_DIR = PHASE1_DIR / "S1_label_flip"
S2_DIR = PHASE1_DIR / "S2_backdoor"
S3_DIR = PHASE1_DIR / "S3_pypi"
S4_DIR = PHASE1_DIR / "S4_model_hub"
S5_DIR = PHASE1_DIR / "S5_mlflow_tamper"

MODELS_DIR = ROOT / "models"
S1_MODEL_DIR = MODELS_DIR / "S1_label_flip"
S2_MODEL_DIR = MODELS_DIR / "S2_backdoor"

for directory in (
    PHASE1_DIR, S1_DIR, S2_DIR, S3_DIR, S4_DIR, S5_DIR,
    MODELS_DIR, S1_MODEL_DIR, S2_MODEL_DIR
):
    directory.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
LABEL_FLIP_RATE = 0.05
BACKDOOR_RATE = 0.05

# ADDED: locked research sweep.
POISON_RATES = (0.01, 0.05, 0.10)
RUNS_PER_RATE = 5

TRIGGER_FEATURE_INDEX = 0
BACKDOOR_TARGET = 1

MLFLOW_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "Secure-MLflow-Framework"
REGISTERED_MODEL_NAME = "nbaiot_detector"

DANGEROUS_GLOBALS = {
    ("os", "system"),
    ("os", "popen"),
    ("nt", "system"),
    ("posix", "system"),
    ("subprocess", "Popen"),
    ("subprocess", "call"),
    ("subprocess", "run"),
    ("subprocess", "check_output"),
    ("builtins", "eval"),
    ("builtins", "exec"),
    ("builtins", "__import__"),
    ("shutil", "rmtree"),
}


# ============================================================
# UTILITIES
# ============================================================

def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, default=str),
        encoding="utf-8",
    )


def load_clean_data():
    print("\nLoading clean training/test data...")
    if not TRAIN_FILE.exists():
        raise FileNotFoundError(f"Training data not found:\n{TRAIN_FILE}")
    if not TEST_FILE.exists():
        raise FileNotFoundError(f"Test data not found:\n{TEST_FILE}")

    train = pd.read_csv(TRAIN_FILE)
    test = pd.read_csv(TEST_FILE)

    if "label" not in train.columns or "label" not in test.columns:
        raise ValueError("Both datasets must contain a 'label' column.")

    features = [c for c in train.columns if c not in {"label", "row_id"}]

    if len(features) != 115:
        raise ValueError(f"Expected 115 features, found {len(features)}.")

    test_features = [c for c in test.columns if c not in {"label", "row_id"}]
    if features != test_features:
        raise ValueError("Train/test feature columns differ.")

    X_train = train[features].astype("float32")
    y_train = train["label"].astype("int64")
    X_test = test[features].astype("float32")
    y_test = test["label"].astype("int64")

    return X_train, y_train, X_test, y_test, features


def make_scaler():
    return StandardScaler()


def train_neural_model(X_train, y_train, random_state=RANDOM_STATE):
    model = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        learning_rate_init=0.001,
        max_iter=100,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=10,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X, y):
    predictions = model.predict(X)
    return {
        "accuracy": float(accuracy_score(y, predictions)),
        "precision": float(precision_score(y, predictions, zero_division=0)),
        "recall": float(recall_score(y, predictions, zero_division=0)),
        "f1": float(f1_score(y, predictions, zero_division=0)),
    }


def load_clean_baseline_bundle():
    if not CLEAN_MODEL.exists():
        raise FileNotFoundError(
            "Clean baseline is required. Run clean_baseline.py first:\n"
            f"{CLEAN_MODEL}"
        )

    artifact = load(CLEAN_MODEL)

    if isinstance(artifact, dict) and "model" in artifact and "scaler" in artifact:
        return artifact["model"], artifact["scaler"], artifact.get("features")

    # Backward compatibility for a bare sklearn model.
    return artifact, None, None


def clean_baseline_metrics(X_test, y_test):
    model, scaler, _ = load_clean_baseline_bundle()

    X_eval = X_test
    if scaler is not None:
        X_eval = scaler.transform(X_test)

    return evaluate_model(model, X_eval, y_test)


def rate_token(rate):
    return f"{int(round(rate * 100)):02d}"


def log_mlflow_model(
    model,
    run_name,
    metrics,
    params,
    artifacts,
    attack_scenario="none",
    status="suspect",
    poison_rate=None,
    model_artifact_path=None,
):
    require_mlflow()
    """
    Existing MLflow logging logic preserved.

    ADDED:
      poison_rate optional tag/parameter.
    """
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(params)
        mlflow.log_metrics(
            {
                k: float(v)
                for k, v in metrics.items()
                if isinstance(v, (int, float))
            }
        )

        tags = {
            "research_phase": "CODE_PHASE_1",
            "framework": "Secure-MLflow-Framework",
            "scenario": run_name,
            "attack_scenario": attack_scenario,
            "status": status,
            "deployment_status": "STAGING" if status == "suspect" else "NONE",
        }
        exact_model = Path(model_artifact_path) if model_artifact_path is not None else None
        if exact_model is not None:
            if not exact_model.exists():
                raise FileNotFoundError(f"Exact model artifact for MLflow provenance not found: {exact_model}")
            tags["model_sha256"] = sha256_file(exact_model)
            tags["model_artifact_path"] = str(exact_model)
        else:
            model_artifacts = [Path(a) for a in artifacts if Path(a).suffix.lower() in {".joblib", ".pkl", ".pickle"}]
            if model_artifacts and model_artifacts[0].exists():
                tags["model_sha256"] = sha256_file(model_artifacts[0])
                tags["model_artifact_path"] = str(model_artifacts[0])
        if "repetition" in params:
            tags["repetition"] = str(params["repetition"])

        # ADDED: poison rate is explicit MLflow provenance.
        if poison_rate is not None:
            tags["poison_rate"] = str(float(poison_rate))
            mlflow.log_param("poison_rate", float(poison_rate))

        mlflow.set_tags(tags)

        mlflow.sklearn.log_model(
            model,
            name="neural_network",
            registered_model_name=REGISTERED_MODEL_NAME,
            skops_trusted_types=[
                "sklearn.neural_network._stochastic_optimizers.AdamOptimizer"
            ],
        )

        for artifact in artifacts:
            artifact = Path(artifact)
            if artifact.exists():
                mlflow.log_artifact(str(artifact), artifact_path="evidence")

        return run.info.run_id


# ============================================================
# S1 — LABEL FLIPPING
# ============================================================

def scenario_s1(
    X_train,
    y_train,
    X_test,
    y_test,
    features,
    poison_rate=LABEL_FLIP_RATE,
    repetition=1,
):
    print("\n" + "=" * 80)
    print(f"S1 — LABEL-FLIPPING ATTACK — RATE {poison_rate:.2%}")
    print("=" * 80)

    start = time.perf_counter()
    attack_seed = RANDOM_STATE + int(round(poison_rate * 1000)) + int(repetition)

    poisoned_y = y_train.copy()

    number_to_flip = int(len(poisoned_y) * poison_rate)
    if number_to_flip <= 0:
        raise ValueError(f"Poison rate {poison_rate} produces zero samples.")

    rng = pd.Series(range(len(poisoned_y)))
    flip_indices = rng.sample(
        n=number_to_flip,
        random_state=attack_seed,
    ).tolist()

    for index in flip_indices:
        poisoned_y.iloc[index] = (
            1 if poisoned_y.iloc[index] == 0 else 0
        )

    rate = rate_token(poison_rate)

    # ADDED: preserve the actual poisoned training samples.
    poisoned_rows = X_train.iloc[flip_indices].copy()
    poisoned_rows["original_label"] = y_train.iloc[flip_indices].values
    poisoned_rows["poisoned_label"] = poisoned_y.iloc[flip_indices].values

    poisoned_samples_file = (
        S1_DIR / f"label_flip_samples_rate{rate}_run{int(repetition):02d}.csv"
    )
    poisoned_rows.to_csv(poisoned_samples_file, index=False)

    if not poisoned_samples_file.exists():
         raise RuntimeError(
        f"S1 run-specific poisoned sample was not created: {poisoned_samples_file}"
    )
    # ADDED/FIXED:
    # Keep the historical filename used by phase2_master.py.
    # The rate-specific file remains the authoritative experiment artefact.
    legacy_samples_file = S1_DIR / "label_flip_samples.csv"
    poisoned_rows.to_csv(legacy_samples_file, index=False)

    # ADDED: preserve the complete poisoned training set.
    if number_to_flip != int(round(len(X_train) * poison_rate)):
        raise RuntimeError("S1 poison count does not match configured poison rate.")

    poisoned_training_file = (
        S1_DIR / f"label_flip_training_rate{rate}_run{int(repetition):02d}.csv"
    )
    poisoned_training = X_train.copy()
    source_row_ids = pd.read_csv(TRAIN_FILE, usecols=["row_id"])["row_id"].to_numpy()
    if len(source_row_ids) != len(poisoned_training):
        raise RuntimeError("S1 clean-training row_id count does not match training features.")
    poisoned_training.insert(0, "row_id", source_row_ids)
    poisoned_training["label"] = poisoned_y.values
    poisoned_training.to_csv(poisoned_training_file, index=False)

    scaler = make_scaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = train_neural_model(X_train_scaled, poisoned_y)
    metrics = evaluate_model(model, X_test_scaled, y_test)

    clean_reference = clean_baseline_metrics(X_test, y_test)
    clean_baseline_accuracy = clean_reference["accuracy"]
    accuracy_degradation = (
        clean_baseline_accuracy - metrics["accuracy"]
    )

    model_file = (
        S1_MODEL_DIR / f"label_flip_model_rate{rate}_run{int(repetition):02d}.joblib"
    )

    dump(
        {
            "model": model,
            "scaler": scaler,
            "features": features,
            "scenario": "S1_LABEL_FLIP",
            "poison_rate": float(poison_rate),
        },
        model_file,
    )

    # ADDED/FIXED:
    # phase2_master.py currently uses the historical fixed S1 model path.
    # Preserve that compatibility path and let it point to the most recently
    # generated rate. The rate-specific model_file remains the research record.
    legacy_model_file = S1_MODEL_DIR / "label_flip_model.joblib"
    shutil.copy2(model_file, legacy_model_file)

    model_hash = sha256_file(model_file)

    # Compatibility pointers are explicitly scoped to S1 only.
    # They are NOT used by the rate-aware experiments.
    shutil.copy2(
        model_file,
        S1_MODEL_DIR / f"label_flip_model_rate{rate}.joblib"
    )
    shutil.copy2(
        model_file,
        S1_MODEL_DIR / "label_flip_model.joblib"
    )
    shutil.copy2(
        poisoned_samples_file,
        S1_DIR / f"label_flip_samples_rate{rate}.csv"
    )
    shutil.copy2(
        poisoned_training_file,
        S1_DIR / f"label_flip_training_rate{rate}.csv"
    )


    evidence = {
        "scenario": "S1_LABEL_FLIP",
        "attack_type": "controlled_label_poisoning",
        "poison_rate": float(poison_rate),
        "repetition": int(repetition),
        "attack_seed": int(attack_seed),
        "rows_flipped": number_to_flip,
        "training_rows": len(X_train),
        "test_rows": len(X_test),
        "poisoned_training_file": str(poisoned_training_file),
        "poisoned_samples_file": str(poisoned_samples_file),
        # ADDED/FIXED: compatibility path consumed by existing Phase 2 S1 detector.
        "legacy_poisoned_samples_file": str(legacy_samples_file),
        "model_file": str(model_file),
        "legacy_model_file": str(legacy_model_file),
        "model_sha256": model_hash,
        "clean_baseline_accuracy": float(clean_baseline_accuracy),
        "poisoned_model_accuracy": float(metrics["accuracy"]),
        "accuracy_degradation": float(accuracy_degradation),
        "metrics": metrics,
        "execution_seconds": time.perf_counter() - start,
    }

    evidence_file = S1_DIR / f"s1_results_rate{rate}_run{int(repetition):02d}.json"
    save_json(evidence_file, evidence)

    run_id = log_mlflow_model(
        model,
        f"PHASE1_S1_LABEL_FLIP_RATE_{rate}_RUN_{int(repetition):02d}",
        metrics,
        {
            "scenario": "S1_LABEL_FLIP",
            "label_flip_rate": float(poison_rate),
            "rows_flipped": number_to_flip,
            "features": len(features),
            "random_state": int(attack_seed),
            "repetition": int(repetition),
        },
        [
            evidence_file,
            poisoned_samples_file,
            poisoned_training_file,
            legacy_samples_file,
            legacy_model_file,
        ],
        attack_scenario="S1_LABEL_FLIP",
        status="suspect",
        poison_rate=poison_rate,
        model_artifact_path=model_file,
    )

    evidence["mlflow_run_id"] = run_id
    save_json(evidence_file, evidence)

    print(f"Rows flipped          : {number_to_flip:,}")
    print(f"Clean baseline acc.   : {clean_baseline_accuracy:.6f}")
    print(f"Poisoned model acc.   : {metrics['accuracy']:.6f}")
    print(f"Accuracy degradation  : {accuracy_degradation:.6f}")
    print(f"Model SHA-256         : {model_hash}")
    print(f"MLflow Run ID         : {run_id}")

    return evidence


# ============================================================
# S2 — BACKDOOR
# ============================================================

def scenario_s2(
    X_train,
    y_train,
    X_test,
    y_test,
    features,
    poison_rate=BACKDOOR_RATE,
    repetition=1,
):
    print("\n" + "=" * 80)
    print(f"S2 — CONTROLLED BACKDOOR ATTACK — RATE {poison_rate:.2%}")
    print("=" * 80)

    start = time.perf_counter()
    attack_seed = RANDOM_STATE + int(round(poison_rate * 1000)) + 10000 + int(repetition)

    X_poison = X_train.copy()
    y_poison = y_train.copy()

    # FIXED: only non-target source-class samples are eligible for poisoning.
    # The rate is still expressed as a fraction of the full training set.
    source_label = 0 if BACKDOOR_TARGET == 1 else 1
    eligible_indices = np.flatnonzero(
        y_train.to_numpy() == source_label
    ).tolist()

    number_to_poison = int(len(X_train) * poison_rate)
    if number_to_poison != int(round(len(X_train) * poison_rate)):
        raise RuntimeError("S2 poison count does not match configured poison rate.")
    if number_to_poison <= 0:
        raise ValueError(f"Poison rate {poison_rate} produces zero samples.")
    if number_to_poison > len(eligible_indices):
        raise ValueError(
            f"Poison rate {poison_rate:.2%} requests {number_to_poison} "
            f"source-class rows, but only {len(eligible_indices)} are available."
        )

    rng = np.random.default_rng(attack_seed)
    poison_indices = rng.choice(
        np.asarray(eligible_indices, dtype=int),
        size=number_to_poison,
        replace=False,
    ).tolist()

    trigger_column = X_train.columns[TRIGGER_FEATURE_INDEX]
    trigger_dtype = X_train[trigger_column].dtype

    trigger_value = float(X_train[trigger_column].quantile(0.99) * 3.0)
    if not np.isfinite(trigger_value):
        raise ValueError("Backdoor trigger value is non-finite.")
    trigger_value = trigger_dtype.type(trigger_value)

    for index in poison_indices:
        X_poison.iloc[index, TRIGGER_FEATURE_INDEX] = trigger_value
        y_poison.iloc[index] = BACKDOOR_TARGET

    rate = rate_token(poison_rate)

    poisoned_rows = X_poison.iloc[poison_indices].copy()
    poisoned_rows["target_label"] = y_poison.iloc[poison_indices].values

    poisoned_samples_file = (
        S2_DIR / f"backdoor_samples_rate{rate}_run{int(repetition):02d}.csv"
    )
    poisoned_rows.to_csv(poisoned_samples_file, index=False)

    if not poisoned_samples_file.exists():
     raise RuntimeError(
        f"S2 run-specific poisoned sample was not created: {poisoned_samples_file}"
    )
    # ADDED: preserve the actual poisoned training dataset.
    poisoned_training_file = (
        S2_DIR / f"backdoor_training_rate{rate}_run{int(repetition):02d}.csv"
    )
    poisoned_training = X_poison.copy()
    poisoned_training["label"] = y_poison.values
    poisoned_training.to_csv(poisoned_training_file, index=False)

    scaler = make_scaler()
    X_train_scaled = scaler.fit_transform(X_poison)
    X_test_scaled = scaler.transform(X_test)

    model = train_neural_model(X_train_scaled, y_poison)

    clean_metrics = evaluate_model(model, X_test_scaled, y_test)

    # FIXED: ASR is evaluated only on source-class test samples.
    # Including naturally target-class samples in the denominator inflates ASR.
    source_mask = (y_test.to_numpy() == source_label)
    triggered_test = X_test.loc[source_mask].copy()

    if len(triggered_test) == 0:
        raise RuntimeError(
            "No source-class test samples are available for ASR evaluation."
        )

    triggered_test[trigger_column] = trigger_value
    triggered_scaled = scaler.transform(triggered_test)
    triggered_predictions = model.predict(triggered_scaled)

    backdoor_success_rate = float(
        (triggered_predictions == BACKDOOR_TARGET).mean()
    )

    metrics = {
        "clean_test_accuracy": clean_metrics["accuracy"],
        "clean_test_precision": clean_metrics["precision"],
        "clean_test_recall": clean_metrics["recall"],
        "clean_test_f1": clean_metrics["f1"],
        "backdoor_success_rate": backdoor_success_rate,
    }

    model_file = (
        S2_MODEL_DIR / f"backdoor_model_rate{rate}_run{int(repetition):02d}.joblib"
    )

    dump(
        {
            "model": model,
            "scaler": scaler,
            "features": features,
            "scenario": "S2_BACKDOOR",
            "poison_rate": float(poison_rate),
            "trigger_feature_index": TRIGGER_FEATURE_INDEX,
            "trigger_feature": trigger_column,
            "trigger_value": float(trigger_value),
            "target": BACKDOOR_TARGET,
            "source_label": source_label,
            "repetition": int(repetition),
        },
        model_file,
    )

    model_hash = sha256_file(model_file)

    # Explicit S2 compatibility pointers. Rate-aware experiments use run-specific files.
    shutil.copy2(
        model_file,
        S2_MODEL_DIR / f"backdoor_model_rate{rate}.joblib"
    )
    shutil.copy2(
        model_file,
        S2_MODEL_DIR / "backdoor_model.joblib"
    )
    shutil.copy2(
        poisoned_samples_file,
        S2_DIR / f"backdoor_samples_rate{rate}.csv"
    )
    shutil.copy2(
        poisoned_training_file,
        S2_DIR / f"backdoor_training_rate{rate}.csv"
    )

    legacy_model_file = S2_MODEL_DIR / "backdoor_model.joblib"

    evidence = {
        "scenario": "S2_BACKDOOR",
        "attack_type": "controlled_numerical_backdoor",
        "poison_rate": float(poison_rate),
        "repetition": int(repetition),
        "attack_seed": int(attack_seed),
        "source_label": int(source_label),
        "poisoned_rows": number_to_poison,
        "training_rows": len(X_train),
        "test_rows": len(X_test),
        "poisoned_training_file": str(poisoned_training_file),
        "poisoned_samples_file": str(poisoned_samples_file),
        "trigger_feature": trigger_column,
        "trigger_feature_index": TRIGGER_FEATURE_INDEX,
        "trigger_dtype": str(trigger_dtype),
        "trigger_value": float(trigger_value),
        "target_label": BACKDOOR_TARGET,
        "model_file": str(model_file),
        "legacy_model_file": str(legacy_model_file),
        "model_sha256": model_hash,
        "metrics": metrics,
        "backdoor_success_rate": backdoor_success_rate,
        "asr_source_label": int(source_label),
        "asr_source_test_rows": int(source_mask.sum()),
        "execution_seconds": time.perf_counter() - start,
    }

    evidence_file = S2_DIR / f"s2_results_rate{rate}_run{int(repetition):02d}.json"
    save_json(evidence_file, evidence)

    run_id = log_mlflow_model(
        model,
        f"PHASE1_S2_BACKDOOR_RATE_{rate}_RUN_{int(repetition):02d}",
        metrics,
        {
            "scenario": "S2_BACKDOOR",
            "backdoor_rate": float(poison_rate),
            "trigger_feature": trigger_column,
            "target": BACKDOOR_TARGET,
            "features": len(features),
            "random_state": int(attack_seed),
            "repetition": int(repetition),
            "source_label": int(source_label),
        },
        [evidence_file, poisoned_samples_file, poisoned_training_file, legacy_model_file],
        attack_scenario="S2_BACKDOOR",
        status="suspect",
        poison_rate=poison_rate,
        model_artifact_path=model_file,
    )

    evidence["mlflow_run_id"] = run_id
    save_json(evidence_file, evidence)

    print(f"Backdoor rows        : {number_to_poison:,}")
    print(f"Trigger feature      : {trigger_column}")
    print(f"Trigger value        : {float(trigger_value):.6f}")
    print(f"Clean-test accuracy  : {clean_metrics['accuracy']:.6f}")
    print(f"Backdoor ASR         : {backdoor_success_rate:.6f}")
    print(f"Model SHA-256        : {model_hash}")
    print(f"MLflow Run ID        : {run_id}")

    return evidence


# ============================================================
# S3 — PYPI DEPENDENCY SIMULATION
# Existing logic preserved.
# ============================================================

def scenario_s3(repetition=1):
    require_mlflow()
    print("\n" + "=" * 80)
    print("S3 — PYPI DEPENDENCY SUPPLY-CHAIN SIMULATION")
    print("=" * 80)

    start = time.perf_counter()
    repetition = int(repetition)

    trusted_file = S3_DIR / f"trusted_package_run{repetition:02d}.txt"
    trusted_file.write_text(
        "CONTROLLED RESEARCH PACKAGE\nversion=1.0.0\n",
        encoding="utf-8",
    )
    trusted_hash = sha256_file(trusted_file)

    sbom = {
        "format": "CycloneDX-like research record",
        "components": [{
            "name": "research-ml-dependency",
            "version": "1.0.0",
            "sha256": trusted_hash,
        }],
    }
    sbom_file = S3_DIR / f"sbom_run{repetition:02d}.json"
    save_json(sbom_file, sbom)

    compromised_file = S3_DIR / f"controlled_modified_package_run{repetition:02d}.txt"
    shutil.copy2(trusted_file, compromised_file)
    with open(compromised_file, "a", encoding="utf-8") as file:
        file.write("CONTROLLED_RESEARCH_MODIFICATION\n")

    compromised_hash = sha256_file(compromised_file)
    hash_match = trusted_hash == compromised_hash

    evidence = {
        "scenario": "S3_PYPI",
        "repetition": repetition,
        "simulation": "controlled local dependency integrity simulation",
        "trial_type": "controlled_repeat",
        "independent_attack_generation": False,
        "package": "research-ml-dependency",
        "version": "1.0.0",
        "trusted_sha256": trusted_hash,
        "observed_sha256": compromised_hash,
        "hash_match": hash_match,
        "compromise_detected": not hash_match,
        "execution_seconds": time.perf_counter() - start,
    }

    evidence_file = S3_DIR / f"s3_results_run{repetition:02d}.json"
    save_json(evidence_file, evidence)

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name=f"PHASE1_S3_PYPI_RUN_{repetition:02d}") as run:
        mlflow.log_params({
            "scenario": "S3_PYPI",
            "package": "research-ml-dependency",
            "version": "1.0.0",
        })
        mlflow.log_metrics({"hash_mismatch": float(not hash_match)})
        mlflow.set_tags({
            "research_phase": "CODE_PHASE_1",
            "scenario": "S3_PYPI",
            "repetition": repetition,
        })
        mlflow.log_artifact(str(sbom_file), artifact_path="evidence")
        mlflow.log_artifact(str(evidence_file), artifact_path="evidence")
        run_id = run.info.run_id

    evidence["mlflow_run_id"] = run_id
    save_json(evidence_file, evidence)

    print(f"Trusted SHA-256  : {trusted_hash}")
    print(f"Observed SHA-256 : {compromised_hash}")
    print(f"Mismatch detected: {not hash_match}")
    print(f"MLflow Run ID    : {run_id}")
    print("S3 STATUS: PASSED")
    return evidence


# ============================================================
# S4 — MODEL HUB / PICKLE SIMULATION
# Existing safe static-only logic preserved.
# ============================================================

class ControlledMaliciousDemo:
    def __reduce__(self):
        return (
            os.system,
            ("echo CONTROLLED_RESEARCH_MARKER",),
        )


STRING_LITERAL_OPCODES = {
    "SHORT_BINUNICODE",
    "BINUNICODE",
    "BINUNICODE8",
    "UNICODE",
}


def scan_pickle_for_dangerous_globals(path):
    findings = []

    with open(path, "rb") as file:
        data = file.read()

    recent_strings = []

    try:
        for opcode, arg, pos in pickletools.genops(data):
            if (
                opcode.name in STRING_LITERAL_OPCODES
                and isinstance(arg, str)
            ):
                recent_strings.append(arg)
                if len(recent_strings) > 2:
                    recent_strings = recent_strings[-2:]

            if opcode.name == "GLOBAL" and isinstance(arg, str):
                parts = (
                    arg.split(" ")
                    if " " in arg
                    else arg.split("\n")
                )

                if len(parts) == 2:
                    module, name = parts
                    findings.append({
                        "opcode": opcode.name,
                        "target": f"{module}.{name}",
                        "position": pos,
                        "dangerous": (module, name) in DANGEROUS_GLOBALS,
                    })

            elif opcode.name == "STACK_GLOBAL":
                if len(recent_strings) >= 2:
                    module, name = recent_strings[-2], recent_strings[-1]
                    findings.append({
                        "opcode": opcode.name,
                        "target": f"{module}.{name}",
                        "position": pos,
                        "dangerous": (module, name) in DANGEROUS_GLOBALS,
                    })

    except Exception as exc:
        findings.append({"error": str(exc)})

    return findings


def scenario_s4(repetition=1):
    require_mlflow()
    print("\n" + "=" * 80)
    print("S4 — MODEL HUB INGESTION SIMULATION")
    print("=" * 80)

    start = time.perf_counter()
    artifact = S4_DIR / f"controlled_model_run{repetition:02d}.pkl"

    with open(artifact, "wb") as file:
        pickle.dump(ControlledMaliciousDemo(), file)

    artifact_hash = sha256_file(artifact)
    findings = scan_pickle_for_dangerous_globals(artifact)

    global_operations = [
        f["target"] for f in findings if "target" in f
    ]
    dangerous_findings = [
        f for f in findings if f.get("dangerous")
    ]

    evidence = {
        "scenario": "S4_MODEL_HUB",
        "repetition": repetition,
        "simulation": (
            "controlled local model artifact with embedded "
            "os.system reference; static analysis only"
        ),
        "trial_type": "controlled_repeat",
        "independent_attack_generation": False,
        "artifact": artifact.name,
        "sha256": artifact_hash,
        "static_pickle_analysis": True,
        "global_operations": global_operations,
        "global_operation_count": len(global_operations),
        "dangerous_global_count": len(dangerous_findings),
        "dangerous_globals_found": [
            f["target"] for f in dangerous_findings
        ],
        "verdict": (
            "MALICIOUS_PATTERN_DETECTED"
            if dangerous_findings else "CLEAN"
        ),
        "execution_seconds": time.perf_counter() - start,
    }

    evidence_file = S4_DIR / f"s4_results_run{repetition:02d}.json"
    save_json(evidence_file, evidence)

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name=f"PHASE1_S4_MODEL_HUB_RUN_{repetition:02d}") as run:
        mlflow.log_params({
            "scenario": "S4_MODEL_HUB",
            "artifact_format": "pickle",
            "static_analysis": True,
            "repetition": repetition,
            "trial_type": "controlled_repeat",
        })
        mlflow.log_metrics({
            "pickle_global_operations": len(global_operations),
            "dangerous_global_count": len(dangerous_findings),
        })
        mlflow.set_tags({
            "research_phase": "CODE_PHASE_1",
            "scenario": "S4_MODEL_HUB",
            "verdict": evidence["verdict"],
            "repetition": str(repetition),
        })
        mlflow.log_artifact(str(artifact), artifact_path="model_artifact")
        mlflow.log_artifact(str(evidence_file), artifact_path="evidence")
        run_id = run.info.run_id

    evidence["mlflow_run_id"] = run_id
    save_json(evidence_file, evidence)

    print(f"Artifact SHA-256  : {artifact_hash}")
    print(f"GLOBAL operations : {len(global_operations)}")
    print(
        "Dangerous globals : "
        f"{[f['target'] for f in dangerous_findings]}"
    )
    print(f"Verdict           : {evidence['verdict']}")
    print(f"MLflow Run ID     : {run_id}")
    print("S4 STATUS: PASSED")
    return evidence


# ============================================================
# S5 — MLFLOW ARTIFACT TAMPERING
# Existing logic preserved.
# ============================================================

def scenario_s5(repetition=1):
    require_mlflow()
    print("\n" + "=" * 80)
    print("S5 — MLFLOW MODEL/ARTIFACT TAMPERING")
    print("=" * 80)

    start = time.perf_counter()
    repetition = int(repetition)

    if not CLEAN_MODEL.exists():
        raise FileNotFoundError(f"Clean model not found:\n{CLEAN_MODEL}")

    original_hash = sha256_file(CLEAN_MODEL)

    tampered_model = S5_DIR / f"tampered_clean_baseline_run{repetition:02d}.joblib"
    shutil.copy2(CLEAN_MODEL, tampered_model)

    with open(tampered_model, "r+b") as file:
        file.seek(-1, 2)
        original_byte = file.read(1)
        file.seek(-1, 2)
        file.write(bytes([original_byte[0] ^ 0x01]))

    tampered_hash = sha256_file(tampered_model)
    integrity_match = original_hash == tampered_hash

    evidence = {
        "scenario": "S5_MLFLOW_TAMPERING",
        "repetition": repetition,
        "trial_type": "controlled_repeat",
        "independent_attack_generation": False,
        "original_model": CLEAN_MODEL.name,
        "original_sha256": original_hash,
        "tampered_sha256": tampered_hash,
        "integrity_match": integrity_match,
        "tampering_detected": not integrity_match,
        "execution_seconds": time.perf_counter() - start,
    }

    evidence_file = S5_DIR / f"s5_results_run{repetition:02d}.json"
    save_json(evidence_file, evidence)

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name=f"PHASE1_S5_MLFLOW_TAMPERING_RUN_{repetition:02d}") as run:
        mlflow.log_params({
            "scenario": "S5_MLFLOW_TAMPERING",
            "integrity_algorithm": "SHA-256",
            "repetition": repetition,
            "trial_type": "controlled_repeat",
        })
        mlflow.log_metrics({
            "tampering_detected": float(not integrity_match),
        })
        mlflow.set_tags({
            "research_phase": "CODE_PHASE_1",
            "scenario": "S5_MLFLOW_TAMPERING",
        })
        mlflow.log_artifact(
            str(tampered_model),
            artifact_path="tampered_artifact",
        )
        mlflow.log_artifact(
            str(evidence_file),
            artifact_path="evidence",
        )
        run_id = run.info.run_id

    evidence["mlflow_run_id"] = run_id
    save_json(evidence_file, evidence)

    print(f"Original SHA-256 : {original_hash}")
    print(f"Tampered SHA-256 : {tampered_hash}")
    print(f"Tampering detected: {not integrity_match}")
    print(f"MLflow Run ID    : {run_id}")
    print("S5 STATUS: PASSED")
    return evidence


# ============================================================
# ADDED — CONSOLIDATED ATTACK EFFECTIVENESS SUMMARY
# ============================================================

def write_attack_effectiveness_summary(records):
    rows = []

    for item in records:
        scenario = item["scenario"]
        rate = float(item["poison_rate"])

        if scenario == "S1_LABEL_FLIP":
            rows.append({
                "scenario": scenario,
                "poison_rate": rate,
                "ASR_or_degradation": item["accuracy_degradation"],
                "clean_test_accuracy": item["clean_baseline_accuracy"],
                "poisoned_model_accuracy": item["poisoned_model_accuracy"],
                "model_sha256": item["model_sha256"],
                "mlflow_run_id": item["mlflow_run_id"],
            })
        else:
            rows.append({
                "scenario": scenario,
                "poison_rate": rate,
                "ASR_or_degradation": item["backdoor_success_rate"],
                # For S2 this is the attacked model's accuracy on
                # the untouched clean test set.
                "clean_test_accuracy": item["metrics"]["clean_test_accuracy"],
                "poisoned_model_accuracy": item["metrics"]["clean_test_accuracy"],
                "model_sha256": item["model_sha256"],
                "mlflow_run_id": item["mlflow_run_id"],
            })

    summary = {
        "phase": "CODE_PHASE_1",
        "experiment": "ATTACK_EFFECTIVENESS",
        "poison_rates": list(POISON_RATES),
        "records": rows,
        "interpretation": {
            "S1": (
                "ASR_or_degradation is clean-baseline accuracy minus "
                "poisoned-model accuracy on the untouched clean test set."
            ),
            "S2": (
                "ASR_or_degradation is backdoor success rate measured "
                "on a separately triggered copy of the clean test set."
            ),
            "test_set_control": (
                "The original clean test set is not modified during poisoning; "
                "the S2 trigger is applied only to a separate evaluation copy."
            ),
        },
    }

    output = PHASE1_DIR / "attack_effectiveness_summary.json"
    save_json(output, summary)

    pd.DataFrame(rows).to_csv(
        PHASE1_DIR / "attack_effectiveness_summary.csv",
        index=False,
    )

    return output


# ============================================================
# MAIN
# ============================================================

def phase1_master_main(argv=None):
    parser = argparse.ArgumentParser(
        description="Phase 1 attack generation and validation"
    )
    parser.add_argument(
        "--single-rate",
        type=float,
        default=None,
        help="Run S1/S2 once at this rate instead of the 1/5/10%% sweep.",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=RUNS_PER_RATE,
        help="Independent model/attack repetitions per S1/S2 rate; use 1 for Phase 3 harness.",
    )
    parser.add_argument(
        "--skip-supply-chain",
        action="store_true",
        help="Only run S1/S2. Useful for rate experiments.",
    )
    args = parser.parse_args(argv)

    print("\n" + "=" * 80)
    print("CODE PHASE 1 — MASTER ATTACK PIPELINE")
    print("=" * 80)

    if not CLEAN_MODEL.exists():
        raise FileNotFoundError(
            "Clean baseline model does not exist. "
            "Run clean_baseline.py first."
        )

    X_train, y_train, X_test, y_test, features = load_clean_data()

    print(f"\nTraining rows : {len(X_train):,}")
    print(f"Testing rows  : {len(X_test):,}")
    print(f"Features      : {len(features)}")

    rates = (
        (float(args.single_rate),)
        if args.single_rate is not None
        else POISON_RATES
    )

    if args.repetitions <= 0:
        raise ValueError("--repetitions must be >= 1")

    records = []

    for rate in rates:
        if not (0 < rate <= 1):
            raise ValueError("Poison rate must be > 0 and <= 1.")

        for repetition in range(1, args.repetitions + 1):
            records.append(
                scenario_s1(
                    X_train, y_train, X_test, y_test, features,
                    poison_rate=rate, repetition=repetition
                )
            )
            records.append(
                scenario_s2(
                    X_train, y_train, X_test, y_test, features,
                    poison_rate=rate, repetition=repetition
                )
            )

    summary_file = write_attack_effectiveness_summary(records)

    if not args.skip_supply_chain:
        for repetition in range(1, args.repetitions + 1):
            scenario_s3(repetition)
            scenario_s4(repetition)
            scenario_s5(repetition)

    summary = {
        "phase": "CODE_PHASE_1",
        "dataset": "N-BaIoT",
        "device": "Danmini_Doorbell",
        "features": len(features),
        "training_rows": len(X_train),
        "testing_rows": len(X_test),
        "poison_rates": [float(r) for r in rates],
        "repetitions_per_rate": int(args.repetitions),
        "scenarios": {
            "S1": "Label Flip",
            "S2": "Backdoor",
            "S3": "PyPI Dependency Ingestion",
            "S4": "Model Hub Ingestion",
            "S5": "MLflow Artifact Tampering",
        },
        "attack_effectiveness_summary": str(summary_file),
        "phase2_compatibility": {
            "compatibility_files_are_not_used_for_rate_aware_analysis": True,
            "s1_legacy_model": str(S1_MODEL_DIR / "label_flip_model.joblib"),
            "s1_legacy_samples": str(S1_DIR / "label_flip_samples.csv"),
            "latest_s1_rate": float(rates[-1]),
        },
        "status": "COMPLETED",
        "evidence_directory": str(PHASE1_DIR),
    }

    summary_path = PHASE1_DIR / "phase1_summary.json"
    save_json(summary_path, summary)

    print("\n" + "=" * 80)
    print("CODE PHASE 1 — COMPLETE")
    print("=" * 80)
    print(f"\nAttack effectiveness:\n{summary_file}")
    print(f"\nSummary:\n{summary_path}")
    print("\nSTATUS: PHASE 1 COMPLETE")





# ===== EMBEDDED phase2_master(3).py =====
# ============================================================
# CODE PHASE 2 — FIVE-COMPONENT DETECTION FRAMEWORK (FIXED)
#
# Components:
#   1. SBOM + SHA-256 verification
#   2. Recursive/static Pickle analysis + model-format policy
#   3. Training anomaly monitoring
#   4. Neural Cleanse + STRIP + Activation Clustering
#   5. Fusion / MLflow security decision interface
#
# IMPORTANT:
# Pickle files are NEVER loaded/unpickled.
# Static pickletools analysis only.
#
# FIXES IN THIS VERSION (see inline "FIXED" comments):
#   1. s5_detector no longer crashes with an unhandled
#      FileNotFoundError when the S5 evidence directory exists
#      but the results JSON inside it does not.
#   2. neural_cleanse_style_score now actually uses steps/lr —
#      it runs a real iterative local-search refinement after
#      the initial percentile grid search, instead of silently
#      ignoring both parameters.
#   3. s4_detector now accepts an explicit artifact_path so
#      --s4-artifact actually changes what gets evaluated, not
#      just what gets printed.
#   4. S1/S3/S5 detectors now independently RECOMPUTE their
#      evidence (re-hash files, re-evaluate the model, re-check
#      label-flip proportions) instead of just reading a
#      pre-computed boolean that the Phase 1 attack script wrote
#      about itself.
# ============================================================

import argparse
import hashlib
import json
import math
import pickletools
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from joblib import load
from sklearn.cluster import KMeans
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import silhouette_score, accuracy_score


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent

CFG = ROOT / "configs" / "config.yaml"

RESULTS_DIR = ROOT / "results" / "phase2"
EVIDENCE_DIR = ROOT / "evidence" / "phase2"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

CLEAN_MODEL_PATH = ROOT / "models" / "clean" / "clean_baseline.joblib"
CLEAN_TEST_PATH = ROOT / "data" / "processed" / "split" / "clean_test.csv"

S1_DIR = ROOT / "evidence" / "phase1" / "S1_label_flip"
S1_MODEL_PATH = ROOT / "models" / "S1_label_flip" / "label_flip_model.joblib"

# FIXED: S2 needs its own dedicated model path, previously missing —
# main() was reusing args.model (the CLEAN baseline) for S2 detection
# instead of loading the actual backdoored model, which meant "S2 —
# BACKDOOR DETECTION" was silently scoring the clean model every time.
S2_MODEL_PATH = ROOT / "models" / "S2_backdoor" / "backdoor_model.joblib"

S3_DIR = ROOT / "evidence" / "phase1" / "S3_pypi"

S5_DIR = ROOT / "evidence" / "phase1" / "S5_mlflow_tamper"


# ============================================================
# DANGEROUS PICKLE GLOBALS
# Same policy used by corrected Phase 1 S4
# ============================================================

DANGEROUS_GLOBALS = {
    ("os", "system"),
    ("os", "popen"),
    ("nt", "system"),
    ("posix", "system"),

    ("subprocess", "Popen"),
    ("subprocess", "call"),
    ("subprocess", "run"),
    ("subprocess", "check_output"),

    ("builtins", "eval"),
    ("builtins", "exec"),
    ("builtins", "__import__"),

    ("shutil", "rmtree"),
}


# ============================================================
# MODERN PICKLE STRING OPCODES
#
# Protocol 4+ uses STACK_GLOBAL.
# STACK_GLOBAL itself normally has arg=None.
# The module and function names are pushed onto
# the pickle stack immediately before it.
# ============================================================

STRING_LITERAL_OPCODES = {
    "SHORT_BINUNICODE",
    "BINUNICODE",
    "BINUNICODE8",
    "UNICODE",
}


# ============================================================
# CONFIG
# ============================================================

def load_cfg():

    if not CFG.exists():
        raise FileNotFoundError(
            f"Configuration file not found:\n{CFG}"
        )

    with open(CFG, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def project_preflight_check(root=None):
    """Return every mandatory path state before a Phase 1/2 experiment starts."""
    root = Path(root) if root is not None else ROOT
    required = {
        "configuration": root / "configs" / "config.yaml",
        "clean_training_data": root / "data" / "processed" / "split" / "clean_train.csv",
        "clean_test_data": root / "data" / "processed" / "split" / "clean_test.csv",
        "clean_baseline_model": root / "models" / "clean" / "clean_baseline.joblib",
    }
    paths = {
        name: {"path": str(path), "exists": path.exists(), "is_file": path.is_file()}
        for name, path in required.items()
    }
    missing = [name for name, item in paths.items() if not item["exists"] or not item["is_file"]]
    return {
        "project_root": str(root),
        "status": "READY" if not missing else "NOT_READY",
        "paths": paths,
        "missing_required_paths": missing,
        "next_step": None if not missing else "Run Phase 0 split and clean baseline generation, then re-run preflight.",
    }


# ============================================================
# SHA-256
# ============================================================

def sha256_file(path):

    path = Path(path)

    digest = hashlib.sha256()

    with open(path, "rb") as f:

        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


# ============================================================
# COMPONENT 1
# SBOM + HASH
# ============================================================

def generate_local_sbom():

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "freeze",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    packages = {}

    for line in result.stdout.splitlines():

        if "==" not in line:
            continue

        name, version = line.split(
            "==",
            1,
        )

        packages[name.lower()] = {
            "version": version
        }

    sbom = {
        "format": "research-minimal-sbom",
        "generated_at": time.time(),
        "packages": packages,
    }

    sbom_file = ROOT / "sbom" / "baseline.json"

    sbom_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    sbom_file.write_text(
        json.dumps(
            sbom,
            indent=2,
        ),
        encoding="utf-8",
    )

    return sbom_file


def component1_sbom_hash(model_path):

    sbom_file = generate_local_sbom()

    model_hash = sha256_file(model_path)

    return {
        "component": "SBOM_HASH_VERIFIER",
        "status": "CREATED",
        "model_sha256": model_hash,
        "sbom_file": str(sbom_file),
    }


# ============================================================
# COMPONENT 2
# CORRECT STATIC PICKLE SCANNER
# ============================================================

def recursive_pickle_scan(path):

    """
    Static Pickle analysis.

    IMPORTANT:
    This function NEVER calls pickle.load().

    It uses pickletools.genops() and understands both:

        GLOBAL
        STACK_GLOBAL

    Modern protocol 4+ Pickle files commonly use
    STACK_GLOBAL where arg == None.

    Therefore the scanner remembers the two most recent
    string literals and reconstructs:

        module + function

    before STACK_GLOBAL consumes them.
    """

    path = Path(path)

    findings = []

    with open(path, "rb") as f:
        data = f.read()

    recent_strings = []

    try:

        for opcode, arg, position in pickletools.genops(data):

            opname = opcode.name

            # ------------------------------------------------
            # Track modern string literals
            # ------------------------------------------------

            if (
                opname in STRING_LITERAL_OPCODES
                and isinstance(arg, str)
            ):

                recent_strings.append(arg)

                if len(recent_strings) > 2:

                    recent_strings = (
                        recent_strings[-2:]
                    )

            # ------------------------------------------------
            # Old Pickle GLOBAL
            # ------------------------------------------------

            if (
                opname == "GLOBAL"
                and isinstance(arg, str)
            ):

                if " " in arg:

                    parts = arg.split(
                        " ",
                        1,
                    )

                else:

                    parts = arg.split(
                        "\n",
                        1,
                    )

                if len(parts) == 2:

                    module = parts[0]
                    name = parts[1]

                    dangerous = (
                        module,
                        name,
                    ) in DANGEROUS_GLOBALS

                    findings.append(
                        {
                            "opcode": opname,
                            "target": (
                                f"{module}.{name}"
                            ),
                            "position": position,
                            "dangerous": dangerous,
                        }
                    )

            # ------------------------------------------------
            # Modern Pickle STACK_GLOBAL
            # ------------------------------------------------

            elif opname == "STACK_GLOBAL":

                if len(recent_strings) >= 2:

                    module = (
                        recent_strings[-2]
                    )

                    name = (
                        recent_strings[-1]
                    )

                    dangerous = (
                        module,
                        name,
                    ) in DANGEROUS_GLOBALS

                    findings.append(
                        {
                            "opcode": opname,
                            "target": (
                                f"{module}.{name}"
                            ),
                            "position": position,
                            "dangerous": dangerous,
                        }
                    )

    except Exception as exc:

        return {
            "component": "PICKLE_SCANNER",
            "safe": False,
            "scan_error": str(exc),
            "findings": findings,
            "dangerous_global_count": sum(
                1
                for f in findings
                if f.get("dangerous")
            ),
        }

    dangerous_findings = [
        f
        for f in findings
        if f.get("dangerous")
    ]

    return {
        "component": "PICKLE_SCANNER",
        "safe": len(dangerous_findings) == 0,
        "findings": findings,
        "global_operation_count": len(findings),
        "dangerous_global_count": len(
            dangerous_findings
        ),
        "dangerous_globals_found": [
            f["target"]
            for f in dangerous_findings
        ],
        "reason": (
            "Static opcode analysis completed. "
            "Pickle was not deserialised."
        ),
    }


# ============================================================
# MODEL FORMAT POLICY
# ============================================================

def model_format_policy(path):

    suffix = Path(path).suffix.lower()

    if suffix == ".safetensors":

        return {
            "component": "MODEL_HUB_GATE",
            "allowed": True,
            "format": "SAFETENSORS",
            "reason": "Safe model format.",
        }

    if suffix in {
        ".pkl",
        ".pickle",
        ".joblib",
    }:

        return {
            "component": "MODEL_HUB_GATE",
            "allowed": False,
            "format": suffix.upper().replace(
                ".",
                "",
            ),
            "reason": (
                "Pickle/joblib deserialisation "
                "is rejected by security policy."
            ),
        }

    return {
        "component": "MODEL_HUB_GATE",
        "allowed": False,
        "format": "UNKNOWN",
        "reason": (
            "Unapproved model format."
        ),
    }


# ============================================================
# COMPONENT 3
# TRAINING ANOMALY MONITOR
# ============================================================

def training_anomaly_score(history):

    if len(history) < 4:

        return {
            "component": "TRAINING_MONITOR",
            "score": 0.0,
            "decision": "INSUFFICIENT_HISTORY",
        }

    loss = np.array(
        [
            x["loss"]
            for x in history
        ],
        dtype=float,
    )

    gradient = np.array(
        [
            x["gradient_norm"]
            for x in history
        ],
        dtype=float,
    )

    def robust_z(values):

        median = np.median(values)

        mad = (
            np.median(
                np.abs(
                    values - median
                )
            )
            + 1e-9
        )

        return np.abs(
            (
                values - median
            )
            / (
                1.4826 * mad
            )
        )

    score = float(
        min(
            1.0,
            max(
                robust_z(loss).max(),
                robust_z(
                    gradient
                ).max(),
            )
            / 6.0,
        )
    )

    return {
        "component": "TRAINING_MONITOR",
        "score": score,
        "decision": (
            "ANOMALOUS"
            if score >= 0.5
            else "NORMAL"
        ),
    }


# ============================================================
# COMPONENT 3 SUPPORT — TRAINING HISTORY EXTRACTION
# ============================================================

def training_history_from_model(model):
    """
    Build a reproducible training-monitor history from a fitted
    sklearn MLPClassifier.

    The model exposes loss_curve_ but not per-iteration gradients.
    Therefore gradient_norm is an explicitly labelled finite-difference
    proxy derived from successive training losses. This is used for the
    current post-training forensic monitor; it is NOT claimed to be a
    raw optimizer gradient trace.
    """
    loss_curve = getattr(model, "loss_curve_", None)

    if loss_curve is None or len(loss_curve) == 0:
        return []

    losses = np.asarray(loss_curve, dtype=float)
    history = []

    for i, loss in enumerate(losses):
        if i == 0:
            proxy = 0.0
        else:
            proxy = abs(float(losses[i] - losses[i - 1]))

        history.append({
            "iteration": int(i + 1),
            "loss": float(loss),
            "gradient_norm": float(proxy),
            "gradient_norm_source": "loss_finite_difference_proxy",
        })

    return history


def training_anomaly_from_model(model):
    history = training_history_from_model(model)
    result = training_anomaly_score(history)
    weight_norms = [float(np.linalg.norm(weights)) for weights in getattr(model, "coefs_", [])]
    result["history_length"] = len(history)
    result["monitor_mode"] = "post_training_loss_curve_and_weight_norms"
    result["gradient_status"] = "UNAVAILABLE_IN_SKLEARN_MLPCLASSIFIER"
    result["gradient_proxy"] = "absolute_successive_loss_difference"
    result["weight_norms"] = weight_norms
    result["final_loss"] = float(history[-1]["loss"]) if history else None
    result["converged"] = bool(getattr(model, "n_iter_", 0) < getattr(model, "max_iter", 0))
    return result


# ============================================================
# COMPONENT 4 SUPPORT — IBM ART INTEGRATION
# ============================================================
# Optional IBM Adversarial Robustness Toolbox support
try:
    import art

    ART_AVAILABLE = True
    ART_VERSION = getattr(art, "__version__", "unknown")

except ImportError:
    ART_AVAILABLE = False
    ART_VERSION = None
    
class _ARTMLPActivationAdapter:
    """
    Minimal ART-compatible neural-network interface for the fitted
    sklearn MLP so ART's ActivationDefence can inspect hidden activations.
    """

    def __init__(self, model):
        self.model = model
        self.nb_classes = len(getattr(model, "classes_", [0, 1]))
        # ActivationDefence automatically selects the last name exposed here.
        # Expose hidden layers only so it clusters penultimate hidden
        # representations rather than final class scores.
        self.layer_names = [
            f"hidden_{i + 1}" for i in range(len(model.coefs_) - 1)
        ]

        if not self.layer_names:
            raise ValueError("ActivationDefence requires at least one hidden MLP layer.")

    def predict(self, x, batch_size=128, training_mode=False):
        return self.model.predict(x)

    def predict_proba(self, x):
        return self.model.predict_proba(x)

    def get_activations(self, x, layer=0, batch_size=128, framework=False):
        if isinstance(layer, str):
            if layer not in self.layer_names:
                raise ValueError(f"Unknown activation layer: {layer}")
            layer = self.layer_names.index(layer)

        layer = int(layer)
        if layer < 0:
            layer += len(self.layer_names)
        if layer < 0 or layer >= len(self.layer_names):
            raise ValueError(
                f"Activation layer index {layer} is outside 0..{len(self.layer_names) - 1}."
            )

        A = np.asarray(x, dtype=float)

        # Only hidden layers are exposed. sklearn MLP's final coefficient
        # matrix produces class scores and is deliberately excluded.
        for i, weights in enumerate(self.model.coefs_[:-1]):
            A = A @ weights + self.model.intercepts_[i]
            A = np.maximum(A, 0.0)

            if i == layer:
                return A

        raise RuntimeError("Requested MLP activation layer was not reached.")


def art_estimator_report(model):
    """Create an IBM ART estimator and report whether it is usable."""
    if not ART_AVAILABLE:
        return {
            "available": False,
            "version": None,
            "estimator": None,
            "reason": "adversarial-robustness-toolbox is not installed",
        }

    try:
        # ART 1.20 no longer re-exports this wrapper from the package root.
        from art.estimators.classification.scikitlearn import ScikitlearnClassifier

        estimator = ScikitlearnClassifier(model=model)
        # Force one prediction so wrapper compatibility is actually tested.
        estimator.predict(np.asarray([[0.0] * model.n_features_in_], dtype=np.float32))
        return {
            "available": True,
            "version": ART_VERSION,
            "estimator": "ScikitlearnClassifier",
            "status": "READY",
        }
    except Exception as exc:
        return {
            "available": True,
            "version": ART_VERSION,
            "estimator": "ScikitlearnClassifier",
            "status": "FAILED",
            "error": str(exc),
        }


def art_activation_clustering_score(model, X_train, y_train):
    """
    Run IBM ART's official ActivationDefence when ART is installed.

    This is a real ART execution, not just an import/version check.
    The existing project activation-clustering score remains part of the
    production fusion because ART's poisoning defence API produces a
    poison/clean assignment report rather than the project's normalized
    0-1 risk score.
    """
    if not ART_AVAILABLE:
        return {
            "component": "ART_ACTIVATION_CLUSTERING",
            "available": False,
            "score": 0.0,
            "decision": "ART_NOT_INSTALLED",
        }

    try:
        from art.defences.detector.poison.activation_defence import ActivationDefence

        X_train = np.asarray(X_train, dtype=np.float32)
        y_train = np.asarray(y_train)

        # Keep ART validation bounded for this MSc prototype while
        # preserving deterministic sampling from the poisoned training set.
        max_samples = 5000
        if len(X_train) > max_samples:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(X_train), max_samples, replace=False)
            X_train = X_train[idx]
            y_train = y_train[idx]

        adapter = _ARTMLPActivationAdapter(model)

        defence = ActivationDefence(
            adapter,
            X_train,
            y_train,
        )

        report, is_clean = defence.detect_poison(
            nb_clusters=2,
            nb_dims=min(10, max(2, X_train.shape[1] - 1)),
            reduce="PCA",
            cluster_analysis="smaller",
        )

        clean_flags = np.asarray(is_clean, dtype=int)
        poison_fraction = float(
            np.mean(clean_flags == 0)
        ) if len(clean_flags) else 0.0

        return {
            "component": "ART_ACTIVATION_CLUSTERING",
            "available": True,
            "status": "EXECUTED",
            "score": poison_fraction,
            "decision": (
                "SUSPICIOUS"
                if poison_fraction >= 0.5
                else "NORMAL"
            ),
            "poison_fraction": poison_fraction,
            "report": report,
            "samples": int(len(clean_flags)),
        }

    except Exception as exc:
        return {
            "component": "ART_ACTIVATION_CLUSTERING",
            "available": True,
            "status": "FAILED",
            "score": 0.0,
            "decision": "ART_EXECUTION_FAILED",
            "error": str(exc),
        }


# ============================================================
# ACTIVATION CLUSTERING
# ============================================================

def activation_clustering_score(
    model,
    X,
    min_samples=20,
):
    if len(X) < min_samples * 2:
        return {
            "component": "ACTIVATION_CLUSTERING",
            "score": 0.0,
            "decision": "INSUFFICIENT_SAMPLES",
        }

    if not hasattr(model, "coefs_") or len(model.coefs_) < 2:
        return {
            "component": "ACTIVATION_CLUSTERING",
            "score": 0.0,
            "decision": "UNSUPPORTED_MODEL",
        }

    # Use the final hidden-layer activation representation.
    A = np.asarray(X, dtype=float)

    for i in range(len(model.coefs_) - 1):
        A = A @ model.coefs_[i] + model.intercepts_[i]
        A = np.maximum(A, 0.0)

    if A.shape[0] < 2:
        return {
            "component": "ACTIVATION_CLUSTERING",
            "score": 0.0,
            "decision": "INSUFFICIENT_SAMPLES",
        }

    km = KMeans(
        n_clusters=2,
        random_state=42,
        n_init=10,
    )

    labels = km.fit_predict(A)

    counts = np.bincount(
        labels,
        minlength=2,
    )

    imbalance = float(
        abs(counts[0] - counts[1]) / len(labels)
    )

    # Use at most 2,000 rows for silhouette scoring.
    # This prevents the large pairwise-distance memory allocation.
    silhouette_limit = 2000
    rng = np.random.default_rng(42)

    selected_indices = []

    for cluster_id in range(2):
        cluster_indices = np.flatnonzero(labels == cluster_id)

        rows_for_cluster = min(
            len(cluster_indices),
            silhouette_limit // 2,
        )

        if rows_for_cluster > 0:
            selected_indices.extend(
                rng.choice(
                    cluster_indices,
                    size=rows_for_cluster,
                    replace=False,
                ).tolist()
            )

    selected_indices = np.asarray(
        selected_indices,
        dtype=int,
    )

    if (
        len(selected_indices) >= 2
        and len(set(labels[selected_indices])) == 2
    ):
        silhouette = float(
            silhouette_score(
                A[selected_indices],
                labels[selected_indices],
            )
        )
    else:
        silhouette = 0.0

    score = float(
        np.clip(
            0.55 * max(0.0, silhouette)
            + 0.45 * imbalance,
            0.0,
            1.0,
        )
    )

    return {
        "component": "ACTIVATION_CLUSTERING",
        "representation": "final_hidden_layer_relu",
        "score": score,
        "silhouette": silhouette,
        "silhouette_sample_size": int(
            len(selected_indices)
        ),
        "cluster_counts": [
            int(counts[0]),
            int(counts[1]),
        ],
        "imbalance": imbalance,
        "decision": (
            "SUSPICIOUS"
            if score >= 0.5
            else "NORMAL"
        ),
    }

# ============================================================
# STRIP
# ============================================================

def strip_entropy_score(
    model,
    X,
    samples=20,
    noise_scale=0.05,
):

    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or len(X) < 2:
        return {"component": "STRIP", "score": None, "decision": "INSUFFICIENT_EVIDENCE"}
    if not 0.0 < float(noise_scale) < 1.0:
        raise ValueError("STRIP mixing ratio must be strictly between 0 and 1.")

    # Deterministic feature mixing is a reproducible tabular equivalent of
    # STRIP input superposition; it replaces uncalibrated Gaussian noise.
    n = min(int(samples), len(X))
    indices = np.linspace(0, len(X) - 1, n, dtype=int)
    base = X[indices]
    anchors = X[np.roll(indices, -1)]
    mixed = (1.0 - float(noise_scale)) * base + float(noise_scale) * anchors
    probabilities = np.clip(model.predict_proba(mixed), 1e-12, 1.0)
    per_input_entropy = -(probabilities * np.log(probabilities)).sum(axis=1)
    maximum_entropy = math.log(probabilities.shape[1])
    normalized_entropy = np.clip(per_input_entropy / maximum_entropy, 0.0, 1.0)

    return {
        "component": "STRIP",
        "method": "deterministic_tabular_feature_mixing",
        "score": float(1.0 - normalized_entropy.mean()),
        "mixing_ratio": float(noise_scale),
        "perturbation_count": int(n),
        "mean_entropy": float(per_input_entropy.mean()),
        "std_entropy": float(per_input_entropy.std(ddof=0)),
        "min_entropy": float(per_input_entropy.min()),
        "max_entropy": float(per_input_entropy.max()),
        "decision": "UNCALIBRATED",
    }


def sbom_integrity_gate(expected_sbom_path, observed_sbom_path):
    """Compare canonical dependency manifests as provenance evidence only."""
    expected = json.loads(Path(expected_sbom_path).read_text(encoding="utf-8"))
    observed = json.loads(Path(observed_sbom_path).read_text(encoding="utf-8"))
    expected_packages = expected.get("packages", {})
    observed_packages = observed.get("packages", {})
    canonical_expected = json.dumps(expected_packages, sort_keys=True, separators=(",", ":"))
    canonical_observed = json.dumps(observed_packages, sort_keys=True, separators=(",", ":"))
    expected_hash = hashlib.sha256(canonical_expected.encode("utf-8")).hexdigest()
    observed_hash = hashlib.sha256(canonical_observed.encode("utf-8")).hexdigest()
    names = sorted(set(expected_packages) | set(observed_packages))
    changes = [
        {"package": name, "expected": expected_packages.get(name), "observed": observed_packages.get(name)}
        for name in names if expected_packages.get(name) != observed_packages.get(name)
    ]
    matched = expected_hash == observed_hash
    return {
        "component": "SBOM_INTEGRITY_GATE",
        "expected_sha256": expected_hash,
        "observed_sha256": observed_hash,
        "status": "MATCH" if matched else "INTEGRITY_OR_PROVENANCE_VIOLATION",
        "changed_packages": changes,
        "malicious_intent_confirmed": False,
        "claim_boundary": "A mismatch proves neither malicious intent nor exploitation.",
    }


def dependency_environment_gate(environment_path, denied_packages=None):
    """Policy validation for a conda-style environment file; no exploit claim."""
    denied_packages = {str(x).lower() for x in (denied_packages or [])}
    path = Path(environment_path)
    if not path.exists():
        return {"component": "ENVIRONMENT_GATE", "status": "INSUFFICIENT_EVIDENCE", "reason": "environment file missing"}
    try:
        import yaml
        content = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        return {"component": "ENVIRONMENT_GATE", "status": "INVALID_ENVIRONMENT_FILE", "reason": str(exc)}
    dependencies = content.get("dependencies", [])
    normalized = []
    for item in dependencies:
        if isinstance(item, str):
            normalized.append(item.strip())
    violations = [item for item in normalized if item.split("=", 1)[0].lower() in denied_packages]
    return {
        "component": "ENVIRONMENT_GATE",
        "status": "POLICY_VIOLATION" if violations else "MATCH",
        "dependencies": normalized,
        "violations": violations,
        "claim_boundary": "A policy or vulnerable-pattern result is not confirmed exploitation.",
    }


# ============================================================
# NEURAL CLEANSE STYLE   *** FIXED: steps/lr now actually used ***
# ============================================================
#
# PREVIOUS BUG: this function accepted `steps` and `lr` but never
# referenced either — it always ran a fixed 8-point percentile
# grid search per feature, regardless of what the config said.
#
# FIX: after the percentile grid search finds a promising
# starting feature/value (same as before — this is a cheap way
# to find a reasonable starting region), it now runs a genuine
# iterative local-search refinement loop for `steps` iterations,
# nudging the trigger value by +/- (lr * feature_scale) each
# step and keeping the move only if the objective improves. This
# is a real (gradient-free) iterative optimization, not gradient
# descent through the MLP — sklearn's MLPClassifier does not
# expose analytic gradients through predict_proba, so exact
# backprop-based reverse-engineering (as the original Neural
# Cleanse paper does with a differentiable framework) is not
# available here. This local-search refinement is the closest
# faithful equivalent, and now genuinely consumes steps/lr
# instead of ignoring them. Document this as a stated limitation
# in your dissertation methodology section: exact gradient-based
# trigger reconstruction was not implemented for sklearn MLPs;
# an iterative local-search approximation was used instead.
# ============================================================

def neural_cleanse_style_score(
    model,
    X,
    target_label=1,
    steps=150,
    lr=0.05,
):

    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or len(X) == 0:
        return {"component": "NEURAL_CLEANSE", "score": None, "decision": "NO_DATA"}
    if not hasattr(model, "predict_proba") or not hasattr(model, "classes_"):
        raise ValueError("Neural Cleanse requires a fitted probabilistic classifier.")

    class_positions = np.flatnonzero(np.asarray(model.classes_) == target_label)
    if len(class_positions) != 1:
        raise ValueError(f"Target label {target_label!r} is not a unique model class.")
    target_position = int(class_positions[0])

    # Deterministic, evenly-spaced sampling avoids a random search trajectory.
    sample = X[np.linspace(0, len(X) - 1, min(200, len(X)), dtype=int)]
    baseline_probability = float(model.predict_proba(sample)[:, target_position].mean())
    percentile_grid = (1, 5, 10, 25, 50, 75, 90, 95, 99)

    def evaluate(feature_index, value, median, scale):
        modified = sample.copy()
        modified[:, feature_index] = value
        probabilities = model.predict_proba(modified)[:, target_position]
        mean_probability = float(probabilities.mean())
        high_confidence_fraction = float((probabilities >= 0.90).mean())
        perturbation_norm = float(abs(value - median) / scale)
        # Continuous objective: target-probability gain per normalized change.
        objective = max(0.0, mean_probability - baseline_probability) / (1.0 + perturbation_norm)
        return objective, mean_probability, high_confidence_fraction, perturbation_norm

    best = None
    candidates_evaluated = 0
    for feature_index in range(sample.shape[1]):
        values = np.unique(np.percentile(sample[:, feature_index], percentile_grid))
        median = float(np.median(sample[:, feature_index]))
        scale = float(np.median(np.abs(sample[:, feature_index] - median)) * 1.4826 + 1e-6)
        for value in values:
            candidate = evaluate(feature_index, float(value), median, scale)
            candidates_evaluated += 1
            record = (candidate[0], feature_index, float(value), median, scale, *candidate[1:])
            if best is None or record[0] > best[0]:
                best = record

    # Deterministic two-sided coordinate refinement replaces random moves.
    objective, best_feature, best_value, median, scale, mean_probability, high_fraction, norm = best
    accepted_moves = 0
    trajectory = []
    step_size = max(float(lr) * scale, 1e-8)
    for _ in range(max(0, int(steps))):
        options = []
        for candidate_value in (best_value - step_size, best_value + step_size):
            candidate = evaluate(best_feature, candidate_value, median, scale)
            options.append((candidate[0], candidate_value, *candidate[1:]))
            candidates_evaluated += 1
        improved = max(options, key=lambda item: item[0])
        if improved[0] <= objective + 1e-12:
            step_size *= 0.5
            if step_size < 1e-8:
                break
            continue
        objective, best_value, mean_probability, high_fraction, norm = improved
        accepted_moves += 1
        trajectory.append({"objective": float(objective), "mean_target_probability": float(mean_probability)})

    return {
        "component": "NEURAL_CLEANSE",
        "method": "deterministic_continuous_one_feature_reverse_trigger_search",
        "score": float(np.clip(objective, 0.0, 1.0)),
        "best_feature": int(best_feature),
        "best_value": float(best_value),
        "baseline_mean_target_probability": baseline_probability,
        "mean_target_probability": float(mean_probability),
        "target_probability_gain": float(mean_probability - baseline_probability),
        "target_probability_ge_0_90": float(high_fraction),
        "perturbation_norm": float(norm),
        "candidates_evaluated": int(candidates_evaluated),
        "refinement_steps_used": int(steps),
        "refinement_accepted_moves": int(accepted_moves),
        "refinement_learning_rate": float(lr),
        "trajectory": trajectory,
        "decision": "UNCALIBRATED",
    }


# ============================================================
# COMPONENT 4 FUSION
# ============================================================

def fuse(
    neural_cleanse,
    strip,
    activation_clustering,
    cfg,
):

    weights = cfg[
        "detection"
    ]

    w1 = float(
        weights["nc_weight"]
    )

    w2 = float(
        weights["strip_weight"]
    )

    w3 = float(
        weights[
            "cluster_weight"
        ]
    )

    # STRIP's raw 1 - normalized-entropy value is a diagnostic statistic,
    # not a portable anomaly probability.  It must not influence an
    # operational decision unless a clean-reference calibration has first
    # demonstrated discrimination.  The Phase 2 evaluator performs that
    # calibration separately; the raw STRIP value remains in the evidence.
    include_strip = bool(weights.get("strip_validated_for_fusion", False))
    operational_strip_weight = w2 if include_strip else 0.0
    total = w1 + operational_strip_weight + w3
    if total <= 0:
        raise ValueError("At least one S2 operational fusion weight must be positive.")

    score = float(
        (
            w1 * neural_cleanse
            + operational_strip_weight * strip
            + w3 * activation_clustering
        )
        / total
    )

    threshold = float(
        weights[
            "risk_threshold"
        ]
    )

    return {
        "risk_score": score,
        "threshold": threshold,
        "decision": (
            "MALICIOUS"
            if score >= threshold
            else "CLEAN"
        ),
        "weights": {
            "neural_cleanse":
                w1 / total,
            "strip":
                operational_strip_weight / total,
            "activation_clustering":
                w3 / total,
        },
        "strip_diagnostic_only": not include_strip,
        "strip_raw_score": float(strip),
    }


# ============================================================
# S1 DETECTION   *** FIXED: independent recomputation ***
# ============================================================
#
# PREVIOUS BEHAVIOUR: trusted the "accuracy" and "rows_flipped"
# fields already written by the Phase 1 attack script — i.e. it
# asked the attacker "did you flip labels?" and believed the
# answer, rather than checking independently.
#
# FIX: this detector now does two independent things instead:
#   (a) recomputes the actual flip proportion directly from
#       label_flip_samples.csv (original_label vs poisoned_label
#       columns), instead of trusting the stored boolean/count.
#   (b) if the S1-trained model and the clean test set are both
#       available, independently RE-RUNS the model on clean_test.csv
#       and recomputes accuracy itself, instead of reading the
#       accuracy number the attack script already computed.
# ============================================================

def s1_detector(
    threshold,
    samples_path=None,
    model_path=None,
    training_path=None,
    clean_training_path=None,
    clean_model_path=None,
    label_weight=0.4,
    oof_weight=0.3,
    degradation_weight=0.3,
    oof_splits=5,
):
    """Detect S1 label-flip evidence without trusting attack-produced metrics.

    ``risk_score`` is deliberately a transparent weighted combination of two
    raw rates in [0, 1].  The weights and operating threshold must be frozen
    from a separate calibration experiment; they must not be tuned on the
    held-out evaluation runs.
    """
    if not np.isclose(label_weight + oof_weight + degradation_weight, 1.0):
        raise ValueError("S1 fusion weights must sum to 1.0.")
    if int(oof_splits) < 2:
        raise ValueError("S1 OOF evaluation requires at least two folds.")

    result = {
        "detector": "S1_LABEL_FLIP",
        "threshold": float(threshold),
        "evidence_status": "COMPLETE",
        "fusion_formula": (
            f"{label_weight:.3f} * label_integrity_score + "
            f"{oof_weight:.3f} * oof_inconsistency_score + "
            f"{degradation_weight:.3f} * model_degradation_score"
        ),
        "fusion_weights": {
            "label_integrity": float(label_weight),
            "oof_inconsistency": float(oof_weight),
            "model_degradation": float(degradation_weight),
        },
    }

    # A protected, immutable row identifier is mandatory.  Positional
    # comparison is invalid because either file may have been shuffled.
    if training_path is None or clean_training_path is None:
        return {
            **result,
            "risk_score": None,
            "decision": "INSUFFICIENT_EVIDENCE",
            "evidence_status": "INCOMPLETE",
            "reason": "Both protected clean and suspect training files are required.",
        }

    poisoned_train = pd.read_csv(training_path)
    clean_train = pd.read_csv(clean_training_path)
    required_columns = {"row_id", "label"}
    missing_clean = required_columns.difference(clean_train.columns)
    missing_suspect = required_columns.difference(poisoned_train.columns)
    if missing_clean or missing_suspect:
        return {
            **result,
            "risk_score": None,
            "decision": "INSUFFICIENT_EVIDENCE",
            "evidence_status": "INCOMPLETE",
            "reason": (
                "S1 requires immutable row_id and label columns in both training "
                "files; regenerate Phase 0/Phase 1 artifacts with row_id preserved."
            ),
            "missing_clean_columns": sorted(missing_clean),
            "missing_suspect_columns": sorted(missing_suspect),
        }

    if clean_train["row_id"].duplicated().any() or poisoned_train["row_id"].duplicated().any():
        raise ValueError("row_id must be unique in each S1 training artifact.")

    label_comparison = clean_train[["row_id", "label"]].merge(
        poisoned_train[["row_id", "label"]],
        on="row_id",
        how="outer",
        suffixes=("_clean", "_suspect"),
        indicator=True,
        validate="one_to_one",
    )
    unmatched = label_comparison["_merge"] != "both"
    if unmatched.any():
        return {
            **result,
            "risk_score": None,
            "decision": "INSUFFICIENT_EVIDENCE",
            "evidence_status": "INCOMPLETE",
            "reason": "Clean and suspect training artifacts do not contain the same row_id set.",
            "clean_only_rows": int((label_comparison["_merge"] == "left_only").sum()),
            "suspect_only_rows": int((label_comparison["_merge"] == "right_only").sum()),
        }

    flip_rate = float(
        (label_comparison["label_clean"] != label_comparison["label_suspect"]).mean()
    )
    # This is a raw, interpretable proportion—not an arbitrary multiplier.
    label_integrity_score = flip_rate
    result["label_integrity"] = {
        "score": label_integrity_score,
        "flip_rate": flip_rate,
        "rows_compared": int(len(label_comparison)),
        "method": "protected_row_id_one_to_one_label_comparison",
    }

    selected_model_path = Path(model_path) if model_path is not None else S1_MODEL_PATH
    selected_clean_model_path = (
        Path(clean_model_path) if clean_model_path is not None else CLEAN_MODEL_PATH
    )
    if not selected_model_path.exists() or not selected_clean_model_path.exists() or not CLEAN_TEST_PATH.exists():
        return {
            **result,
            "risk_score": None,
            "decision": "INSUFFICIENT_EVIDENCE",
            "evidence_status": "INCOMPLETE",
            "reason": "Suspect model, verified clean baseline model, and untouched clean test set are all required.",
        }

    test_df = pd.read_csv(CLEAN_TEST_PATH)
    if "label" not in test_df.columns:
        raise ValueError("The untouched clean test file must contain a label column.")
    y_test = test_df["label"].to_numpy()

    def evaluate_bundle(bundle_path):
        bundle = load(bundle_path)
        if not {"model", "scaler", "features"}.issubset(bundle):
            raise ValueError(f"Model bundle lacks model, scaler, or features: {bundle_path}")
        feature_columns = list(bundle["features"])
        missing_features = set(feature_columns).difference(test_df.columns)
        if missing_features:
            raise ValueError(f"Test data is missing model features: {sorted(missing_features)}")
        X_frame = test_df[feature_columns].replace([np.inf, -np.inf], np.nan)
        if X_frame.isna().any().any():
            raise ValueError("Non-finite or missing values found in the untouched S1 test data.")
        predictions = bundle["model"].predict(bundle["scaler"].transform(X_frame.astype("float64")))
        return float(accuracy_score(y_test, predictions)), bundle

    clean_accuracy, _ = evaluate_bundle(selected_clean_model_path)
    suspect_accuracy, suspect_bundle = evaluate_bundle(selected_model_path)
    accuracy_drop = max(0.0, clean_accuracy - suspect_accuracy)
    # Raw drop from the clean baseline: not the suspect model's total error.
    model_degradation_score = accuracy_drop
    result["model_degradation"] = {
        "score": model_degradation_score,
        "clean_baseline_accuracy": clean_accuracy,
        "suspect_model_accuracy": suspect_accuracy,
        "accuracy_drop": accuracy_drop,
        "method": "same_untouched_test_set_clean_minus_suspect_accuracy",
    }

    # Genuine 5-fold out-of-fold behavioural evidence.  Every fold trains a
    # fresh clone only on its fold-training rows and predicts its held-out rows.
    # The signal rises only when the suspect-trained model agrees more with the
    # suspect labels than with the protected clean labels.
    oof_features = list(suspect_bundle["features"])
    missing_oof_features = set(oof_features).difference(clean_train.columns)
    if missing_oof_features:
        raise ValueError(f"S1 training data is missing model features: {sorted(missing_oof_features)}")
    X_oof = clean_train[oof_features].replace([np.inf, -np.inf], np.nan)
    if X_oof.isna().any().any():
        raise ValueError("Non-finite or missing values found in S1 clean training data.")
    aligned = label_comparison.set_index("row_id").loc[clean_train["row_id"]]
    y_clean = aligned["label_clean"].to_numpy()
    y_suspect = aligned["label_suspect"].to_numpy()
    class_counts = pd.Series(y_suspect).value_counts()
    effective_splits = min(int(oof_splits), int(class_counts.min()))
    if effective_splits < 2:
        raise ValueError("S1 OOF evaluation requires at least two suspect-label samples per class.")
    oof_predictions = np.empty(len(X_oof), dtype=y_suspect.dtype)
    splitter = StratifiedKFold(n_splits=effective_splits, shuffle=True, random_state=42)
    template_model = suspect_bundle["model"]
    for train_index, test_index in splitter.split(X_oof, y_suspect):
        fold_scaler = clone(suspect_bundle["scaler"])
        fold_model = clone(template_model)
        X_train_fold = fold_scaler.fit_transform(X_oof.iloc[train_index].astype("float64"))
        X_test_fold = fold_scaler.transform(X_oof.iloc[test_index].astype("float64"))
        fold_model.fit(X_train_fold, y_suspect[train_index])
        oof_predictions[test_index] = fold_model.predict(X_test_fold)
    agreement_clean = float(np.mean(oof_predictions == y_clean))
    agreement_suspect = float(np.mean(oof_predictions == y_suspect))
    oof_inconsistency_score = max(0.0, agreement_suspect - agreement_clean)
    result["oof_inconsistency"] = {
        "score": float(oof_inconsistency_score),
        "folds": int(effective_splits),
        "agreement_with_clean_labels": agreement_clean,
        "agreement_with_suspect_labels": agreement_suspect,
        "method": "stratified_kfold_held_out_prediction_agreement_gap",
    }

    risk_score = float(
        label_weight * label_integrity_score
        + oof_weight * oof_inconsistency_score
        + degradation_weight * model_degradation_score
    )
    result["risk_score"] = risk_score
    result["decision"] = "MALICIOUS" if risk_score >= threshold else "CLEAN"
    return result


# ============================================================
# S1 / NEURAL CLEANSE CALIBRATION
# ============================================================

def calibrate_continuous_detector(clean_scores, attack_scores, target_fpr=0.05):
    """Freeze a threshold from calibration scores, never held-out scores.

    The threshold is the empirical clean-score quantile associated with the
    chosen maximum false-positive rate.  Call this only with repetitions 1–3;
    reserve repetitions 4–5 for final evaluation.
    """
    clean = np.asarray([s for s in clean_scores if s is not None], dtype=float)
    attack = np.asarray([s for s in attack_scores if s is not None], dtype=float)
    if not (0.0 <= target_fpr < 1.0):
        raise ValueError("target_fpr must be in [0, 1).")
    if len(clean) == 0 or len(attack) == 0:
        raise ValueError("Calibration requires at least one clean and one attack score.")
    threshold = float(np.quantile(clean, 1.0 - target_fpr))
    return {
        "threshold": threshold,
        "target_fpr": float(target_fpr),
        "clean_distribution": {
            "n": int(len(clean)), "mean": float(clean.mean()), "std": float(clean.std(ddof=0)),
            "min": float(clean.min()), "max": float(clean.max()),
        },
        "attack_distribution": {
            "n": int(len(attack)), "mean": float(attack.mean()), "std": float(attack.std(ddof=0)),
            "min": float(attack.min()), "max": float(attack.max()),
        },
        "calibration_detection_rate": float(np.mean(attack >= threshold)),
        "calibration_false_positive_rate": float(np.mean(clean >= threshold)),
        "protocol": "threshold_frozen_from_calibration_runs_only",
    }


def validate_neural_cleanse_sensitivity(clean_result, poisoned_results, minimum_delta=1e-6):
    """Report whether the NC-style optimisation changes across model states.

    This is a validation gate, not a detector: identical optimisation outputs
    must be reported as a failed sensitivity test rather than as evidence of a
    backdoor or a clean model.
    """
    clean_score = float(clean_result["score"])
    comparisons = []
    for poisoned in poisoned_results:
        delta = float(poisoned["score"]) - clean_score
        changed = abs(delta) > float(minimum_delta)
        comparisons.append({
            "poisoned_score": float(poisoned["score"]),
            "score_delta_from_clean": delta,
            "optimisation_changed": changed,
            "clean_best_feature": clean_result.get("best_feature"),
            "poisoned_best_feature": poisoned.get("best_feature"),
        })
    return {
        "validation": "neural_cleanse_model_sensitivity",
        "passed": bool(comparisons) and any(item["optimisation_changed"] for item in comparisons),
        "minimum_score_delta": float(minimum_delta),
        "comparisons": comparisons,
        "interpretation": (
            "FAILED means the clean and poisoned models were not separated by "
            "this NC-style method; do not claim NC detection until it passes."
        ),
    }


# ============================================================
# S3 DETECTION   *** FIXED: independent recomputation ***
# ============================================================
#
# PREVIOUS BEHAVIOUR: read "compromise_detected" — a boolean the
# attack script already computed and stored — and trusted it.
#
# FIX: independently re-reads the trusted and observed/compromised
# files by their known fixed filenames and RECOMPUTES both
# SHA-256 hashes itself, then compares them directly. It only
# falls back to the stored evidence hashes if the raw files are
# no longer on disk.
# ============================================================

def s3_detector(threshold):

    evidence_file = S3_DIR / "s3_results.json"

    trusted_file = S3_DIR / "trusted_package.txt"
    compromised_file = S3_DIR / "controlled_modified_package.txt"

    if trusted_file.exists() and compromised_file.exists():

        recomputed_trusted_hash = sha256_file(trusted_file)
        recomputed_observed_hash = sha256_file(compromised_file)

        mismatch = (
            recomputed_trusted_hash
            != recomputed_observed_hash
        )

        score = 1.0 if mismatch else 0.0

        return {
            "risk_score": score,
            "threshold": threshold,
            "decision": (
                "MALICIOUS"
                if score >= threshold
                else "CLEAN"
            ),
            "hash_mismatch": bool(mismatch),
            "recomputed_trusted_sha256":
                recomputed_trusted_hash,
            "recomputed_observed_sha256":
                recomputed_observed_hash,
            "verification": "recomputed_from_raw_files",
        }

    # Fallback: raw files are gone, only stored evidence remains.
    if not evidence_file.exists():

        return {
            "risk_score": 0.0,
            "decision": "NO_EVIDENCE",
        }

    with open(evidence_file, "r", encoding="utf-8") as f:

        data = json.load(f)

    mismatch = data.get("compromise_detected", False)

    score = 1.0 if mismatch else 0.0

    return {
        "risk_score": score,
        "threshold": threshold,
        "decision": (
            "MALICIOUS" if score >= threshold else "CLEAN"
        ),
        "hash_mismatch": bool(mismatch),
        "verification": "stored_evidence_only_raw_files_missing",
    }


# ============================================================
# S4 DETECTION   *** FIXED: accepts artifact_path override ***
# ============================================================

def s4_detector(threshold, artifact_path=None):

    artifact = (
        Path(artifact_path)
        if artifact_path is not None
        else (
            ROOT
            / "evidence"
            / "phase1"
            / "S4_model_hub"
            / "controlled_model.pkl"
        )
    )

    if not artifact.exists():

        return {
            "risk_score": 0.0,
            "decision":
                "NO_ARTIFACT",
        }

    scan = recursive_pickle_scan(
        artifact
    )

    dangerous_count = int(
        scan.get(
            "dangerous_global_count",
            0,
        )
    )

    global_count = int(
        scan.get(
            "global_operation_count",
            0,
        )
    )

    if dangerous_count > 0:

        score = 1.0

    elif global_count > 0:

        score = 0.40

    else:

        score = 0.0

    return {
        "risk_score": score,
        "threshold": threshold,
        "decision": (
            "MALICIOUS"
            if score >= threshold
            else "CLEAN"
        ),
        "artifact":
            str(artifact),
        "sha256":
            sha256_file(
                artifact
            ),
        "pickle_scan":
            scan,
    }


# ============================================================
# S5 DETECTION   *** FIXED: no more unhandled crash ***
# ============================================================
#
# PREVIOUS BUG:
#   if not evidence.exists():
#       alternative = <the S5 evidence DIRECTORY>
#       if not alternative.exists():
#           return NO_EVIDENCE
#   # <-- falls through here if the directory exists but the
#   #     JSON file inside it does not, then crashes below:
#   with open(evidence, ...) as f:   # FileNotFoundError
#
# Since phase1_master.py pre-creates all five evidence
# directories up front, the directory almost always exists —
# so this guard essentially never caught the missing-file case
# it was meant to catch.
#
# FIX: this detector now independently recomputes the SHA-256
# hashes of the original and tampered model files directly
# (rather than trusting the stored "tampering_detected" flag),
# and correctly returns NO_EVIDENCE if either raw file is
# missing, instead of falling through to a crash.
# ============================================================

def s5_detector(threshold, artifact_path=None):

    tampered_file = (
        S5_DIR / "tampered_clean_baseline.joblib"
    )

    if not CLEAN_MODEL_PATH.exists() or not tampered_file.exists():

        # FIXED: correctly checks the actual FILES needed for
        # recomputation, not just the directory, and returns
        # NO_EVIDENCE cleanly instead of falling through.
        return {
            "risk_score": 0.0,
            "decision": "NO_EVIDENCE",
        }

    recomputed_original_hash = sha256_file(
        CLEAN_MODEL_PATH
    )

    recomputed_tampered_hash = sha256_file(
        tampered_file
    )

    tampered = (
        recomputed_original_hash
        != recomputed_tampered_hash
    )

    score = (
        1.0
        if tampered
        else 0.0
    )

    return {
        "risk_score": score,
        "threshold": threshold,
        "decision": (
            "MALICIOUS"
            if score >= threshold
            else "CLEAN"
        ),
        "tampering_detected":
            bool(tampered),
        "recomputed_original_sha256":
            recomputed_original_hash,
        "recomputed_tampered_sha256":
            recomputed_tampered_hash,
        "verification": "recomputed_from_raw_files",
    }


# ============================================================
# MAIN
# ============================================================

def phase2_master_legacy_main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        required=True,
        help=(
            "Clean baseline joblib model "
            "containing model + scaler."
        ),
    )

    parser.add_argument(
        "--data",
        required=True,
        help=(
            "CSV containing test features."
        ),
    )

    parser.add_argument(
        "--s4-artifact",
        default=None,
        help=(
            "Optional explicit S4 Pickle "
            "artifact path."
        ),
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
    )
    parser.add_argument("--s1-model", default=None)
    parser.add_argument("--s1-samples", default=None)
    parser.add_argument("--s1-training", default=None)
    parser.add_argument("--clean-training", default=None)

    parser.add_argument(
        "--require-art",
        action="store_true",
        help="Fail the run unless IBM Adversarial Robustness Toolbox is installed and usable.",
    )

    args = parser.parse_args()

    if args.require_art and not ART_AVAILABLE:
        raise RuntimeError(
            "IBM Adversarial Robustness Toolbox is required for this run but is not installed. "
            "Install adversarial-robustness-toolbox and rerun with --require-art."
        )

    print()
    print("=" * 80)
    print(
        "CODE PHASE 2 — "
        "FIVE-COMPONENT DETECTION FRAMEWORK"
    )
    print("=" * 80)

    cfg = load_cfg()

    if args.threshold is not None:

        cfg[
            "detection"
        ][
            "risk_threshold"
        ] = args.threshold

    threshold = float(
        cfg[
            "detection"
        ][
            "risk_threshold"
        ]
    )

    print(
        f"\nDetection threshold: "
        f"{threshold}"
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    bundle = load(
        args.model
    )

    model = bundle[
        "model"
    ]

    scaler = bundle[
        "scaler"
    ]

    # --------------------------------------------------------
    # Load test data
    # --------------------------------------------------------

    df = pd.read_csv(
        args.data
    )

    target = cfg[
        "dataset"
    ][
        "target_column"
    ]

    if target in df.columns:

        df = df.drop(
            columns=[target]
        )

    for column in df.columns:

        if not pd.api.types.is_numeric_dtype(
            df[column]
        ):

            df[column] = (
                pd.factorize(
                    df[column]
                )[0]
            )

    X_raw = (
        df
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .fillna(0)
        .to_numpy(
            dtype=np.float32
        )
    )

    # X_raw is kept unscaled so the S2 section below can transform
    # it through the S2 backdoor model's OWN scaler instead of the
    # clean baseline's scaler used here for `X`.
    X = scaler.transform(
        X_raw
    )

    print(
        f"\nDataset features : "
        f"{X.shape[1]}"
    )

    print(
        f"Test rows        : "
        f"{X.shape[0]}"
    )

    # --------------------------------------------------------
    # Resolve S4 artifact path ONCE, so Component 2's printed
    # analysis and S4's actual detection always use the same
    # file — FIXED: previously these could silently disagree.
    # --------------------------------------------------------

    if args.s4_artifact:

        s4_path = Path(
            args.s4_artifact
        )

    else:

        s4_path = (
            ROOT
            / "evidence"
            / "phase1"
            / "S4_model_hub"
            / "controlled_model.pkl"
        )

    # ========================================================
    # COMPONENT 1
    # ========================================================

    print()
    print("=" * 80)
    print(
        "COMPONENT 1 — SBOM + HASH"
    )
    print("=" * 80)

    component1 = (
        component1_sbom_hash(
            args.model
        )
    )

    print(
        "SBOM status: "
        f"{component1['status']}"
    )

    # ========================================================
    # COMPONENT 2
    # ========================================================

    print()
    print("=" * 80)
    print(
        "COMPONENT 2 — "
        "PICKLE / MODEL FORMAT"
    )
    print("=" * 80)

    if s4_path.exists():

        pickle_scan = (
            recursive_pickle_scan(
                s4_path
            )
        )

        format_policy = (
            model_format_policy(
                s4_path
            )
        )

        print(
            "Pickle static analysis: "
            f"{'DANGEROUS_GLOBAL_DETECTED' if pickle_scan.get('dangerous_global_count', 0) > 0 else 'NO_DANGEROUS_GLOBAL'}"
        )

        print(
            "Model format policy: "
            f"{format_policy['format']}"
        )

        print(
            "Dangerous globals: "
            f"{pickle_scan.get('dangerous_globals_found', [])}"
        )

    else:

        pickle_scan = {
            "component":
                "PICKLE_SCANNER",
            "safe": True,
            "findings": [],
            "global_operation_count":
                0,
            "dangerous_global_count":
                0,
            "dangerous_globals_found":
                [],
        }

        format_policy = {
            "component":
                "MODEL_HUB_GATE",
            "allowed": False,
            "format": "NOT_PROVIDED",
        }

        print(
            "S4 artifact not provided."
        )

    # ========================================================
    # COMPONENT 3 — TRAINING ANOMALY MONITOR
    # ========================================================

    print()
    print("=" * 80)
    print("COMPONENT 3 — TRAINING ANOMALY MONITOR")
    print("=" * 80)

    component3 = training_anomaly_from_model(model)

    print(
        f"Training anomaly score: "
        f"{component3['score']:.6f}"
    )
    print(
        f"Training monitor decision: "
        f"{component3['decision']}"
    )
    print(
        f"Training history length: "
        f"{component3.get('history_length', 0)}"
    )
    print(
        "Gradient source: loss finite-difference proxy"
    )

    # ========================================================
    # COMPONENT 4 — IBM ART BACKEND VALIDATION
    # ========================================================

    print()
    print("=" * 80)
    print("COMPONENT 4 — IBM ART INTEGRATION")
    print("=" * 80)

    art_report = art_estimator_report(model)

    print(
        f"ART available: {art_report['available']}"
    )
    print(
        f"ART version: {art_report.get('version')}"
    )

    if not art_report["available"]:
        print(
            "ART status: NOT INSTALLED — install "
            "adversarial-robustness-toolbox before the final run."
        )
    else:
        print(
            f"ART estimator: {art_report.get('estimator')}"
        )
        print(
            f"ART status: {art_report.get('status')}"
        )

    # ========================================================
    # S1
    # ========================================================

    print()
    print("=" * 80)
    print(
        "S1 — LABEL FLIP DETECTION"
    )
    print("=" * 80)

    s1 = s1_detector(
        threshold,
        samples_path=args.s1_samples,
        model_path=args.s1_model,
        training_path=args.s1_training,
        clean_training_path=args.clean_training,
    )

    print(
        f"S1 risk score: "
        f"{s1['risk_score']:.6f}"
    )

    print(
        f"S1 decision: "
        f"{s1['decision']}"
    )

    # ========================================================
    # S2
    # ========================================================

    print()
    print("=" * 80)
    print(
        "S2 — BACKDOOR DETECTION"
    )
    print("=" * 80)

    # FIXED: previously this section ran neural_cleanse_style_score /
    # strip_entropy_score / activation_clustering_score against
    # `model`/`X`, which were loaded from --model (the CLEAN
    # baseline) at the top of main(). That meant "S2 — BACKDOOR
    # DETECTION" was silently scoring the clean model every run,
    # never the actual backdoored model. Confirmed by comparing
    # against a clean-control run: the scores were identical to six
    # decimal places, which is only possible if the same model was
    # scored both times.
    #
    # Now loads models/S2_backdoor/backdoor_model.joblib directly,
    # with its OWN saved scaler (not the clean baseline's scaler —
    # each attack model bundle carries its own scaler fit during
    # that scenario's training run), and transforms the raw test
    # dataframe through it fresh, matching how S1 already does its
    # own independent model load.

    if S2_MODEL_PATH.exists():

        s2_bundle = load(S2_MODEL_PATH)

        s2_model = s2_bundle["model"]
        s2_scaler = s2_bundle["scaler"]

        X_s2 = s2_scaler.transform(X_raw)

        nc = neural_cleanse_style_score(
            s2_model,
            X_s2,
            cfg[
                "attacks"
            ][
                "target_label"
            ],
            cfg[
                "detection"
            ][
                "nc_steps"
            ],
            cfg[
                "detection"
            ][
                "nc_learning_rate"
            ],
        )

        strip = strip_entropy_score(
            s2_model,
            X_s2,
            cfg[
                "detection"
            ][
                "strip_samples"
            ],
        )

        cluster = (
            activation_clustering_score(
                s2_model,
                X_s2,
                cfg[
                    "detection"
                ][
                    "cluster_min_samples"
                ],
            )
        )

        fusion = fuse(
            nc["score"],
            strip["score"],
            cluster["score"],
            cfg,
        )

        # ADDED: run the real IBM ART ActivationDefence against the
        # corresponding poisoned S2 training dataset when available.
        # Resolve the exact S2 training artifact from the model bundle rather than
        # choosing the newest file, which could belong to another rate/run.
        bundle_rate = s2_bundle.get("poison_rate")
        bundle_repetition = s2_bundle.get("repetition")
        art_s2 = {
            "component": "ART_ACTIVATION_CLUSTERING",
            "available": ART_AVAILABLE,
            "status": "NO_TRAINING_FILE",
            "score": 0.0,
        }
        if bundle_rate is not None and bundle_repetition is not None:
            token = f"{int(round(float(bundle_rate) * 100)):02d}"
            exact_art_path = ROOT / "evidence" / "phase1" / "S2_backdoor" / (
                f"backdoor_training_rate{token}_run{int(bundle_repetition):02d}.csv"
            )
            if exact_art_path.exists():
                train_df = pd.read_csv(exact_art_path)
                if "label" not in train_df.columns:
                    raise ValueError(f"Exact S2 ART training file has no label column: {exact_art_path}")
                X_art = train_df.drop(columns=["label"]).to_numpy(dtype=np.float32)
                y_art = train_df["label"].to_numpy(dtype=int)
                art_s2 = art_activation_clustering_score(
                    s2_model, X_art, y_art,
                )
            else:
                art_s2["status"] = "EXACT_TRAINING_FILE_MISSING"
                art_s2["expected_path"] = str(exact_art_path)
        else:
            art_s2["status"] = "MODEL_METADATA_MISSING_RATE_OR_REPETITION"


    else:

        print(
            f"S2 backdoor model not found at:\n{S2_MODEL_PATH}"
        )

        nc = {"score": 0.0, "decision": "NO_MODEL"}
        strip = {"score": 0.0, "decision": "NO_MODEL"}
        cluster = {"score": 0.0, "decision": "NO_MODEL"}

        fusion = {
            "risk_score": 0.0,
            "threshold": threshold,
            "decision": "NO_EVIDENCE",
            "weights": {},
        }

        art_s2 = {
            "component": "ART_ACTIVATION_CLUSTERING",
            "available": ART_AVAILABLE,
            "status": "NO_MODEL",
            "score": 0.0,
        }

    print(
        f"Neural Cleanse: "
        f"{nc['score']:.6f} "
        f"(refinement steps used: "
        f"{nc.get('refinement_steps_used', 0)}, "
        f"accepted moves: "
        f"{nc.get('refinement_accepted_moves', 0)})"
    )

    print(
        f"STRIP: "
        f"{strip['score']:.6f}"
    )

    print(
        f"Activation Clustering: "
        f"{cluster['score']:.6f}"
    )

    print(
        f"Fusion risk score: "
        f"{fusion['risk_score']:.6f}"
    )

    print(
        f"S2 decision: "
        f"{fusion['decision']}"
    )

    print(
        f"IBM ART Activation Clustering: "
        f"{art_s2.get('score', 0.0):.6f} "
        f"status={art_s2.get('status')}"
    )

    # ========================================================
    # S3
    # ========================================================

    print()
    print("=" * 80)
    print(
        "S3 — PYPI DEPENDENCY DETECTION"
    )
    print("=" * 80)

    s3 = s3_detector(
        threshold
    )

    print(
        f"S3 risk score: "
        f"{s3['risk_score']:.6f}"
    )

    print(
        f"S3 decision: "
        f"{s3['decision']}"
    )

    print(
        f"S3 verification: "
        f"{s3.get('verification', 'n/a')}"
    )

    # ========================================================
    # S4
    # ========================================================

    print()
    print("=" * 80)
    print(
        "S4 — MODEL HUB DETECTION"
    )
    print("=" * 80)

    # FIXED: pass the resolved s4_path through explicitly, so
    # this section always matches what Component 2 just printed.
    s4 = s4_detector(
        threshold,
        artifact_path=s4_path
    )

    print(
        f"S4 risk score: "
        f"{s4['risk_score']:.6f}"
    )

    print(
        f"S4 decision: "
        f"{s4['decision']}"
    )

    if "pickle_scan" in s4:

        print(
            "Dangerous globals: "
            f"{s4['pickle_scan'].get('dangerous_globals_found', [])}"
        )

    # ========================================================
    # S5
    # ========================================================

    print()
    print("=" * 80)
    print(
        "S5 — MLFLOW TAMPERING DETECTION"
    )
    print("=" * 80)

    s5 = s5_detector(
        threshold
    )

    print(
        f"S5 risk score: "
        f"{s5['risk_score']:.6f}"
    )

    print(
        f"S5 decision: "
        f"{s5['decision']}"
    )

    print(
        f"S5 verification: "
        f"{s5.get('verification', 'n/a')}"
    )

    # ========================================================
    # COMPONENT 5 — MLFLOW SECURITY DECISION INTERFACE
    # ========================================================

    component_scores = {
        "component_3_training_monitor": float(component3.get("score", 0.0)),
        "S1_LABEL_FLIP": float(s1.get("risk_score", 0.0)),
        "S2_BACKDOOR": float(fusion.get("risk_score", 0.0)),
        "S3_PYPI": float(s3.get("risk_score", 0.0)),
        "S4_MODEL_HUB": float(s4.get("risk_score", 0.0)),
        "S5_MLFLOW_TAMPERING": float(s5.get("risk_score", 0.0)),
    }

    overall_risk = max(component_scores.values())
    component5 = {
        "component": "MLFLOW_SECURITY_DECISION_INTERFACE",
        "risk_score": float(overall_risk),
        "threshold": float(threshold),
        "decision": "MALICIOUS" if overall_risk >= threshold else "CLEAN",
        "evidence_sources": component_scores,
        "component_3_training_monitor_metadata": component3,
        "promotion_allowed": bool(overall_risk < threshold),
        "methodology": {
            "detector_type": "research_heuristic_fusion",
            "not_claimed_as_art_state_of_the_art": True,
            "component_3_mode": component3.get("monitor_mode"),
        },
    }

    print()
    print("=" * 80)
    print("COMPONENT 5 — MLFLOW SECURITY DECISION INTERFACE")
    print("=" * 80)
    print(f"Overall risk score: {overall_risk:.6f}")
    print(f"Decision: {component5['decision']}")
    print(f"Promotion allowed: {component5['promotion_allowed']}")

    # ========================================================
    # FINAL EVIDENCE
    # ========================================================

    output = {

        "timestamp":
            time.time(),

        "phase":
            "CODE_PHASE_2",

        "threshold":
            threshold,

        "component_1":
            component1,

        "component_2": {
            "pickle_scan":
                pickle_scan,
            "format_policy":
                format_policy,
        },

        "component_3_training_monitor": component3,

        "component_4_art": {
            "backend": "IBM_Adversarial_Robustness_Toolbox",
            "report": art_report,
        },

        "component_5_decision_interface": component5,

        "scenarios": {

            "S1_LABEL_FLIP":
                s1,

            "S2_BACKDOOR": {

                "neural_cleanse":
                    nc,

                "strip":
                    strip,

                "activation_clustering":
                    cluster,

                "fusion":
                    fusion,
                "art_activation_clustering":
                    art_s2,
            },

            "S3_PYPI":
                s3,

            "S4_MODEL_HUB":
                s4,

            "S5_MLFLOW_TAMPERING":
                s5,
        },
    }

    result_file = (
        RESULTS_DIR
        / "phase2_detection_results.json"
    )

    evidence_file = (
        EVIDENCE_DIR
        / "phase2_evidence.json"
    )

    result_file.write_text(
        json.dumps(
            output,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    evidence_file.write_text(
        json.dumps(
            output,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 80)
    print(
        "PHASE 2 — DETECTION SUMMARY"
    )
    print("=" * 80)

    print(
        f"\nResults:"
        f"\n{result_file}"
    )

    print(
        f"\nEvidence:"
        f"\n{evidence_file}"
    )

    print()
    print("=" * 80)
    print(
        "CODE PHASE 2 — COMPLETE"
    )
    print("=" * 80)





# ===== EMBEDDED phase2_25_experiments(3).py =====
from pathlib import Path
import json
import time
import hashlib
import shutil
import tempfile

import numpy as np
import pandas as pd
from joblib import load

# ============================================================
# PHASE 2 — RATE-AWARE EXPERIMENTAL EVALUATION (45 RUNS)
#
# ADDED:
#   - Explicit S1/S2 poison-rate evaluation
#   - 1%, 5%, 10% rate-specific model selection
#   - poison_rate stored in every experiment record
#   - 45 total experiments:
#       S1: 3 rates x 5 runs = 15
#       S2: 3 rates x 5 runs = 15
#       S3: 5 runs
#       S4: 5 runs
#       S5: 5 runs
#
# UNTOUCHED:
#   - Existing Phase 2 detector internals
#   - Neural Cleanse
#   - STRIP
#   - Activation Clustering
#   - S3/S4/S5 detection logic
# ============================================================

ROOT = Path(__file__).resolve().parent

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

TEST_FILE = (
    ROOT
    / "data"
    / "processed"
    / "split"
    / "clean_test.csv"
)

S1_MODEL_DIR_PHASE2 = ROOT / "models" / "S1_label_flip"
S2_MODEL_DIR_PHASE2 = ROOT / "models" / "S2_backdoor"

EVIDENCE_DIR = (
    ROOT
    / "evidence"
    / "phase2"
    / "25_experiments"
)

EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

RESULT_FILE = EVIDENCE_DIR / "25_experiment_results.json"
SUMMARY_FILE = EVIDENCE_DIR / "25_experiment_summary.json"

# ------------------------------------------------------------
# Experiment configuration
# ------------------------------------------------------------

POISON_RATES = [0.01, 0.05, 0.10]

RUNS_PER_RATE = 5
RUNS_OTHER_SCENARIOS = 5

THRESHOLD = 0.60

# ------------------------------------------------------------
# Rate-specific models confirmed from Phase 1
# ------------------------------------------------------------

S1_MODELS = {
    0.01: S1_MODEL_DIR_PHASE2 / "label_flip_model_rate01.joblib",
    0.05: S1_MODEL_DIR_PHASE2 / "label_flip_model_rate05.joblib",
    0.10: S1_MODEL_DIR_PHASE2 / "label_flip_model_rate10.joblib",
}

S2_MODELS = {
    0.01: S2_MODEL_DIR_PHASE2 / "backdoor_model_rate01.joblib",
    0.05: S2_MODEL_DIR_PHASE2 / "backdoor_model_rate05.joblib",
    0.10: S2_MODEL_DIR_PHASE2 / "backdoor_model_rate10.joblib",
}


# ============================================================
# Utility functions
# ============================================================

def sha256_file(path):
    """Return SHA-256 hash of a file."""
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def load_json(path):
    if not path.exists():
        raise FileNotFoundError(f"Required JSON not found:\n{path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)




def rate_run_model_path(root, scenario, rate, run_number):
    """Resolve the exact Phase 1 repetition artifact.

    Final research runs fail closed when a run-specific artifact is missing.
    Legacy fallback is available only when explicitly requested for compatibility.
    """
    token = f"{int(round(rate * 100)):02d}"
    if scenario == "S1_LABEL_FLIP":
        path = root / "models" / "S1_label_flip" / f"label_flip_model_rate{token}_run{int(run_number):02d}.joblib"
    else:
        path = root / "models" / "S2_backdoor" / f"backdoor_model_rate{token}_run{int(run_number):02d}.joblib"
    if path.exists():
        return path, "INDEPENDENT_RUN_MODEL"
    raise FileNotFoundError(
        f"Required independent model missing for {scenario}, rate={rate}, run={run_number}: {path}"
    )


def verify_rate_models():
    """
    Verify that all six Phase 1 rate-specific models exist.
    """

    print("=" * 80)
    print("VERIFYING RATE-SPECIFIC PHASE 1 MODELS")
    print("=" * 80)

    missing = []

    for rate, path in S1_MODELS.items():
        status = "OK" if path.exists() else "MISSING"

        print(
            f"S1 | rate={rate:.0%} | "
            f"{path.name} | {status}"
        )

        if not path.exists():
            missing.append(path)

    for rate, path in S2_MODELS.items():
        status = "OK" if path.exists() else "MISSING"

        print(
            f"S2 | rate={rate:.0%} | "
            f"{path.name} | {status}"
        )

        if not path.exists():
            missing.append(path)

    if missing:
        message = "\n".join(str(p) for p in missing)

        raise FileNotFoundError(
            "\nThe following rate-specific Phase 1 models are missing:\n"
            + message
            + "\n\nRun phase1_master.py before Phase 2."
        )

    print("\nAll six rate-specific models found.")
    print()


def load_phase2_module():
    """Return the embedded Phase 2 detector module (this file)."""
    return sys.modules[__name__]


def load_test_data():
    if not TEST_FILE.exists():
        raise FileNotFoundError(
            f"Clean test dataset not found:\n{TEST_FILE}"
        )

    df = pd.read_csv(TEST_FILE)

    if "label" not in df.columns:
        raise ValueError(
            "clean_test.csv must contain a 'label' column."
        )

    return df


# ============================================================
# S1
# ============================================================

def run_s1(rate, run_number, phase2):
    """Run the real Phase 2 S1 detector against the exact rate-specific artefacts."""
    model_path, model_source = rate_run_model_path(ROOT, "S1_LABEL_FLIP", rate, run_number, )
    rate_token = f"{int(round(rate * 100)):02d}"
    rate_samples = (
        ROOT / "evidence" / "phase1" / "S1_label_flip"
        / f"label_flip_samples_rate{rate_token}_run{int(run_number):02d}.csv"
    )
    rate_training = (
        ROOT / "evidence" / "phase1" / "S1_label_flip"
        / f"label_flip_training_rate{rate_token}_run{int(run_number):02d}.csv"
    )

    if not rate_samples.exists():
        raise FileNotFoundError(
            f"Rate-specific S1 samples not found:\n{rate_samples}"
        )

    print(f"S1 | Rate {rate:.0%} | Run {run_number}")
    start = time.perf_counter()

    if not rate_samples.exists() or not rate_training.exists():
        raise FileNotFoundError(
            f"Rate-specific S1 training evidence is missing for {rate:.0%} run {run_number}.\n"
            f"Samples: {rate_samples}\nTraining: {rate_training}"
        )

    detector = phase2.s1_detector(
        THRESHOLD,
        samples_path=rate_samples,
        model_path=model_path,
        training_path=rate_training,
        clean_training_path=ROOT / "data" / "processed" / "split" / "clean_train.csv",
    )

    elapsed = time.perf_counter() - start

    # ADDED: Component 3 training-integrity evidence for this attack model.
    s1_bundle = load(model_path)
    training_monitor = phase2.training_anomaly_from_model(s1_bundle["model"])

    score = float(detector.get("risk_score", detector.get("score", 0.0)))
    decision = detector.get(
        "decision",
        "MALICIOUS" if score >= THRESHOLD else "CLEAN",
    )

    result = {
        "scenario": "S1_LABEL_FLIP",
        "poison_rate": float(rate),
        "run": int(run_number),
        "score": score,
        "risk_score": score,
        "s1_evidence": detector,
        "decision": decision,
        "detected": bool(score >= THRESHOLD),
        "detection_seconds": float(elapsed),
        "model_path": str(model_path),
        "model_source": model_source,
        "model_sha256": sha256_file(model_path),
        "samples_path": str(rate_samples),
        "samples_sha256": sha256_file(rate_samples),
        "threshold": THRESHOLD,
        "training_anomaly": training_monitor,
        "replicate_type": "independent_phase1_model" if model_source == "INDEPENDENT_RUN_MODEL" else "legacy_fallback",
    }

    print(
        f"  score={score:.6f} decision={decision} "
        f"time={elapsed:.4f}s"
    )
    return result


# ============================================================
# S2
# ============================================================

def load_rate_model_input(model_path, test_df):
    """Load a Phase 1 attack bundle and transform the untouched clean test set."""
    bundle = load(model_path)
    if not isinstance(bundle, dict) or "model" not in bundle or "scaler" not in bundle:
        raise ValueError(f"Invalid model bundle: {model_path}")

    model = bundle["model"]
    scaler = bundle["scaler"]
    features = bundle.get("features")

    if features is None:
        features = [c for c in test_df.columns if c not in {"label", "row_id"}]

    if "label" not in test_df.columns:
        raise ValueError("clean_test.csv must contain 'label'.")

    missing = [c for c in features if c not in test_df.columns]
    if missing:
        raise ValueError(
            f"Model requires missing test features: {missing[:10]}"
        )

    X_frame = test_df[features].replace([np.inf, -np.inf], np.nan).fillna(0).astype("float32")
    X = scaler.transform(X_frame)
    return model, X


def run_s2(rate, run_number, phase2, test_df):
    """
    Execute S2 backdoor detection using the EXISTING Phase 2 S2
    components directly.

    IMPORTANT:
    phase2_master.py does not define an s2_detector() function.
    Therefore this function calls the four real S2 functions exposed
    by phase2_master.py:

        1. neural_cleanse_style_score()
        2. strip_entropy_score()
        3. activation_clustering_score()
        4. fuse()

    This preserves the existing S2 detection methodology while allowing
    the experiment to evaluate each Phase 1 rate-specific model.
    """

    model_path, model_source = rate_run_model_path(ROOT, "S2_BACKDOOR", rate, run_number, )

    print(
        f"S2 | Rate {rate:.0%} | Run {run_number}"
    )

    start = time.perf_counter()

    # --------------------------------------------------------
    # Load the exact Phase 1 model for this poison rate.
    # --------------------------------------------------------

    model, X = load_rate_model_input(
        model_path,
        test_df,
    )

    # --------------------------------------------------------
    # Existing Phase 2 S2 detector components.
    # DO NOT replace these with a new detector.
    # --------------------------------------------------------

    neural_cleanse = phase2.neural_cleanse_style_score(
        model,
        X,
        target_label=1,
        steps=150,
        lr=0.05,
    )

    strip_score = phase2.strip_entropy_score(
        model,
        X,
        samples=20,
        noise_scale=0.05,
    )

    activation_score = phase2.activation_clustering_score(
        model,
        X,
        min_samples=20,
    )

    # --------------------------------------------------------
    # ADDED/FIXED:
    # The existing S2 component functions may return either:
    #   * a numeric score, OR
    #   * a dictionary containing the score/evidence.
    #
    # phase2_master.fuse() requires NUMERIC scores.
    # Therefore normalize each component BEFORE calling fuse().
    # --------------------------------------------------------

    def _numeric_component_score(value, component_name):
        if isinstance(value, dict):
            for key in (
                "risk_score",
                "score",
                "fusion_score",
                "anomaly_score",
            ):
                if key in value:
                    return float(value[key])

            raise ValueError(
                f"{component_name} returned a dictionary, but no "
                f"numeric score key was found. Keys: {list(value.keys())}"
            )

        if isinstance(value, (int, float, np.integer, np.floating)):
            return float(value)

        raise TypeError(
            f"{component_name} returned unsupported type: "
            f"{type(value).__name__}"
        )

    neural_cleanse_result = neural_cleanse
    strip_result = strip_score
    activation_result = activation_score

    neural_cleanse = _numeric_component_score(
        neural_cleanse_result,
        "Neural Cleanse",
    )

    strip_score = _numeric_component_score(
        strip_result,
        "STRIP",
    )

    activation_score = _numeric_component_score(
        activation_result,
        "Activation Clustering",
    )

    # --------------------------------------------------------
    # CRITICAL FIX:
    # fuse() receives numeric component scores, not dictionaries.
    # --------------------------------------------------------

    cfg = phase2.load_cfg()

    fusion = phase2.fuse(
        neural_cleanse,
        strip_score,
        activation_score,
        cfg,
    )

    elapsed = time.perf_counter() - start

    # ADDED: Component 3 training-integrity evidence.
    training_monitor = phase2.training_anomaly_from_model(model)

    # ADDED: real IBM ART ActivationDefence against the selected poisoned
    # training file, when ART is installed.
    art_result = {
        "component": "ART_ACTIVATION_CLUSTERING",
        "available": bool(getattr(phase2, "ART_AVAILABLE", False)),
        "status": "NOT_RUN",
        "score": 0.0,
    }
    rate_token = f"{int(round(rate * 100)):02d}"
    art_train_file = ROOT / "evidence" / "phase1" / "S2_backdoor" / f"backdoor_training_rate{rate_token}_run{int(run_number):02d}.csv"
    if art_train_file.exists() and getattr(phase2, "ART_AVAILABLE", False):
        art_df = pd.read_csv(art_train_file)
        if "label" in art_df.columns:
            art_X = art_df.drop(columns=["label", "row_id"], errors="ignore").to_numpy(dtype=np.float32)
            art_y = art_df["label"].to_numpy(dtype=int)
            art_result = phase2.art_activation_clustering_score(model, art_X, art_y)

    if isinstance(fusion, dict):
        fusion_score = float(
            fusion.get(
                "risk_score",
                fusion.get(
                    "score",
                    fusion.get("fusion_score", 0.0),
                ),
            )
        )
        decision = fusion.get(
            "decision",
            "MALICIOUS"
            if fusion_score >= THRESHOLD
            else "CLEAN",
        )
    else:
        fusion_score = float(fusion)
        decision = (
            "MALICIOUS"
            if fusion_score >= THRESHOLD
            else "CLEAN"
        )

    result = {
        "scenario": "S2_BACKDOOR",
        "poison_rate": rate,
        "run": run_number,

        "neural_cleanse_score": neural_cleanse,
        "neural_cleanse_evidence": neural_cleanse_result,
        "strip_score": strip_score,
        "strip_evidence": strip_result,
        "activation_clustering_score": activation_score,

        "fusion_score": fusion_score,
        "score": fusion_score,
        "risk_score": fusion_score,

        "decision": decision,
        "detected": bool(
            fusion_score >= THRESHOLD
        ),

        "detection_seconds": elapsed,

        "model_path": str(model_path),
        "model_source": model_source,
        "model_sha256": sha256_file(model_path),

        "threshold": THRESHOLD,
        "training_anomaly": training_monitor,
        "replicate_type": "independent_phase1_model" if model_source == "INDEPENDENT_RUN_MODEL" else "legacy_fallback",
        "art_activation_clustering": art_result,
    }

    print(
        f"  Neural Cleanse={neural_cleanse:.6f} "
        f"STRIP={strip_score:.6f} "
        f"Activation={activation_score:.6f}"
    )

    print(
        f"  fusion={fusion_score:.6f} "
        f"decision={decision} "
        f"ART_AC={art_result.get('score', 0.0):.6f} "
        f"time={elapsed:.4f}s"
    )

    return result


# ============================================================
# S3
# ============================================================

def run_s3(run_number, phase2):
    print(f"S3 | Run {run_number}")
    start = time.perf_counter()
    exact_dir = ROOT / "evidence" / "phase1" / "S3_pypi"
    trusted = exact_dir / f"trusted_package_run{int(run_number):02d}.txt"
    observed = exact_dir / f"controlled_modified_package_run{int(run_number):02d}.txt"
    if not trusted.exists() or not observed.exists():
        raise FileNotFoundError(f"Missing exact S3 repetition files for run {run_number}.")

    # phase2.s3_detector uses the historical fixed filenames; isolate this run.
    temp_root = Path(tempfile.mkdtemp(prefix=f"s3_run_{int(run_number):02d}_"))
    try:
        shutil.copy2(trusted, temp_root / "trusted_package.txt")
        shutil.copy2(observed, temp_root / "controlled_modified_package.txt")
        old_dir = phase2.S3_DIR
        phase2.S3_DIR = temp_root
        detector = phase2.s3_detector(THRESHOLD)
    finally:
        phase2.S3_DIR = old_dir
        shutil.rmtree(temp_root, ignore_errors=True)

    elapsed = time.perf_counter() - start
    score = float(detector.get("risk_score", detector.get("score", 0.0)))
    decision = detector.get("decision", "MALICIOUS" if score >= THRESHOLD else "CLEAN")
    return {
        "scenario": "S3_PYPI", "poison_rate": None, "run": run_number,
        "score": score, "risk_score": score, "decision": decision,
        "detected": bool(score >= THRESHOLD), "detection_seconds": elapsed,
        "trusted_path": str(trusted), "observed_path": str(observed),
        "trusted_sha256": sha256_file(trusted), "observed_sha256": sha256_file(observed),
        "threshold": THRESHOLD, "replicate_type": "independent_phase1_artifact",
    }


# ============================================================
# S4
# ============================================================

def run_s4(run_number, phase2):
    print(f"S4 | Run {run_number}")

    start = time.perf_counter()

    # FIXED: exact run-specific Phase 1 S4 path; no filename heuristics.
    artifact = (
        ROOT / "evidence" / "phase1" / "S4_model_hub"
        / f"controlled_model_run{int(run_number):02d}.pkl"
    )

    if not artifact.exists():
        raise FileNotFoundError(
            f"Expected S4 artifact not found:\n{artifact}"
        )

    detector = phase2.s4_detector(
        THRESHOLD,
        str(artifact)
    )

    elapsed = time.perf_counter() - start

    score = float(
        detector.get(
            "risk_score",
            detector.get("score", 0.0)
        )
    )

    decision = detector.get(
        "decision",
        "MALICIOUS"
        if score >= THRESHOLD
        else "CLEAN"
    )

    result = {
        "scenario": "S4_MODEL_HUB",
        "poison_rate": None,
        "run": run_number,
        "score": score,
        "risk_score": score,
        "decision": decision,
        "detected": bool(
            score >= THRESHOLD
        ),
        "detection_seconds": elapsed,
        "artifact_path": str(artifact),
        "artifact_source": "exact_phase1_repetition",
        "artifact_sha256": sha256_file(artifact),
        "threshold": THRESHOLD,
    }

    print(
        f"  score={score:.6f} "
        f"decision={decision} "
        f"time={elapsed:.4f}s"
    )

    return result


# ============================================================
# S5
# ============================================================

def run_s5(run_number, phase2):
    print(f"S5 | Run {run_number}")

    start = time.perf_counter()

    # Require the exact repetition-specific Phase 1 tampered artifact.
    artifact = ROOT / "evidence" / "phase1" / "S5_mlflow_tamper" / f"tampered_clean_baseline_run{int(run_number):02d}.joblib"
    if not artifact.exists():
        raise FileNotFoundError(f"Expected S5 artifact not found:\n{artifact}")
    detector = phase2.s5_detector(THRESHOLD, artifact_path=artifact)

    elapsed = time.perf_counter() - start

    score = float(
        detector.get(
            "risk_score",
            detector.get("score", 0.0)
        )
    )

    decision = detector.get(
        "decision",
        "MALICIOUS"
        if score >= THRESHOLD
        else "CLEAN"
    )

    result = {
        "scenario": "S5_MLFLOW_TAMPERING",
        "poison_rate": None,
        "run": run_number,
        "score": score,
        "risk_score": score,
        "decision": decision,
        "detected": bool(
            score >= THRESHOLD
        ),
        "detection_seconds": elapsed,
        "threshold": THRESHOLD,
    }

    print(
        f"  score={score:.6f} "
        f"decision={decision} "
        f"time={elapsed:.4f}s"
    )

    return result


# ============================================================
# MAIN
# ============================================================

def phase2_experiments_main():

    print()
    print("=" * 80)
    print("PHASE 2 — RATE-AWARE EXPERIMENTAL EVALUATION")
    print("=" * 80)
    print()
    print("Provisional global threshold:", THRESHOLD)
    print("NOTE: The decisions in this experiment log are provisional only.")
    print("      Final claims use the scenario-specific, held-out evaluation below.")
    print("Poison rates:", POISON_RATES)
    print("Runs per rate:", RUNS_PER_RATE)
    print()

    # --------------------------------------------------------
    # Verify the six actual Phase 1 files.
    # --------------------------------------------------------

    verify_rate_models()

    # --------------------------------------------------------
    # Load test data.
    # --------------------------------------------------------

    test_df = load_test_data()

    print(
        f"Clean test rows : {len(test_df):,}"
    )
    print(
        f"Features        : {len([c for c in test_df.columns if c not in {'label', 'row_id'}])}"
    )
    print()

    # --------------------------------------------------------
    # Import existing detector.
    # --------------------------------------------------------

    phase2 = load_phase2_module()

    # HARD GUARD: use only functions that actually exist in the authoritative
    # phase2_master.py. In particular, there is intentionally NO s2_detector().
    required = [
        "s1_detector",
        "s3_detector",
        "s4_detector",
        "s5_detector",
        "neural_cleanse_style_score",
        "strip_entropy_score",
        "activation_clustering_score",
        "fuse",
        "load_cfg",
    ]
    missing = [name for name in required if not hasattr(phase2, name)]
    if missing:
        raise RuntimeError(
            "Authoritative phase2_master.py is missing required functions: "
            + ", ".join(missing)
        )

    results = []

    # ========================================================
    # S1 — 3 poison rates × 5 repetitions
    # ========================================================

    print()
    print("-" * 80)
    print("S1 — LABEL FLIP — RATE-AWARE EVALUATION")
    print("-" * 80)

    for rate in POISON_RATES:

        print()
        print(
            f"S1 — {rate:.0%} POISON RATE — "
            f"{RUNS_PER_RATE} RUNS"
        )

        for run in range(
            1,
            RUNS_PER_RATE + 1
        ):
            result = run_s1(
                rate,
                run,
                phase2
            )

            results.append(result)

    # ========================================================
    # S2 — 3 poison rates × 5 repetitions
    # ========================================================

    print()
    print("-" * 80)
    print("S2 — BACKDOOR — RATE-AWARE EVALUATION")
    print("-" * 80)

    for rate in POISON_RATES:

        print()
        print(
            f"S2 — {rate:.0%} POISON RATE — "
            f"{RUNS_PER_RATE} RUNS"
        )

        for run in range(
            1,
            RUNS_PER_RATE + 1
        ):
            result = run_s2(
                rate,
                run,
                phase2,
                test_df,
            )

            results.append(result)

    # ========================================================
    # S3
    # ========================================================

    print()
    print("-" * 80)
    print("S3 — PYPI — 5 RUNS")
    print("-" * 80)

    for run in range(
        1,
        RUNS_OTHER_SCENARIOS + 1
    ):
        results.append(
            run_s3(
                run,
                phase2
            )
        )

    # ========================================================
    # S4
    # ========================================================

    print()
    print("-" * 80)
    print("S4 — MODEL HUB — 5 RUNS")
    print("-" * 80)

    for run in range(
        1,
        RUNS_OTHER_SCENARIOS + 1
    ):
        results.append(
            run_s4(
                run,
                phase2
            )
        )

    # ========================================================
    # S5
    # ========================================================

    print()
    print("-" * 80)
    print("S5 — MLFLOW TAMPERING — 5 RUNS")
    print("-" * 80)

    for run in range(
        1,
        RUNS_OTHER_SCENARIOS + 1
    ):
        results.append(
            run_s5(
                run,
                phase2
            )
        )

    # ========================================================
    # Validation
    # ========================================================

    expected_runs = (
        2
        * len(POISON_RATES)
        * RUNS_PER_RATE
        + 3
        * RUNS_OTHER_SCENARIOS
    )

    if len(results) != expected_runs:
        raise RuntimeError(
            f"Expected {expected_runs} runs, "
            f"but produced {len(results)}."
        )

    # ========================================================
    # Summary
    # ========================================================

    detected_count = sum(
        1
        for r in results
        if r["detected"]
    )

    detection_rate = (
        detected_count / len(results)
        if results
        else 0.0
    )

    # Rate-specific summary
    rate_summary = []

    for scenario in [
        "S1_LABEL_FLIP",
        "S2_BACKDOOR",
    ]:

        for rate in POISON_RATES:

            subset = [
                r
                for r in results
                if (
                    r["scenario"] == scenario
                    and r["poison_rate"] == rate
                )
            ]

            detected = sum(
                1
                for r in subset
                if r["detected"]
            )

            rate_summary.append(
                {
                    "scenario": scenario,
                    "poison_rate": rate,
                    "runs": len(subset),
                    "detected": detected,
                    "detection_rate": (
                        detected / len(subset)
                        if subset
                        else 0.0
                    ),
                    "mean_score": (
                        float(
                            np.mean(
                                [
                                    r["score"]
                                    for r in subset
                                ]
                            )
                        )
                        if subset
                        else 0.0
                    ),
                }
            )

    # ========================================================
    # Save complete results
    # ========================================================

    output = {
        "experiment": (
            "RATE_AWARE_PHASE2_EXPERIMENTAL_EVALUATION_45_RUNS"
        ),
        "threshold": THRESHOLD,

        "poison_rates": POISON_RATES,

        "runs_per_rate": RUNS_PER_RATE,

        "runs_other_scenarios":
            RUNS_OTHER_SCENARIOS,

        "total_runs":
            len(results),

        "detected_runs":
            detected_count,

        "overall_detection_rate":
            detection_rate,

        "expected_total_runs":
            expected_runs,

        "rate_summary":
            rate_summary,

        "experiments":
            results,
    }

    RESULT_FILE.write_text(
        json.dumps(
            output,
            indent=2
        ),
        encoding="utf-8"
    )

    # ========================================================
    # Save summary
    # ========================================================

    summary = {
        "total_runs": len(results),
        "detected_runs": detected_count,
        "overall_detection_rate":
            detection_rate,

        "scenario_counts": {
            scenario: sum(
                1
                for r in results
                if r["scenario"] == scenario
            )
            for scenario in [
                "S1_LABEL_FLIP",
                "S2_BACKDOOR",
                "S3_PYPI",
                "S4_MODEL_HUB",
                "S5_MLFLOW_TAMPERING",
            ]
        },

        "rate_summary":
            rate_summary,

        "threshold":
            THRESHOLD,
    }

    SUMMARY_FILE.write_text(
        json.dumps(
            summary,
            indent=2
        ),
        encoding="utf-8"
    )

    # ========================================================
    # Final output
    # ========================================================

    print()
    print("=" * 80)
    print("RATE-AWARE EXPERIMENTS COMPLETE — PROVISIONAL GLOBAL-THRESHOLD LOG")
    print("=" * 80)

    print()
    print(
        f"Total runs       : {len(results)}"
    )

    print(
        f"Provisional detected runs : {detected_count}"
    )

    print(
        f"Provisional detection rate: {detection_rate:.6f}"
    )

    print()
    print("Provisional rate-specific detection (not final calibrated results):")
    print()

    for item in rate_summary:

        print(
            f"{item['scenario']:20s} "
            f"{item['poison_rate']:.0%} "
            f"runs={item['runs']} "
            f"detected={item['detected']} "
            f"rate={item['detection_rate']:.4f} "
            f"mean_score={item['mean_score']:.6f}"
        )

    print()
    print("Results:")
    print(RESULT_FILE)

    print()
    print("Summary:")
    print(SUMMARY_FILE)

    print()
    print("STATUS: PASSED")





# ===== EMBEDDED phase2_clean_controls(3).py =====
# ============================================================
# PHASE 2 — CLEAN CONTROL GENERATION (FIXED)
#
# PURPOSE:
# Generate genuine clean-control scores for S1-S5.
#
# These controls are NOT fabricated.
# They are measured using the clean baseline model,
# clean test data, and clean reference artefacts.
#
# FIXES IN THIS VERSION:
#   S1 previously never called the real s1_detector() — it just
#   hardcoded risk_score = 0.0 after checking a "label" column
#   existed. Now builds a genuinely unflipped samples file
#   (original_label == poisoned_label for every row, sampled from
#   real clean training data) and calls the actual s1_detector()
#   against it, re-evaluating the CLEAN baseline model's accuracy
#   as the model-side signal (there is no "clean-trained S1 model"
#   to use, so the clean baseline is the correct honest substitute).
#
#   S3 previously hashed one file against ITSELF (same path read
#   twice), which is tautologically always a match and never
#   exercises real file-comparison logic. Now writes two separate
#   files with identical content under the exact filenames
#   s3_detector() expects, and calls the actual s3_detector().
#
#   S5 previously hashed CLEAN_MODEL_PATH against itself the same
#   way. Now writes a genuine untampered byte-for-byte copy under
#   a separate path and calls the actual s5_detector().
#
# All three now delegate to the REAL functions from phase2_master
# via temporary, restored monkeypatching of that module's path
# constants — so this script tests the exact same code path your
# dissertation results depend on, not a simplified reimplementation.
#
# OUTPUT:
# evidence/phase2/clean_controls/clean_control_scores.json
# ============================================================

import json
import pickle
import shutil
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load

pm = sys.modules[__name__]



# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent

CLEAN_MODEL_PATH = (
    ROOT
    / "models"
    / "clean"
    / "clean_baseline.joblib"
)

CLEAN_TRAIN_PATH = (
    ROOT
    / "data"
    / "processed"
    / "split"
    / "clean_train.csv"
)

CLEAN_TEST_PATH = (
    ROOT
    / "data"
    / "processed"
    / "split"
    / "clean_test.csv"
)

OUTPUT_DIR = (
    ROOT
    / "evidence"
    / "phase2"
    / "clean_controls"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "clean_control_scores.json"
)

# Number of rows to sample for the S1 clean-control "unflipped"
# evidence file — large enough for a stable recomputed_flip_rate.
S1_CONTROL_SAMPLE_SIZE = 3000

S1_CONTROL_RANDOM_STATE = 4242
CONTROL_REPETITIONS = 5


# ============================================================
# HELPER
# ============================================================

def decision(score, threshold):
    return (
        "MALICIOUS"
        if score >= threshold
        else "CLEAN"
    )


@contextmanager
def patched_module_paths(**overrides):
    """
    Temporarily overrides module-level path constants on the
    imported phase2_master module (pm), so its REAL detector
    functions (which read those constants as globals, not
    parameters) operate against clean-control artefacts instead
    of the real Phase 1 attack evidence. Always restores the
    original values afterward, even if the detector call raises.
    """

    originals = {}

    for name, value in overrides.items():

        originals[name] = getattr(pm, name)

        setattr(pm, name, value)

    try:

        yield

    finally:

        for name, value in originals.items():

            setattr(pm, name, value)


# ============================================================
# S1 — CLEAN LABEL CONTROL   *** FIXED: calls real s1_detector ***
# ============================================================

def build_s1_clean_evidence(repetition=1):
    """
    Builds a genuinely unflipped label_flip_samples.csv — sampled
    from real clean training data, with original_label and
    poisoned_label set IDENTICALLY (0% flip rate) — so the real
    s1_detector() has authentic clean evidence to recompute
    against, instead of a hardcoded score.
    """

    control_dir = (
        OUTPUT_DIR / "s1_reference"
    )

    control_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    train_df = pd.read_csv(
        CLEAN_TRAIN_PATH
    )

    sample_size = min(
        S1_CONTROL_SAMPLE_SIZE,
        len(train_df)
    )

    sample = train_df.sample(
        n=sample_size,
        random_state=S1_CONTROL_RANDOM_STATE + int(repetition)
    ).copy()

    sample["original_label"] = sample["label"]

    # No flip: poisoned_label == original_label for every row.
    sample["poisoned_label"] = sample["label"]

    samples_file = (
        control_dir / "label_flip_samples.csv"
    )

    sample.to_csv(
        samples_file,
        index=False
    )

    return control_dir


def clean_s1(threshold, repetition=1):
    """
    Runs the REAL s1_detector() against genuinely unflipped
    evidence and the clean baseline model (re-evaluated for
    accuracy, since there is no separately-trained "clean S1
    model" — the clean baseline is the correct honest stand-in).
    """

    control_dir = build_s1_clean_evidence(repetition)

    with patched_module_paths(
        S1_DIR=control_dir,
        S1_MODEL_PATH=CLEAN_MODEL_PATH,
    ):

        raw_result = s1_detector(
            threshold,
            samples_path=control_dir / "label_flip_samples.csv",
            model_path=CLEAN_MODEL_PATH,
            training_path=CLEAN_TRAIN_PATH,
            clean_training_path=CLEAN_TRAIN_PATH,
        )

    result = {
        "scenario": "S1_LABEL_FLIP",
        "control_type": "CLEAN",
        "verification": (
            "real_s1_detector_against_unflipped_sample"
        ),
        **raw_result,
    }

    return result


# ============================================================
# S2 — CLEAN BACKDOOR CONTROL   (already correct — unchanged)
# ============================================================

def clean_s2(
    model,
    X,
    cfg,
    threshold
):

    """
    Run the ACTUAL Phase 2 backdoor detectors against
    the clean baseline model.

    This is the most important clean control because
    S2 uses statistical/model-behaviour detectors.
    """

    nc = neural_cleanse_style_score(
        model,
        X,
        cfg["attacks"]["target_label"],
        cfg["detection"]["nc_steps"],
        cfg["detection"]["nc_learning_rate"],
    )

    strip = strip_entropy_score(
        model,
        X,
        cfg["detection"]["strip_samples"],
    )

    cluster = activation_clustering_score(
        model,
        X,
        cfg["detection"]["cluster_min_samples"],
    )

    fusion = fuse(
        nc["score"],
        strip["score"],
        cluster["score"],
        cfg,
    )

    result = {
        "scenario": "S2_BACKDOOR",
        "control_type": "CLEAN",

        "neural_cleanse": nc,
        "strip": strip,
        "activation_clustering": cluster,
        "fusion": fusion,

        "risk_score": float(
            fusion["risk_score"]
        ),

        "threshold": threshold,

        "decision": decision(
            fusion["risk_score"],
            threshold
        ),

        "model": str(
            CLEAN_MODEL_PATH
        ),

        "dataset": str(
            CLEAN_TEST_PATH
        ),
    }

    return result


# ============================================================
# S3 — CLEAN PYPI CONTROL   *** FIXED: calls real s3_detector ***
# ============================================================

def build_s3_clean_evidence():
    """
    Writes TWO SEPARATE files with identical content, under the
    exact filenames s3_detector() expects (trusted_package.txt /
    controlled_modified_package.txt). Previously the same single
    file was hashed against itself, which is tautologically always
    a match and never actually exercised two-file comparison
    logic. Two genuinely distinct files with matching content is
    the correct honest representation of "no compromise occurred."
    """

    control_dir = (
        OUTPUT_DIR / "s3_reference"
    )

    control_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    content = (
        "clean-package==1.0.0\n"
        "CONTROLLED_RESEARCH_CLEAN_REFERENCE\n"
    )

    trusted_file = (
        control_dir / "trusted_package.txt"
    )

    observed_file = (
        control_dir / "controlled_modified_package.txt"
    )

    trusted_file.write_text(
        content,
        encoding="utf-8"
    )

    observed_file.write_text(
        content,
        encoding="utf-8"
    )

    return control_dir


def clean_s3(threshold):

    control_dir = build_s3_clean_evidence()

    with patched_module_paths(
        S3_DIR=control_dir,
    ):

        raw_result = s3_detector(threshold)

    result = {
        "scenario": "S3_PYPI",
        "control_type": "CLEAN",
        "verification_source": (
            "real_s3_detector_against_two_matching_files"
        ),
        **raw_result,
    }

    return result


# ============================================================
# S4 — CLEAN MODEL-HUB CONTROL   (already correct — unchanged)
# ============================================================

def create_clean_pickle():

    """
    Create a harmless Pickle artefact.

    IMPORTANT:
    This file is NEVER loaded by the detector.

    It is created only so the static scanner has a genuine
    clean artefact to inspect.
    """

    clean_pickle = (
        OUTPUT_DIR
        / "clean_model_hub_control.pkl"
    )

    clean_object = {
        "framework": "research_clean_control",
        "status": "verified-clean",
        "purpose": "static scanner control",
    }

    with open(
        clean_pickle,
        "wb"
    ) as f:
        pickle.dump(
            clean_object,
            f,
            protocol=4
        )

    return clean_pickle


def clean_s4(threshold):

    """
    Run the REAL Phase 2 Pickle scanner against a
    harmless control Pickle.
    """

    artifact = create_clean_pickle()

    scan = recursive_pickle_scan(
        artifact
    )

    format_policy = model_format_policy(
        artifact
    )

    dangerous_count = int(
        scan.get(
            "dangerous_global_count",
            0
        )
    )

    global_count = int(
        scan.get(
            "global_operation_count",
            0
        )
    )

    if dangerous_count > 0:
        score = 1.0

    elif global_count > 0:
        score = 0.40

    else:
        score = 0.0

    return {
        "scenario": "S4_MODEL_HUB",
        "control_type": "CLEAN",

        "risk_score": score,
        "threshold": threshold,

        "decision": decision(
            score,
            threshold
        ),

        "artifact": str(
            artifact
        ),

        "sha256": sha256_file(
            artifact
        ),

        "pickle_scan": scan,

        "format_policy": format_policy,

        "dangerous_globals": scan.get(
            "dangerous_globals_found",
            []
        ),

        "verification": (
            "static_analysis_of_clean_control"
        ),
    }


# ============================================================
# S5 — CLEAN MLFLOW TAMPERING CONTROL  *** FIXED: real s5_detector ***
# ============================================================

def build_s5_clean_evidence():
    """
    Writes a genuine BYTE-FOR-BYTE COPY of the clean baseline
    model under a separate path, in the exact location
    s5_detector() expects (tampered_clean_baseline.joblib inside
    S5_DIR). Previously CLEAN_MODEL_PATH was hashed against
    itself, which is tautologically always a match. A real copy
    at a distinct path is the correct honest representation of
    "the artefact currently in the registry is untampered."
    """

    control_dir = (
        OUTPUT_DIR / "s5_reference"
    )

    control_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    untampered_copy = (
        control_dir / "tampered_clean_baseline.joblib"
    )

    shutil.copy2(
        CLEAN_MODEL_PATH,
        untampered_copy
    )

    return control_dir


def clean_s5(threshold):

    control_dir = build_s5_clean_evidence()

    with patched_module_paths(
        S5_DIR=control_dir,
    ):

        raw_result = s5_detector(threshold)

    result = {
        "scenario": "S5_MLFLOW_TAMPERING",
        "control_type": "CLEAN",
        "verification_source": (
            "real_s5_detector_against_untampered_copy"
        ),
        **raw_result,
    }

    return result


# ============================================================
# MAIN
# ============================================================

def phase2_clean_controls_main():
    print("\n" + "=" * 80)
    print("PHASE 2 — CLEAN CONTROL GENERATION")
    print("=" * 80)

    if not CLEAN_MODEL_PATH.exists():
        raise FileNotFoundError(f"Clean model not found:\n{CLEAN_MODEL_PATH}")
    if not CLEAN_TRAIN_PATH.exists() or not CLEAN_TEST_PATH.exists():
        raise FileNotFoundError("Clean train/test data are required for controls.")

    cfg = load_cfg()
    threshold = float(cfg["detection"]["risk_threshold"])
    print(f"\nDetection threshold: {threshold}")
    print(f"Clean-control repetitions: {CONTROL_REPETITIONS}")

    bundle = load(CLEAN_MODEL_PATH)
    model = bundle["model"]
    scaler = bundle["scaler"]
    features = bundle.get("features") or [c for c in pd.read_csv(CLEAN_TEST_PATH, nrows=1).columns if c not in {"label", "row_id"}]
    clean_test = pd.read_csv(CLEAN_TEST_PATH)
    X_test_frame = clean_test[features].replace([np.inf, -np.inf], np.nan).fillna(0).astype("float32")
    X_test_scaled = scaler.transform(X_test_frame)

    experiments = []

    for repetition in range(1, CONTROL_REPETITIONS + 1):
        print("\n" + "-" * 80)
        print(f"CLEAN CONTROL REPETITION {repetition}/{CONTROL_REPETITIONS}")
        print("-" * 80)

        # Bootstrap the clean test observations for the sample-dependent S2 detectors.
        rng = np.random.default_rng(9000 + repetition)
        indices = rng.choice(len(X_test_scaled), size=max(20, int(len(X_test_scaled) * 0.60)), replace=True)
        X_s2 = X_test_scaled[indices]
        s2 = clean_s2(model, X_s2, cfg, threshold)

        s1 = clean_s1(threshold, repetition)
        s3 = clean_s3(threshold)
        s4 = clean_s4(threshold)
        s5 = clean_s5(threshold)

        for scenario, item in (
            ("S1_LABEL_FLIP", s1),
            ("S2_BACKDOOR", s2),
            ("S3_PYPI", s3),
            ("S4_MODEL_HUB", s4),
            ("S5_MLFLOW_TAMPERING", s5),
        ):
            experiments.append({
                **item,
                "scenario": scenario,
                "repetition": repetition,
                "run": repetition,
                "control_type": "clean_control",
            })

    output = {
        "phase": "CODE_PHASE_2",
        "threshold": threshold,
        "control_repetitions": CONTROL_REPETITIONS,
        "experiments": experiments,
        "interpretation": {
            "controls_are_clean": True,
            "threshold_selection_use": "calibration subset only",
            "s2_repetition_method": "bootstrap-resampled clean test observations",
            "integrity_control_note": "S1/S3/S4/S5 are expected to be deterministic clean controls; repetition is verification, not independent attack generation.",
        },
    }

    OUTPUT_FILE.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")

    print("\n" + "=" * 80)
    print("CLEAN CONTROL SUMMARY")
    print("=" * 80)
    for item in experiments:
        score = float(item.get("risk_score", item.get("score", item.get("fusion_score", 0.0))))
        print(f"{item['scenario']:22s} run={item['run']} score={score:.6f} decision={item.get('decision', 'CLEAN')}")

    print(f"\nSaved:\n{OUTPUT_FILE}")
    print("STATUS: PASSED")





# ===== EMBEDDED phase2_final_evaluation(3).py =====
from pathlib import Path
import json

import numpy as np
import pandas as pd


# ============================================================
# PHASE 2 — FINAL RESEARCH EVALUATION
#
# ADDED:
#   - poison-rate detection breakdown for S1/S2
#   - results/phase2/detection_rate_by_poison_rate.csv
#
# Existing pooled threshold evaluation is preserved.
# ============================================================

ROOT = Path(__file__).resolve().parent

EVIDENCE_DIR = ROOT / "evidence" / "phase2"
RESULT_DIR = ROOT / "results" / "phase2"
FINAL_DIR = EVIDENCE_DIR / "final_evaluation"

RESULT_DIR.mkdir(parents=True, exist_ok=True)
FINAL_DIR.mkdir(parents=True, exist_ok=True)

ATTACK_FILE = (
    EVIDENCE_DIR
    / "25_experiments"
    / "25_experiment_results.json"
)

# Compatibility with the path used by the existing 25-experiment script.
if not ATTACK_FILE.exists():
    ATTACK_FILE = (
        EVIDENCE_DIR
        / "25_experiments"
        / "25_experiment_results.json"
    )

CLEAN_FILE = (
    EVIDENCE_DIR
    / "clean_controls"
    / "clean_control_scores.json"
)

FINAL_JSON = FINAL_DIR / "phase2_final_evaluation.json"
THRESHOLD_CSV = RESULT_DIR / "phase2_threshold_analysis.csv"
SCENARIO_CSV = RESULT_DIR / "phase2_scenario_statistics.csv"
METRICS_CSV = RESULT_DIR / "phase2_final_metrics.csv"

# ADDED: requested poison-rate result.
POISON_RATE_CSV = RESULT_DIR / "detection_rate_by_poison_rate.csv"

THRESHOLDS = [
    0.40, 0.45, 0.50, 0.55, 0.60,
    0.65, 0.70, 0.75, 0.80, 0.85, 0.90,
]


def load_json(path):
    if not path.exists():
        raise FileNotFoundError(f"Required file not found:\n{path}")
    return json.loads(path.read_text(encoding="utf-8"))


def normalise_attack_results(data):
    if isinstance(data, dict):
        experiments = data.get("experiments", [])

        # ADDED/FIXED: tolerate dict-shaped experiment collections.
        if isinstance(experiments, dict):
            experiments = [
                dict(value, scenario=key)
                if isinstance(value, dict) else value
                for key, value in experiments.items()
            ]

        if not experiments:
            for key in ("results", "runs", "attack_results"):
                candidate = data.get(key)
                if candidate:
                    experiments = candidate
                    break

    elif isinstance(data, list):
        experiments = data
    else:
        raise ValueError("Attack results must be a JSON list/dict.")

    normalised = []

    for item in experiments:
        scenario = item.get("scenario")
        score = item.get(
            "score",
            item.get(
                "fusion_score",
                item.get("risk_score"),
            ),
        )

        if score is None:
            raise ValueError(
                f"Missing detection score in experiment: {item}"
            )

        # ADDED/FIXED: accept poison_rate from the experiment record
        # or nested metadata/config without inventing missing values.
        poison_rate = item.get("poison_rate")

        if poison_rate is None:
            metadata = item.get("metadata", {})
            if isinstance(metadata, dict):
                poison_rate = metadata.get("poison_rate")

        if poison_rate is None:
            config = item.get("config", {})
            if isinstance(config, dict):
                poison_rate = config.get("poison_rate")

        normalised.append({
            "scenario": scenario,
            "score": float(score),
            "decision": item.get("decision"),
            "poison_rate": (
                float(poison_rate)
                if poison_rate is not None
                else None
            ),
            "run": item.get("run", item.get("run_number")),
        })

    if not normalised:
        raise ValueError("No attack experiments found.")

    return normalised


def normalise_clean_controls(data):
    controls = []

    if isinstance(data, dict):
        source = data.get("controls", data.get("results", []))

        # FIXED: actual clean-control output may use:
        # {"experiments": {"S1_LABEL_FLIP": {"risk_score": ...}, ...}}
        if not source:
            experiments = data.get("experiments", [])
            if isinstance(experiments, list):
                source = experiments

            elif isinstance(experiments, dict):
                source = [
                    dict(v, scenario=k)
                    for k, v in experiments.items()
                    if isinstance(v, dict)
                    and (
                        "risk_score" in v
                        or "score" in v
                        or "fusion_score" in v
                    )
                ]

        # Compatibility with a flat mapping:
        # {"S1_LABEL_FLIP": {"score": ...}, ...}
        if not source and any(
            isinstance(v, dict)
            and (
                "score" in v
                or "risk_score" in v
                or "fusion_score" in v
            )
            for v in data.values()
        ):
            source = [
                dict(v, scenario=k)
                for k, v in data.items()
                if isinstance(v, dict)
                and (
                    "score" in v
                    or "risk_score" in v
                    or "fusion_score" in v
                )
            ]
    elif isinstance(data, list):
        source = data
    else:
        raise ValueError("Clean-control results must be a JSON list/dict.")

    for item in source:
        if not isinstance(item, dict):
            continue

        scenario = item.get("scenario")
        score = item.get(
            "score",
            item.get("fusion_score", item.get("risk_score")),
        )

        if scenario is None or score is None:
            continue

        controls.append({
            "scenario": scenario,
            "score": float(score),
            "run": item.get("run", item.get("repetition")),
        })

    if not controls:
        raise ValueError("No clean-control scores found.")

    return controls


def confusion(scores, labels, threshold):
    predictions = [score >= threshold for score in scores]

    tp = sum(p and y for p, y in zip(predictions, labels))
    tn = sum((not p) and (not y) for p, y in zip(predictions, labels))
    fp = sum(p and (not y) for p, y in zip(predictions, labels))
    fn = sum((not p) and y for p, y in zip(predictions, labels))

    total = tp + tn + fp + fn

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )
    fpr = fp / (fp + tn) if fp + tn else 0.0
    accuracy = (tp + tn) / total if total else 0.0

    return {
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
    }


def scenario_statistics(attack_results, thresholds):
    rows = []

    for scenario in sorted(
        {r["scenario"] for r in attack_results}
    ):
        subset = [
            r for r in attack_results
            if r["scenario"] == scenario
        ]

        scores = [r["score"] for r in subset]
        threshold = float(thresholds[scenario])
        detected = sum(s >= threshold for s in scores)

        rows.append({
            "scenario": scenario,
            "runs": len(subset),
            "detected": detected,
            "detection_rate": detected / len(subset),
            "mean_score": float(np.mean(scores)),
            "std_score": float(np.std(scores, ddof=0)),
            "operating_threshold": threshold,
        })

    return rows


# ============================================================
# ADDED — POISON-RATE BREAKDOWN
# ============================================================

def poison_rate_breakdown(attack_results, thresholds):
    rows = []

    for scenario in ("S1", "S1_LABEL_FLIP", "S2", "S2_BACKDOOR"):
        subset = [
            r for r in attack_results
            if r["scenario"] == scenario
            and r["poison_rate"] is not None
        ]

        if not subset:
            continue

        for rate in sorted(
            {float(r["poison_rate"]) for r in subset}
        ):
            rate_runs = [
                r for r in subset
                if abs(float(r["poison_rate"]) - rate) < 1e-12
            ]

            detected = sum(
                r["score"] >= float(thresholds[r["scenario"]])
                for r in rate_runs
            )

            rows.append({
                "scenario": scenario,
                "poison_rate": rate,
                "operating_threshold": float(thresholds[scenario]),
                "runs": len(rate_runs),
                "detected": detected,
                "detection_rate": (
                    detected / len(rate_runs)
                    if rate_runs else 0.0
                ),
            })

    # If the existing 25-run file has no poison_rate metadata,
    # fail safely rather than inventing rates.
    if not rows:
        print(
            "\nWARNING: 25-experiment results contain no poison_rate "
            "metadata for S1/S2."
        )
        print(
            "Run the updated 25_experiments.py before using the "
            "poison-rate breakdown."
        )

    return rows




def split_calibration_holdout(rows, calibration_runs=(1, 2, 3)):
    """Split repeated runs so threshold selection never uses held-out runs."""
    calibration = []
    holdout = []
    for row in rows:
        run = row.get("run")
        try:
            run_i = int(run)
        except (TypeError, ValueError):
            raise ValueError("Every experimental record must contain an integer run for leakage-safe threshold selection.")
        (calibration if run_i in calibration_runs else holdout).append(row)
    if not calibration or not holdout:
        raise ValueError(
            f"Calibration/holdout split failed: calibration={len(calibration)}, holdout={len(holdout)}."
        )
    return calibration, holdout


def build_labeled(attack_results, clean_controls):
    rows = []
    for r in attack_results:
        rows.append({"score": float(r["score"]), "label": True, **{k: r.get(k) for k in ("scenario", "poison_rate", "run")}})
    for r in clean_controls:
        rows.append({"score": float(r["score"]), "label": False, **{k: r.get(k) for k in ("scenario", "poison_rate", "repetition")}})
    return rows


INTEGRITY_GATE_SCENARIOS = {"S3_PYPI", "S4_MODEL_HUB", "S5_MLFLOW_TAMPERING"}
CALIBRATION_FPR_TARGET = 0.05


def select_scenario_threshold(calibration_attack, calibration_clean, scenario):
    """Select one threshold per detector without using held-out runs.

    S1 and S2 emit different continuous statistics, so thresholds are never
    shared.  S3--S5 are deterministic integrity gates (0=match, 1=violation)
    and retain their fixed, semantically meaningful threshold of 0.5.
    """
    if scenario in INTEGRITY_GATE_SCENARIOS:
        threshold = 0.5
        scores = [r["score"] for r in calibration_attack] + [r["score"] for r in calibration_clean]
        labels = [True] * len(calibration_attack) + [False] * len(calibration_clean)
        return threshold, {"threshold": threshold, "selection": "fixed_binary_integrity_gate", **confusion(scores, labels, threshold)}

    clean_scores = np.asarray([r["score"] for r in calibration_clean], dtype=float)
    attack_scores = np.asarray([r["score"] for r in calibration_attack], dtype=float)
    if len(clean_scores) == 0 or len(attack_scores) == 0:
        raise ValueError(f"Scenario {scenario} has no calibration attack or clean-control scores.")

    # Evaluate every decision boundary visible in calibration, including the
    # boundary immediately above each clean score.  This gives a reproducible
    # clean-FPR-constrained choice without inventing a cross-detector scale.
    candidates = sorted({
        0.0,
        *[float(x) for x in attack_scores],
        *[float(np.nextafter(x, np.inf)) for x in clean_scores],
    })
    scores = list(attack_scores) + list(clean_scores)
    labels = [True] * len(attack_scores) + [False] * len(clean_scores)
    analyses = [{"threshold": float(t), **confusion(scores, labels, float(t))} for t in candidates]
    allowed = [row for row in analyses if row["fpr"] <= CALIBRATION_FPR_TARGET]
    pool = allowed if allowed else analyses
    selected = max(pool, key=lambda row: (row["recall"], row["f1"], -row["fpr"], row["threshold"]))
    selected = dict(selected)
    selected["selection"] = "per_scenario_calibration_fpr_constrained"
    selected["calibration_fpr_target"] = CALIBRATION_FPR_TARGET
    selected["candidate_count"] = len(analyses)
    return float(selected["threshold"]), selected


def per_scenario_evaluation(calibration_attack, calibration_clean, holdout_attack, holdout_clean):
    scenarios = sorted({r["scenario"] for r in calibration_attack + calibration_clean})
    thresholds = {}
    threshold_rows = []
    scenario_rows = []
    all_holdout_scores = []
    all_holdout_labels = []
    all_holdout_rows = []

    for scenario in scenarios:
        cal_attack = [r for r in calibration_attack if r["scenario"] == scenario]
        cal_clean = [r for r in calibration_clean if r["scenario"] == scenario]
        test_attack = [r for r in holdout_attack if r["scenario"] == scenario]
        test_clean = [r for r in holdout_clean if r["scenario"] == scenario]
        if not cal_attack or not cal_clean or not test_attack or not test_clean:
            raise ValueError(f"Scenario {scenario} is missing calibration or held-out repetitions.")

        threshold, calibration = select_scenario_threshold(cal_attack, cal_clean, scenario)
        thresholds[scenario] = threshold
        holdout_scores = [r["score"] for r in test_attack] + [r["score"] for r in test_clean]
        holdout_labels = [True] * len(test_attack) + [False] * len(test_clean)
        heldout = confusion(holdout_scores, holdout_labels, threshold)
        threshold_rows.append({"scenario": scenario, **calibration})
        scenario_rows.append({
            "scenario": scenario,
            "operating_threshold": threshold,
            "threshold_selection": calibration["selection"],
            "calibration_fpr": calibration["fpr"],
            "heldout_runs_attack": len(test_attack),
            "heldout_runs_clean": len(test_clean),
            **{f"heldout_{k.lower()}": v for k, v in heldout.items()},
        })
        all_holdout_scores.extend(holdout_scores)
        all_holdout_labels.extend(holdout_labels)
        all_holdout_rows.extend(test_attack + test_clean)

    return thresholds, threshold_rows, scenario_rows, all_holdout_rows, all_holdout_scores, all_holdout_labels



def phase2_final_evaluation_main():
    print("=" * 80)
    print("PHASE 2 — FINAL RESEARCH EVALUATION")
    print("=" * 80)

    attack_data = load_json(ATTACK_FILE)
    clean_data = load_json(CLEAN_FILE)

    attack_results = normalise_attack_results(attack_data)
    clean_controls = normalise_clean_controls(clean_data)

    print(f"\nAttack runs loaded : {len(attack_results)}")
    print(f"Clean controls loaded : {len(clean_controls)}")

    # FIXED: leakage-safe threshold selection.
    # First 3 repetitions calibrate the threshold; repetitions 4-5 are held out
    # for the final reported performance. The held-out set is never used to choose
    # the threshold.
    calibration_attack, holdout_attack = split_calibration_holdout(attack_results)
    calibration_clean, holdout_clean = split_calibration_holdout(clean_controls)

    thresholds, threshold_rows, scenario_rows, holdout_rows, holdout_scores, holdout_labels = per_scenario_evaluation(
        calibration_attack, calibration_clean, holdout_attack, holdout_clean
    )
    # Aggregate only after applying each row's own scenario-specific threshold.
    predictions = [
        score >= thresholds[row["scenario"]]
        for row, score in zip(holdout_rows, holdout_scores)
    ]
    tp = sum(p and y for p, y in zip(predictions, holdout_labels))
    tn = sum((not p) and (not y) for p, y in zip(predictions, holdout_labels))
    fp = sum(p and (not y) for p, y in zip(predictions, holdout_labels))
    fn = sum((not p) and y for p, y in zip(predictions, holdout_labels))
    heldout_metrics = confusion([1.0 if p else 0.0 for p in predictions], holdout_labels, 0.5)

    print("\n" + "=" * 80)
    print("SCENARIO-SPECIFIC OPERATING THRESHOLDS (CALIBRATION SET)")
    print("=" * 80)
    for row in threshold_rows:
        print(f"{row['scenario']:22s} threshold={row['threshold']:.8f} calibration_FPR={row['fpr']:.6f} selection={row['selection']}")
    print("\nHELD-OUT FINAL METRICS")
    print(f"TP        : {heldout_metrics['TP']}")
    print(f"TN        : {heldout_metrics['TN']}")
    print(f"FP        : {heldout_metrics['FP']}")
    print(f"FN        : {heldout_metrics['FN']}")
    print(f"Accuracy  : {heldout_metrics['accuracy']:.6f}")
    print(f"Precision : {heldout_metrics['precision']:.6f}")
    print(f"Recall    : {heldout_metrics['recall']:.6f}")
    print(f"F1        : {heldout_metrics['f1']:.6f}")
    print(f"FPR       : {heldout_metrics['fpr']:.6f}")

    poison_rows = poison_rate_breakdown(holdout_attack, thresholds)

    pd.DataFrame(threshold_rows).to_csv(
        THRESHOLD_CSV,
        index=False,
    )

    pd.DataFrame(scenario_rows).to_csv(
        SCENARIO_CSV,
        index=False,
    )

    pd.DataFrame([{
        "operating_thresholds": json.dumps(thresholds, sort_keys=True),
        "TP": heldout_metrics["TP"],
        "TN": heldout_metrics["TN"],
        "FP": heldout_metrics["FP"],
        "FN": heldout_metrics["FN"],
        "accuracy": heldout_metrics["accuracy"],
        "precision": heldout_metrics["precision"],
        "recall": heldout_metrics["recall"],
        "f1": heldout_metrics["f1"],
        "fpr": heldout_metrics["fpr"],
    }]).to_csv(
        METRICS_CSV,
        index=False,
    )

    # ADDED: requested detection-rate-by-poison-rate table.
    pd.DataFrame(
        poison_rows,
        columns=[
            "scenario",
            "poison_rate",
            "runs",
            "detected",
            "detection_rate",
        ],
    ).to_csv(
        POISON_RATE_CSV,
        index=False,
    )

    final = {
        "phase": "CODE_PHASE_2",
        "attack_runs": len(attack_results),
        "clean_controls": len(clean_controls),
        "operating_thresholds": thresholds,
        "metrics": heldout_metrics,
        "calibration_metrics_by_scenario": threshold_rows,
        "evaluation_protocol": {
            "threshold_selection": "first 3 repetitions, separately for each scenario; calibration FPR constrained to 0.05",
            "final_evaluation": "held-out repetitions 4-5",
            "no_test_set_threshold_tuning": True,
            "strip_policy": "STRIP raw entropy is retained as diagnostic evidence and excluded from operational fusion until independently validated against clean controls.",
        },
        "threshold_analysis": threshold_rows,
        "scenario_statistics": scenario_rows,
        "detection_rate_by_poison_rate": poison_rows,
        "poison_rate_analysis_status": (
            "AVAILABLE" if poison_rows
            else "NOT_AVAILABLE_IN_INPUT_RESULTS"
        ),
        "files": {
            "threshold_csv": str(THRESHOLD_CSV),
            "scenario_csv": str(SCENARIO_CSV),
            "metrics_csv": str(METRICS_CSV),
            "poison_rate_csv": str(POISON_RATE_CSV),
        },
    }

    FINAL_JSON.write_text(
        json.dumps(final, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 80)
    print("PHASE 2 FINAL EVALUATION COMPLETE")
    print("=" * 80)
    print(f"Operating thresholds: {json.dumps(thresholds, sort_keys=True)}")
    print(f"Final evidence     : {FINAL_JSON}")
    print(f"Poison-rate CSV    : {POISON_RATE_CSV}")

    if not poison_rows:
        print(
            "\nNOTE: poison-rate CSV is empty because the existing "
            "25-run file does not contain poison_rate metadata."
        )
        print(
            "This is deliberate: no poison-rate values are invented."
        )


# ============================================================
# PHASE 2 — SUPPLEMENTARY RESEARCH EVIDENCE
# ============================================================

PHASE2_ANALYSIS_DIR = RESULT_DIR / "supplementary_analysis"


def _wilson_interval(successes, total, z=1.96):
    if total <= 0:
        return {"estimate": None, "lower": None, "upper": None, "n": 0}
    p = float(successes) / float(total)
    denominator = 1.0 + (z * z / total)
    centre = (p + z * z / (2.0 * total)) / denominator
    margin = z * math.sqrt((p * (1.0 - p) / total) + (z * z / (4.0 * total * total))) / denominator
    return {"estimate": p, "lower": max(0.0, centre - margin), "upper": min(1.0, centre + margin), "n": int(total)}


def _bootstrap_f1_interval(scores, labels, threshold, repeats=2000):
    rng = np.random.default_rng(42)
    scores = list(scores)
    labels = list(labels)
    values = []
    for _ in range(repeats):
        idx = rng.integers(0, len(scores), size=len(scores))
        values.append(confusion([scores[i] for i in idx], [labels[i] for i in idx], threshold)["f1"])
    return {"estimate": confusion(scores, labels, threshold)["f1"], "lower": float(np.quantile(values, 0.025)), "upper": float(np.quantile(values, 0.975)), "n": len(scores)}


def _component_score(row, component):
    direct = {
        "neural_cleanse": "neural_cleanse_score",
        "activation_clustering": "activation_clustering_score",
        "strip": "strip_score",
    }[component]
    if direct in row:
        return float(row[direct])
    nested = row.get(component, {})
    if isinstance(nested, dict) and nested.get("score") is not None:
        return float(nested["score"])
    raise ValueError(f"Missing {component} score in Phase 2 record for {row.get('scenario')}")


def phase2_supplementary_analysis_main():
    """Create reproducible Phase 2 analysis files from saved, immutable evidence."""
    from sklearn.metrics import auc, precision_recall_curve, roc_curve

    PHASE2_ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    attacks_raw = load_json(ATTACK_FILE).get("experiments", load_json(ATTACK_FILE))
    clean_raw = load_json(CLEAN_FILE).get("experiments", load_json(CLEAN_FILE))
    attacks = normalise_attack_results(attacks_raw)
    controls = normalise_clean_controls(clean_raw)
    final = load_json(FINAL_JSON)
    thresholds = {str(k): float(v) for k, v in final["operating_thresholds"].items()}
    cal_attack, test_attack = split_calibration_holdout(attacks)
    cal_clean, test_clean = split_calibration_holdout(controls)

    # S3--S5 are evaluated as deterministic binary integrity gates, not as a
    # statistically tuned score fusion.
    gate_rows = []
    for scenario in sorted(INTEGRITY_GATE_SCENARIOS):
        a = [r for r in test_attack if r["scenario"] == scenario]
        c = [r for r in test_clean if r["scenario"] == scenario]
        t = thresholds[scenario]
        m = confusion([r["score"] for r in a + c], [True] * len(a) + [False] * len(c), t)
        gate_rows.append({"scenario": scenario, "gate_type": "binary_integrity", "threshold": t, "attack_runs": len(a), "clean_runs": len(c), **m})
    pd.DataFrame(gate_rows).to_csv(PHASE2_ANALYSIS_DIR / "integrity_gate_evaluation.csv", index=False)

    sensitivity_rows, ci_rows, missed_rows = [], [], []
    for scenario in ("S1_LABEL_FLIP", "S2_BACKDOOR"):
        a = [r for r in attacks if r["scenario"] == scenario]
        c = [r for r in controls if r["scenario"] == scenario]
        scores = [r["score"] for r in a + c]
        labels = [True] * len(a) + [False] * len(c)
        for threshold in sorted(set(np.linspace(0, 1, 101).tolist() + [thresholds[scenario]])):
            sensitivity_rows.append({"scenario": scenario, "threshold": float(threshold), **confusion(scores, labels, float(threshold))})

        a_test = [r for r in test_attack if r["scenario"] == scenario]
        c_test = [r for r in test_clean if r["scenario"] == scenario]
        m = confusion([r["score"] for r in a_test + c_test], [True] * len(a_test) + [False] * len(c_test), thresholds[scenario])
        heldout_scores = [r["score"] for r in a_test + c_test]
        heldout_labels = [True] * len(a_test) + [False] * len(c_test)
        ci_rows.extend([
            {"scenario": scenario, "metric": "recall", **_wilson_interval(m["TP"], m["TP"] + m["FN"])},
            {"scenario": scenario, "metric": "precision", **_wilson_interval(m["TP"], m["TP"] + m["FP"])},
            {"scenario": scenario, "metric": "fpr", **_wilson_interval(m["FP"], m["FP"] + m["TN"])},
            {"scenario": scenario, "metric": "f1_bootstrap_95ci", **_bootstrap_f1_interval(heldout_scores, heldout_labels, thresholds[scenario])},
        ])
        if scenario == "S2_BACKDOOR":
            for r in a_test:
                if abs(float(r.get("poison_rate", -1)) - 0.05) < 1e-12 and r["score"] < thresholds[scenario]:
                    missed_rows.append({**r, "operating_threshold": thresholds[scenario], "reason": "held_out_score_below_calibrated_threshold"})

        # PR/ROC are descriptive score-separation artefacts; reported final
        # metrics remain the held-out metrics in phase2_final_evaluation.
        precision, recall, pr_thresholds = precision_recall_curve(labels, scores)
        fpr, tpr, roc_thresholds = roc_curve(labels, scores)
        pd.DataFrame({"precision": precision, "recall": recall, "threshold": list(pr_thresholds) + [None]}).to_csv(PHASE2_ANALYSIS_DIR / f"{scenario.lower()}_pr_curve.csv", index=False)
        pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": roc_thresholds}).to_csv(PHASE2_ANALYSIS_DIR / f"{scenario.lower()}_roc_curve.csv", index=False)

    pd.DataFrame(sensitivity_rows).to_csv(PHASE2_ANALYSIS_DIR / "threshold_sensitivity.csv", index=False)
    pd.DataFrame(ci_rows).to_csv(PHASE2_ANALYSIS_DIR / "heldout_confidence_intervals.csv", index=False)
    pd.DataFrame(missed_rows).to_csv(PHASE2_ANALYSIS_DIR / "s2_rate05_heldout_missed_runs.csv", index=False)

    # Ablation uses the same leakage-safe run split and calibrates each score
    # independently.  Every variant uses the identical calibration/hold-out
    # rows so the comparison is leakage-safe.  STRIP remains diagnostic-only
    # by policy until it is independently validated against clean controls.
    ablation_rows = []
    variants = {
        "neural_cleanse_only": lambda r: _component_score(r, "neural_cleanse"),
        "activation_clustering_only": lambda r: _component_score(r, "activation_clustering"),
        "nc_strip_equal_fusion": lambda r: 0.5 * (_component_score(r, "neural_cleanse") + _component_score(r, "strip")),
        "nc_ac_equal_fusion": lambda r: 0.5 * (_component_score(r, "neural_cleanse") + _component_score(r, "activation_clustering")),
        "strip_ac_equal_fusion": lambda r: 0.5 * (_component_score(r, "strip") + _component_score(r, "activation_clustering")),
        "full_nc_strip_ac_equal_fusion": lambda r: (
            _component_score(r, "neural_cleanse")
            + _component_score(r, "strip")
            + _component_score(r, "activation_clustering")
        ) / 3.0,
        "strip_diagnostic_only": lambda r: _component_score(r, "strip"),
    }
    raw_s2_attack = [r for r in attacks_raw if r.get("scenario") == "S2_BACKDOOR"]
    raw_s2_clean = [r for r in clean_raw if r.get("scenario") == "S2_BACKDOOR"]
    raw_cal_attack, raw_test_attack = split_calibration_holdout(raw_s2_attack)
    raw_cal_clean, raw_test_clean = split_calibration_holdout(raw_s2_clean)
    for name, scorer in variants.items():
        ca = [{**r, "score": scorer(r)} for r in raw_cal_attack]
        cc = [{**r, "score": scorer(r)} for r in raw_cal_clean]
        ta = [{**r, "score": scorer(r)} for r in raw_test_attack]
        tc = [{**r, "score": scorer(r)} for r in raw_test_clean]
        threshold, calibration = select_scenario_threshold(ca, cc, "S2_BACKDOOR")
        heldout = confusion([r["score"] for r in ta + tc], [True] * len(ta) + [False] * len(tc), threshold)
        uses_strip = "strip" in name
        ablation_rows.append({
            "variant": name,
            "scope": "S2_BACKDOOR_DETECTOR_ABLATION",
            "operational": not uses_strip,
            "policy_status": (
                "DIAGNOSTIC_ONLY_STRIP_NOT_ENABLED"
                if uses_strip else "OPERATIONAL"
            ),
            "threshold": threshold,
            "calibration_fpr": calibration["fpr"],
            **heldout,
        })

    # Integrity gates are deterministic binary controls for S3--S5, so they
    # are reported as a separate aggregate row rather than mixed with the S2
    # detector-score ablation.  This row uses the same held-out gate rows that
    # are written to integrity_gate_evaluation.csv.
    gate_scores, gate_labels = [], []
    for scenario in sorted(INTEGRITY_GATE_SCENARIOS):
        gate_attack = [r for r in test_attack if r["scenario"] == scenario]
        gate_clean = [r for r in test_clean if r["scenario"] == scenario]
        gate_scores.extend([float(r["score"]) for r in gate_attack + gate_clean])
        gate_labels.extend([True] * len(gate_attack) + [False] * len(gate_clean))
    if gate_scores:
        gate_metrics = confusion(gate_scores, gate_labels, 0.5)
        ablation_rows.append({
            "variant": "integrity_gates_only",
            "scope": "S3_S4_S5_BINARY_INTEGRITY_GATES",
            "operational": True,
            "policy_status": "OPERATIONAL",
            "threshold": 0.5,
            "calibration_fpr": 0.0,
            **gate_metrics,
        })
    ablation_csv = PHASE2_ANALYSIS_DIR / "s2_ablation.csv"
    pd.DataFrame(ablation_rows).to_csv(ablation_csv, index=False)

    summary = {
        "phase": "PHASE_2_SUPPLEMENTARY_ANALYSIS",
        "scope": "Saved Phase 2 evidence; PR/ROC and sensitivity are descriptive, while final claims use held-out metrics.",
        "integrity_gate_evaluation": gate_rows,
        "sbom_pickle_environment_controls": (
    "Dedicated clean/malicious gate controls are saved in "
    "evidence/phase2/gate_controls/gate_control_evidence.json."
     ),
        "files": {
            "directory": str(PHASE2_ANALYSIS_DIR),
            "ablation_csv": str(ablation_csv),
            "integrity_gate_csv": str(PHASE2_ANALYSIS_DIR / "integrity_gate_evaluation.csv"),
        },
    }
    (PHASE2_ANALYSIS_DIR / "phase2_supplementary_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Phase 2 supplementary analysis saved: {PHASE2_ANALYSIS_DIR}")


def phase2_gate_controls_main():
    """Execute independent clean/malicious controls for the three policy gates.

    The unsafe Pickle control contains only static opcode bytes and is never
    deserialised; recursive_pickle_scan performs static inspection only.
    """
    control_dir = EVIDENCE_DIR / "gate_controls"
    control_dir.mkdir(parents=True, exist_ok=True)

    baseline_sbom = control_dir / "baseline_sbom.json"
    clean_sbom = control_dir / "clean_sbom.json"
    modified_sbom = control_dir / "modified_sbom.json"
    baseline_payload = {"packages": {"numpy": "1.26.4", "scikit-learn": "1.5.1"}}
    baseline_sbom.write_text(json.dumps(baseline_payload, indent=2), encoding="utf-8")
    clean_sbom.write_text(json.dumps(baseline_payload, indent=2), encoding="utf-8")
    modified_sbom.write_text(json.dumps({"packages": {"numpy": "1.26.4", "scikit-learn": "9.9.9"}}, indent=2), encoding="utf-8")
    sbom_clean = sbom_integrity_gate(baseline_sbom, clean_sbom)
    sbom_modified = sbom_integrity_gate(baseline_sbom, modified_sbom)

    safe_pickle = control_dir / "safe_control.pkl"
    unsafe_pickle = control_dir / "unsafe_static_control.pkl"
    safe_pickle.write_bytes(pickle.dumps(("safe_control", 1), protocol=4))
    # GLOBAL os.system is deliberately static test data. It is not executed.
    unsafe_pickle.write_bytes(b"cos\nsystem\n.")
    pickle_clean = recursive_pickle_scan(safe_pickle)
    pickle_modified = recursive_pickle_scan(unsafe_pickle)

    approved_env = control_dir / "approved_environment.yml"
    denied_env = control_dir / "denied_environment.yml"
    approved_env.write_text("name: approved\ndependencies:\n  - python=3.11\n  - numpy=1.26.4\n", encoding="utf-8")
    denied_env.write_text("name: denied\ndependencies:\n  - python=3.11\n  - forbiddenpkg=1.0\n", encoding="utf-8")
    environment_clean = dependency_environment_gate(approved_env, denied_packages={"forbiddenpkg"})
    environment_modified = dependency_environment_gate(denied_env, denied_packages={"forbiddenpkg"})

    controls = [
        {"gate": "SBOM", "control": "clean", "passed": sbom_clean["status"] == "MATCH", "result": sbom_clean},
        {"gate": "SBOM", "control": "modified", "passed": sbom_modified["status"] == "INTEGRITY_OR_PROVENANCE_VIOLATION", "result": sbom_modified},
        {"gate": "PICKLE", "control": "clean", "passed": bool(pickle_clean.get("safe")), "result": pickle_clean},
        {"gate": "PICKLE", "control": "modified", "passed": not bool(pickle_modified.get("safe")), "result": pickle_modified},
        {"gate": "ENVIRONMENT", "control": "clean", "passed": environment_clean["status"] == "MATCH", "result": environment_clean},
        {"gate": "ENVIRONMENT", "control": "modified", "passed": environment_modified["status"] == "POLICY_VIOLATION", "result": environment_modified},
    ]
    evidence = {"phase": "PHASE_2_GATE_CONTROLS", "static_pickle_only": True, "all_controls_passed": all(c["passed"] for c in controls), "controls": controls}
    (control_dir / "gate_control_evidence.json").write_text(json.dumps(evidence, indent=2, default=str), encoding="utf-8")
    pd.DataFrame([{"gate": c["gate"], "control": c["control"], "passed": c["passed"]} for c in controls]).to_csv(control_dir / "gate_control_summary.csv", index=False)
    if not evidence["all_controls_passed"]:
        raise RuntimeError("One or more Phase 2 gate controls did not produce the expected result.")
    print(f"Phase 2 gate controls passed: {control_dir}")


def phase2_figures_main():
    """Render dissertation-ready PR and ROC figures from saved CSV evidence."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = PHASE2_ANALYSIS_DIR / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    for curve_type, x_column, y_column, title in (
        ("pr", "recall", "precision", "Descriptive Precision–Recall Curves (S1/S2)"),
        ("roc", "fpr", "tpr",  "Descriptive ROC Curves (S1/S2)"),
    ):
        plt.figure(figsize=(7, 5))
        for scenario in ("s1_label_flip", "s2_backdoor"):
            frame = pd.read_csv(PHASE2_ANALYSIS_DIR / f"{scenario}_{curve_type}_curve.csv")
            plt.plot(frame[x_column], frame[y_column], label=scenario.upper())
        if curve_type == "roc":
            plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random baseline")
        plt.xlabel(x_column.upper())
        plt.ylabel(y_column.upper())
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(figures / f"s1_s2_{curve_type}_curves.png", dpi=300)
        plt.close()
    print(f"Phase 2 PR/ROC figures saved: {figures}")





# ===== EMBEDDED overhead_benchmark(1).py =====
import argparse
import csv
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
from joblib import load
import pandas as pd


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("phase2_runtime_overhead", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load Phase 2 module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def timed(name, fn, repeats: int):
    values = []
    last = None
    for _ in range(repeats):
        start = time.perf_counter()
        last = fn()
        values.append(time.perf_counter() - start)
    record = {
        "component": name,
        "repeats": repeats,
        "mean_seconds": float(np.mean(values)),
        "std_seconds": float(np.std(values, ddof=1)) if repeats > 1 else 0.0,
        "min_seconds": float(np.min(values)),
        "max_seconds": float(np.max(values)),
        "last_status": "OK" if last is not None else "NONE",
        "result_status": last.get("status") if isinstance(last, dict) else None,
        "result_error": last.get("error") if isinstance(last, dict) else None,
    }
    return record


def overhead_benchmark_main(argv=None):
    parser = argparse.ArgumentParser(description="Pipeline overhead benchmark for the Secure MLflow Phase 2 detector stack")
    parser.add_argument("--model", default="models/clean/clean_baseline.joblib")
    parser.add_argument("--data", default="data/processed/split/clean_test.csv")
    parser.add_argument("--phase2", default="research_pipeline.py")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--sample-rows", type=int, default=1000)
    parser.add_argument("--output", default="results/phase2/overhead_benchmark.json")
    args = parser.parse_args(argv)

    if args.repeats <= 0 or args.sample_rows <= 0:
        raise ValueError("--repeats and --sample-rows must be > 0")

    root = Path(__file__).resolve().parent
    model_path = root / args.model
    data_path = root / args.data
    phase2_path = root / args.phase2
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"Data not found: {data_path}")

    pm = load_module(phase2_path) if phase2_path != root / "research_pipeline.py" else sys.modules[__name__]
    bundle = load(model_path)
    model = bundle["model"]
    scaler = bundle["scaler"]
    df = pd.read_csv(data_path)
    X_raw = df.drop(columns=["label","row_id"], errors="ignore")
    X_raw = X_raw.replace([np.inf, -np.inf], np.nan).fillna(0)
    sample_n = min(args.sample_rows, len(X_raw))
    if sample_n < len(X_raw):
        X_raw = X_raw.sample(n=sample_n, random_state=12345).sort_index()
    else:
        X_raw = X_raw.copy()
    X_raw = X_raw.astype("float32")
    X = scaler.transform(X_raw)
    cfg = pm.load_cfg()

    rows = []
    rows.append(timed("baseline_model_inference", lambda: model.predict(X), args.repeats))
    rows.append(timed("component_1_sbom", pm.generate_local_sbom, args.repeats))
    rows.append(timed("component_2_pickle_scan", lambda: pm.recursive_pickle_scan(model_path), args.repeats))
    rows.append(timed("component_3_training_monitor", lambda: pm.training_anomaly_from_model(model), args.repeats))
    rows.append(timed("component_4_neural_cleanse", lambda: pm.neural_cleanse_style_score(model, X, cfg["attacks"]["target_label"], cfg["detection"]["nc_steps"], cfg["detection"]["nc_learning_rate"]), args.repeats))
    rows.append(timed("component_4_strip", lambda: pm.strip_entropy_score(model, X, cfg["detection"]["strip_samples"]), args.repeats))
    rows.append(timed("component_4_activation_clustering", lambda: pm.activation_clustering_score(model, X, cfg["detection"]["cluster_min_samples"]), args.repeats))
    if getattr(pm, "ART_AVAILABLE", False) and "label" in df.columns:
        # Keep labels aligned with the deterministic sampled feature rows.
        y_art = df.loc[X_raw.index, "label"].astype(int).to_numpy()
        art_timing = timed(
            "component_4_ibm_art_activation_defence",
            lambda: pm.art_activation_clustering_score(model, X, y_art),
            args.repeats,
        )
        # A returned failure dictionary is not a successful ART measurement.
        if art_timing.get("result_status") != "EXECUTED":
            art_timing["mean_seconds"] = None
            art_timing["std_seconds"] = None
            art_timing["min_seconds"] = None
            art_timing["max_seconds"] = None
            art_timing["last_status"] = "FAILED"
        rows.append(art_timing)

    def fusion_call():
        nc = pm.neural_cleanse_style_score(model, X, cfg["attacks"]["target_label"], cfg["detection"]["nc_steps"], cfg["detection"]["nc_learning_rate"])
        st = pm.strip_entropy_score(model, X, cfg["detection"]["strip_samples"])
        ac = pm.activation_clustering_score(model, X, cfg["detection"]["cluster_min_samples"])
        return pm.fuse(nc["score"], st["score"], ac["score"], cfg)

    rows.append(timed("component_5_fusion", fusion_call, args.repeats))

    def full_detector_stack_call():
        """One complete configured pass; this is the only end-to-end timing."""
        sbom = pm.generate_local_sbom()
        pickle_scan = pm.recursive_pickle_scan(model_path)
        training = pm.training_anomaly_from_model(model)
        nc = pm.neural_cleanse_style_score(
            model, X, cfg["attacks"]["target_label"],
            cfg["detection"]["nc_steps"], cfg["detection"]["nc_learning_rate"],
        )
        st = pm.strip_entropy_score(model, X, cfg["detection"]["strip_samples"])
        ac = pm.activation_clustering_score(model, X, cfg["detection"]["cluster_min_samples"])
        fusion = pm.fuse(nc["score"], st["score"], ac["score"], cfg)
        return {
            "status": "EXECUTED",
            "fusion_score": fusion.get("risk_score") if isinstance(fusion, dict) else None,
            "component_statuses": {
                "sbom": sbom.get("status") if isinstance(sbom, dict) else None,
                "pickle": "SAFE" if isinstance(pickle_scan, dict) and pickle_scan.get("safe") else "NOT_SAFE_OR_UNKNOWN",
                "training": training.get("decision") if isinstance(training, dict) else None,
                "neural_cleanse": nc.get("decision") if isinstance(nc, dict) else None,
                "strip": st.get("decision") if isinstance(st, dict) else None,
                "activation_clustering": ac.get("decision") if isinstance(ac, dict) else None,
            },
        }

    end_to_end = timed("full_detector_stack", full_detector_stack_call, args.repeats)

    try:
        art_status = pm.art_estimator_report(model)
        rows.append({"component": "component_4_ibm_art_estimator", "repeats": args.repeats, "mean_seconds": None, "std_seconds": None, "min_seconds": None, "max_seconds": None, "last_status": art_status.get("status", "UNKNOWN"), "result_status": art_status.get("status"), "result_error": art_status.get("error") or art_status.get("reason")})
    except Exception as exc:
        rows.append({"component": "component_4_ibm_art_estimator", "repeats": args.repeats, "mean_seconds": None, "std_seconds": None, "min_seconds": None, "max_seconds": None, "last_status": "FAILED", "result_status": "FAILED", "result_error": str(exc)})

    result = {
        "phase": "PHASE_2_OVERHEAD_BENCHMARK",
        "model": str(model_path),
        "sample_rows": int(len(X)),
        "repeats": int(args.repeats),
        "components": rows,
        "end_to_end_mean_seconds": end_to_end["mean_seconds"],
        "end_to_end_std_seconds": end_to_end["std_seconds"],
        "end_to_end_min_seconds": end_to_end["min_seconds"],
        "end_to_end_max_seconds": end_to_end["max_seconds"],
        "end_to_end_result_status": end_to_end["result_status"],
        "benchmark_scope": "full_detector_stack_once_per_repeat",
        "sample_selection": "deterministic random sample (seed=12345), not first-N rows",
        "note": "Individual component timings are a breakdown and must not be summed because detection components are recomputed during fusion. end_to_end_mean_seconds is the non-double-counted overhead measurement. Failed optional ART runs are reported with their error and excluded from the end-to-end stack.",
    }
    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    csv_path = output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    print(json.dumps(result, indent=2, default=str))





# ===== EMBEDDED phase2_secondary_unsw_nb15.py =====
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


LABEL_CANDIDATES = ("label", "Label")
DROP_CANDIDATES = {"id", "attack_cat", "attack category", "attack_category"}


def read_csv_checked(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"UNSW-NB15 CSV not found: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"UNSW-NB15 CSV is empty: {path}")
    return df


def find_label(df: pd.DataFrame) -> str:
    for candidate in LABEL_CANDIDATES:
        if candidate in df.columns:
            return candidate
    raise ValueError("UNSW-NB15 file must contain a binary 'label' column.")


def normalise_binary_label(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        labels = pd.to_numeric(series, errors="raise").astype(int)
    else:
        mapping = {"normal": 0, "benign": 0, "0": 0, "attack": 1, "malicious": 1, "1": 1}
        lowered = series.astype(str).str.strip().str.lower()
        unknown = sorted(set(lowered) - set(mapping))
        if unknown:
            raise ValueError(f"Unsupported UNSW-NB15 labels: {unknown[:10]}")
        labels = lowered.map(mapping).astype(int)
    unique = set(labels.unique().tolist())
    if not unique.issubset({0, 1}) or len(unique) < 2:
        raise ValueError(f"Expected binary labels 0/1, found: {sorted(unique)}")
    return labels


def cap_stratified(df: pd.DataFrame, label_col: str, max_rows: int, seed: int) -> pd.DataFrame:
    if len(df) <= max_rows:
        return df.copy()
    per_class = max_rows // 2
    parts = []
    for label in (0, 1):
        part = df[df[label_col] == label]
        if len(part) < per_class:
            raise ValueError(f"Cannot cap stratified data: class {label} has only {len(part)} rows.")
        parts.append(part.sample(n=per_class, random_state=seed + label))
    out = pd.concat(parts, ignore_index=True)
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def make_preprocessor(X: pd.DataFrame):
    numeric = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    categorical = [c for c in X.columns if c not in numeric]
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encode", encoder)]), categorical),
        ],
        remainder="drop",
    )


def training_integrity(model) -> dict:
    losses = getattr(model, "loss_curve_", []) or []
    if not losses:
        return {"history_length": 0, "monitor_mode": "post_training_loss_curve_proxy", "gradient_norm_is_proxy": True, "anomaly_score": 0.0}
    arr = np.asarray(losses, dtype=float)
    diff = np.abs(np.diff(arr)) if len(arr) > 1 else np.asarray([], dtype=float)
    return {
        "history_length": int(len(arr)),
        "mean_loss": float(arr.mean()),
        "final_loss": float(arr[-1]),
        "mean_loss_change_proxy": float(diff.mean()) if len(diff) else 0.0,
        "max_loss_change_proxy": float(diff.max()) if len(diff) else 0.0,
        "monitor_mode": "post_training_loss_curve_proxy",
        "gradient_norm_is_proxy": True,
    }


def optional_art(model, X_train: np.ndarray, y_train: np.ndarray, max_samples: int) -> dict:
    try:
        from art.defences.detector.poison.activation_defence import ActivationDefence
    except ImportError:
        return {"available": False, "status": "NOT_INSTALLED"}

    try:
        if len(X_train) > max_samples:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(X_train), max_samples, replace=False)
            X_train, y_train = X_train[idx], y_train[idx]
        adapter = _ARTMLPActivationAdapter(model)
        defence = ActivationDefence(adapter, X_train.astype(np.float32), y_train.astype(int))
        report, is_clean = defence.detect_poison(nb_clusters=2, nb_dims=min(10, max(2, X_train.shape[1]-1)), reduce="PCA", cluster_analysis="smaller")
        flags = np.asarray(is_clean, dtype=int)
        poison_fraction = float(np.mean(flags == 0)) if len(flags) else 0.0
        return {
            "available": True,
            "status": "EXECUTED",
            "adapter": "sklearn_mlp_penultimate_hidden_activation",
            "activation_layer": adapter.layer_names[-1],
            "poison_fraction": poison_fraction,
            "samples": int(len(flags)),
            "report": report,
        }
    except Exception as exc:
        return {"available": True, "status": "FAILED", "error": str(exc)}


def unsw_nb15_main(argv=None):
    parser = argparse.ArgumentParser(description="UNSW-NB15 secondary benchmark for the Secure MLflow framework")
    parser.add_argument("--train", required=True, help="UNSW-NB15 training CSV")
    parser.add_argument("--test", default=None, help="UNSW-NB15 testing CSV; optional if train is a single combined file")
    parser.add_argument("--max-train-rows", type=int, default=20000)
    parser.add_argument("--max-test-rows", type=int, default=10000)
    parser.add_argument("--natural-prevalence", action="store_true",
                        help="Preserve the supplied class prevalence instead of balanced capping.")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--output", default="results/secondary/unsw_nb15_benchmark.json")
    args = parser.parse_args(argv)

    start = time.perf_counter()
    train = read_csv_checked(Path(args.train))
    label_col = find_label(train)
    train[label_col] = normalise_binary_label(train[label_col])

    if args.test:
        test = read_csv_checked(Path(args.test))
        test_label = find_label(test)
        if test_label != label_col:
            test = test.rename(columns={test_label: label_col})
        test[label_col] = normalise_binary_label(test[label_col])
        split_protocol = "official_UNSW_NB15_train_test_partition"
    else:
        from sklearn.model_selection import train_test_split
        train, test = train_test_split(train, test_size=0.2, stratify=train[label_col], random_state=args.random_state)
        train = train.reset_index(drop=True); test = test.reset_index(drop=True)
        split_protocol = "generated_stratified_80_20_partition"

    if not args.natural_prevalence:
        train = cap_stratified(train, label_col, args.max_train_rows, args.random_state)
        test = cap_stratified(test, label_col, args.max_test_rows, args.random_state + 99)
        sampling_mode = "balanced_stratified_cap"
    else:
        if len(train) > args.max_train_rows:
            train = train.sample(n=args.max_train_rows, random_state=args.random_state).reset_index(drop=True)
        if len(test) > args.max_test_rows:
            test = test.sample(n=args.max_test_rows, random_state=args.random_state + 99).reset_index(drop=True)
        sampling_mode = "natural_prevalence_random_cap"

    drop_cols = [c for c in train.columns if c.strip().lower() in DROP_CANDIDATES]
    X_train = train.drop(columns=[label_col] + drop_cols, errors="ignore")
    y_train = train[label_col].to_numpy(dtype=int)
    X_test = test.drop(columns=[label_col] + drop_cols, errors="ignore")
    y_test = test[label_col].to_numpy(dtype=int)

    if list(X_train.columns) != list(X_test.columns):
        raise ValueError("UNSW-NB15 train/test feature schemas differ after dropping metadata columns.")

    preprocessor = make_preprocessor(X_train)
    model = MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", solver="adam", learning_rate_init=0.001, max_iter=60, early_stopping=True, validation_fraction=0.15, n_iter_no_change=8, random_state=args.random_state)
    pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])
    pipe.fit(X_train, y_train)
    predictions = pipe.predict(X_test)
    probabilities = pipe.predict_proba(X_test)[:, 1]

    transformed_train = preprocessor.transform(X_train)
    fitted_model = pipe.named_steps["model"]

    tn, fp, fn, tp = confusion_matrix(y_test, predictions, labels=[0, 1]).ravel()
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "pr_auc": float(average_precision_score(y_test, probabilities)),
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
    }

    result = {
        "phase": "SECONDARY_BENCHMARK_UNSW_NB15",
        "dataset": "UNSW-NB15",
        "primary_pipeline_unchanged": True,
        "benchmark_role": "secondary_generalisation",
        "purpose": "Secondary generalisation benchmark; it does not replace the primary N-BaIoT attack experiments.",
        "split_protocol": split_protocol,
        "holdout_used_for_training_or_threshold_selection": False,
        "random_state": int(args.random_state),
        "source_files": {
            "train": str(Path(args.train).resolve()),
            "test": str(Path(args.test).resolve()) if args.test else None,
        },
        "sampling_mode": sampling_mode,
        "class_distribution_train": {str(k): int(v) for k, v in pd.Series(y_train).value_counts().sort_index().items()},
        "class_distribution_test": {str(k): int(v) for k, v in pd.Series(y_test).value_counts().sort_index().items()},
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "feature_count_before_encoding": int(X_train.shape[1]),
        "metrics": metrics,
        "training_integrity_monitor": training_integrity(fitted_model),
        "ibm_art_activation_defence": optional_art(fitted_model, transformed_train.astype(np.float32), y_train, 5000),
        "execution_seconds": float(time.perf_counter() - start),
    }
    output = Path(args.output)
    if not output.is_absolute(): output = Path(__file__).resolve().parent / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(json.dumps(result, indent=2, default=str))





# ============================================================
# PHASE A ORCHESTRATION
# ============================================================

def run_phase_a_core(require_art=False):
    if require_art and not globals().get("ART_AVAILABLE", False):
        raise RuntimeError("IBM ART is required. Install requirements_final.txt and rerun.")
    phase0_dataset_prepare_main()
    phase0_split_main()
    phase1_clean_baseline_main()
    phase1_master_main(argv=[])
    phase2_clean_controls_main()
    phase2_experiments_main()
    phase2_final_evaluation_main()


def main():
    parser=argparse.ArgumentParser(description="Two-phase Secure MLflow research pipeline — Phase A")
    parser.add_argument("--full",action="store_true",help="Run Phase A core + overhead + UNSW-NB15")
    parser.add_argument("--core",action="store_true",help="Run dataset, attacks, detection and final evaluation")
    parser.add_argument("--dataset",action="store_true",help="Run dataset preparation and split")
    parser.add_argument("--phase1",action="store_true",help="Run clean baseline and attacks")
    parser.add_argument("--phase2",action="store_true",help="Run controls, experiments and final evaluation")
    parser.add_argument("--final-evaluation",action="store_true",help="Recalculate final Phase 2 metrics from existing evidence only (no model runs)")
    parser.add_argument("--phase2-analysis",action="store_true",help="Create Phase 2 gate, ablation, threshold, PR/ROC, CI and missed-run evidence from saved results")
    parser.add_argument("--phase2-gate-controls",action="store_true",help="Run clean and malicious SBOM, Pickle, and environment gate controls")
    parser.add_argument("--phase2-figures",action="store_true",help="Render S1/S2 PR and ROC figures from saved Phase 2 analysis")
    parser.add_argument("--overhead",action="store_true",help="Run full-configuration overhead benchmark")
    parser.add_argument("--unsw",action="store_true",help="Run UNSW-NB15 secondary benchmark")
    parser.add_argument("--unsw-poison-validation",action="store_true",help="Run leakage-safe UNSW-NB15 S1/S2 poisoning validation")
    parser.add_argument("--unsw-train",default=None,help="UNSW-NB15 training CSV for --unsw/--full")
    parser.add_argument("--unsw-test",default=None,help="Optional UNSW-NB15 testing CSV")
    parser.add_argument("--unsw-natural-prevalence",action="store_true",help="Do not balance/cap UNSW classes for the secondary benchmark")
    parser.add_argument("--unsw-max-train-rows",type=int,default=20000,help="Maximum UNSW training rows; set above 175341 to use the full official training partition")
    parser.add_argument("--unsw-max-test-rows",type=int,default=10000,help="Maximum UNSW testing rows; set above 82332 to use the full official testing partition")
    parser.add_argument("--unsw-output",default="results/secondary/unsw_nb15_benchmark.json",help="UNSW evidence JSON path; use D: when E: has little free space")
    parser.add_argument("--unsw-poison-output-dir",default="results/secondary/unsw_nb15_poison_validation",help="Directory for UNSW poisoning evidence")
    parser.add_argument("--unsw-poison-max-train-rows",type=int,default=0,help="Maximum official-training rows for poisoning validation; 0 uses all rows")
    parser.add_argument("--unsw-poison-max-iter",type=int,default=60,help="Maximum sklearn MLP iterations per validation model")
    parser.add_argument("--unsw-trigger-feature",default=None,help="Optional pre-specified numeric S2 trigger feature; defaults to sbytes when available")
    parser.add_argument("--unsw-trigger-features",default="sbytes,dbytes",help="Comma-separated trigger features for the S2 robustness matrix")
    parser.add_argument("--unsw-trigger-strength",choices=("weak","medium","strong"),default="strong")
    parser.add_argument("--unsw-s2-source-label",type=int,choices=(0,1),default=0)
    parser.add_argument("--unsw-s2-target-label",type=int,choices=(0,1),default=1)
    parser.add_argument("--unsw-s2-robustness",action="store_true",help="Run both attack directions, multiple trigger features and all trigger strengths")
    parser.add_argument("--unsw-poison-resume",action="store_true",help="Resume from completed per-run UNSW poisoning JSON files")
    parser.add_argument("--overhead-repeats",type=int,default=3)
    parser.add_argument("--overhead-sample-rows",type=int,default=1000)
    parser.add_argument("--require-art",action="store_true",help="Require IBM ART for requested Phase A run")
    args=parser.parse_args()
    if args.full:
        run_phase_a_core(require_art=args.require_art)
        overhead_benchmark_main(argv=["--repeats", str(args.overhead_repeats), "--sample-rows", str(args.overhead_sample_rows)])
        if not args.unsw_train:
            raise SystemExit("--full requires --unsw-train PATH so the secondary benchmark is reproducible.")
        unsw_argv=[
            "--train", args.unsw_train,
            "--max-train-rows", str(args.unsw_max_train_rows),
            "--max-test-rows", str(args.unsw_max_test_rows),
            "--output", args.unsw_output,
        ]
        if args.unsw_test:
            unsw_argv += ["--test", args.unsw_test]
        if args.unsw_natural_prevalence:
            unsw_argv.append("--natural-prevalence")
        unsw_nb15_main(argv=unsw_argv)
        return
    if args.core:
        run_phase_a_core(require_art=args.require_art); return
    if args.dataset:
        phase0_dataset_prepare_main(); phase0_split_main(); return
    if args.phase1:
        phase1_clean_baseline_main(); phase1_master_main(argv=[]); return
    if args.phase2:
        if args.require_art and not globals().get("ART_AVAILABLE",False): raise RuntimeError("IBM ART is required.")
        phase2_clean_controls_main(); phase2_experiments_main(); phase2_final_evaluation_main(); return
    if args.final_evaluation:
        phase2_final_evaluation_main(); return
    if args.phase2_analysis:
        phase2_supplementary_analysis_main(); return
    if args.phase2_gate_controls:
        phase2_gate_controls_main(); return
    if args.phase2_figures:
        phase2_figures_main(); return
    if args.overhead:
        overhead_argv=["--repeats", str(args.overhead_repeats), "--sample-rows", str(args.overhead_sample_rows)]
        overhead_benchmark_main(argv=overhead_argv); return
    if args.unsw_poison_validation:
        if not args.unsw_train or not args.unsw_test:
            raise SystemExit("--unsw-poison-validation requires --unsw-train and --unsw-test.")
        from unsw_poison_validation import main as unsw_poison_main
        poison_argv = [
            "--train", args.unsw_train,
            "--test", args.unsw_test,
            "--output-dir", args.unsw_poison_output_dir,
            "--repetitions", "5",
            "--seed", "20260910",
            "--max-train-rows", str(args.unsw_poison_max_train_rows),
            "--max-iter", str(args.unsw_poison_max_iter),
            "--trigger-strength", args.unsw_trigger_strength,
            "--s2-source-label", str(args.unsw_s2_source_label),
            "--s2-target-label", str(args.unsw_s2_target_label),
            "--trigger-features", args.unsw_trigger_features,
        ]
        if args.unsw_trigger_feature:
            poison_argv += ["--trigger-feature", args.unsw_trigger_feature]
        if args.unsw_poison_resume:
            poison_argv.append("--resume")
        if args.unsw_s2_robustness:
            poison_argv.append("--robustness")
        unsw_poison_main(argv=poison_argv); return
    if args.unsw:
        if not args.unsw_train:
            raise SystemExit("--unsw requires --unsw-train PATH.")
        unsw_argv=[
            "--train", args.unsw_train,
            "--max-train-rows", str(args.unsw_max_train_rows),
            "--max-test-rows", str(args.unsw_max_test_rows),
            "--output", args.unsw_output,
        ]
        if args.unsw_test:
            unsw_argv += ["--test", args.unsw_test]
        if args.unsw_natural_prevalence:
            unsw_argv.append("--natural-prevalence")
        unsw_nb15_main(argv=unsw_argv); return
    parser.print_help()

if __name__ == "__main__":
    main()
