#!/usr/bin/env python3
"""Self-contained Phase B automated response + registry hook."""
import sys
# ============================================================
# PHASE 3 — INTEGRATED AUTOMATED RESPONSE FRAMEWORK
#
# PURPOSE:
#   Connect Phase 2 detection evidence to:
#       1. MLflow model lifecycle response
#       2. Quarantine
#       3. Rollback
#       4. Wazuh SIEM alert
#       5. MTTD / MTTR measurement
#       6. Forensic evidence
#
# IMPORTANT:
#   This script does NOT invent detection results.
#   It reads the real Phase 2 final evaluation.
# ============================================================

import os
import json
import time
import hashlib
import socket
import argparse
import subprocess
import re
import shlex
import csv
import statistics
from datetime import datetime, timezone

import requests
import mlflow
from mlflow import MlflowClient


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.dirname(__file__)
)

PHASE2_EVIDENCE = os.path.join(
    PROJECT_ROOT,
    "evidence",
    "phase2",
    "final_evaluation",
    "phase2_final_evaluation.json"
)

RESULTS_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "phase3"
)

EVIDENCE_DIR = os.path.join(
    PROJECT_ROOT,
    "evidence",
    "phase3"
)

RUN_EVIDENCE_DIR = os.path.join(EVIDENCE_DIR, "response_runs")
CLEAN_CONTROL_DIR = os.path.join(EVIDENCE_DIR, "clean_controls")
LATEST_EVIDENCE_FILE = os.path.join(EVIDENCE_DIR, "phase3_response_evidence.json")
PHASE3_SUMMARY_JSON = os.path.join(RESULTS_DIR, "phase3_response_summary.json")
PHASE3_SUMMARY_CSV = os.path.join(RESULTS_DIR, "phase3_response_runs.csv")
PHASE3_CLEAN_CONTROL_CSV = os.path.join(RESULTS_DIR, "phase3_clean_control_runs.csv")

MLFLOW_URI = "http://127.0.0.1:5000"

MODEL_NAME = "nbaiot_detector"
ACTIVE_DEPLOYMENT_ALIAS = "production"

# Wazuh manager API
WAZUH_API = os.getenv("WAZUH_API", "https://127.0.0.1:55000")
WAZUH_USER = os.getenv("WAZUH_USER")
WAZUH_PASSWORD = os.getenv("WAZUH_PASSWORD")
WAZUH_VERIFY_TLS = os.getenv("WAZUH_VERIFY_TLS", "true").strip().lower() in {"1", "true", "yes", "on"}
WAZUH_API_TIMEOUT_SECONDS = float(os.getenv("WAZUH_API_TIMEOUT_SECONDS", "30"))
WAZUH_AUTH_TIMEOUT_SECONDS = float(os.getenv("WAZUH_AUTH_TIMEOUT_SECONDS", str(WAZUH_API_TIMEOUT_SECONDS)))
WAZUH_INGEST_TIMEOUT_SECONDS = float(os.getenv("WAZUH_INGEST_TIMEOUT_SECONDS", "15"))
WAZUH_ALERT_TIMEOUT_SECONDS = float(os.getenv("WAZUH_ALERT_TIMEOUT_SECONDS", "30"))
WAZUH_RETRY_ATTEMPTS = max(1, int(os.getenv("WAZUH_RETRY_ATTEMPTS", "3")))
WAZUH_RETRY_BACKOFF_SECONDS = max(0.0, float(os.getenv("WAZUH_RETRY_BACKOFF_SECONDS", "1")))
WAZUH_CA_BUNDLE = os.getenv("WAZUH_CA_BUNDLE", "").strip()
WAZUH_PRODUCTION_MODE = os.getenv("WAZUH_PRODUCTION_MODE", "false").strip().lower() in {"1", "true", "yes", "on"}
WAZUH_DOCKER_CONTAINER = os.getenv("WAZUH_DOCKER_CONTAINER", "")
WAZUH_MANAGER_LOG_PATH = os.getenv("WAZUH_MANAGER_LOG_PATH", "/var/ossec/logs/mlflow/phase3_events.json")
WAZUH_EXPECTED_RULE_ID = os.getenv("WAZUH_EXPECTED_RULE_ID", "100103")
WAZUH_EXPECTED_RULE_LEVEL = int(os.getenv("WAZUH_EXPECTED_RULE_LEVEL", "12"))

# Wazuh custom event file
WAZUH_EVENT_FILE = os.path.join(
    PROJECT_ROOT,
    "results",
    "phase3",
    "wazuh_phase3_events.json"
)



# ============================================================
# DIRECTORY SETUP
# ============================================================

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(EVIDENCE_DIR, exist_ok=True)
os.makedirs(RUN_EVIDENCE_DIR, exist_ok=True)
os.makedirs(CLEAN_CONTROL_DIR, exist_ok=True)


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def utc_now():
    return datetime.now(timezone.utc)


def iso_now():
    return utc_now().isoformat()


def sha256_file(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def print_section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


# ============================================================
# LOAD PHASE 2 EVIDENCE
# ============================================================

def load_phase2_evidence():

    if not os.path.exists(PHASE2_EVIDENCE):
        raise FileNotFoundError(
            f"Phase 2 evidence not found:\n{PHASE2_EVIDENCE}"
        )

    with open(PHASE2_EVIDENCE, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# EXTRACT PHASE 2 DECISION
# ============================================================

def extract_phase2_decision(data):
    """Read Phase 2's scenario-specific threshold evidence without inventing values."""
    raw_thresholds = data.get("operating_thresholds")
    if not isinstance(raw_thresholds, dict) or not raw_thresholds:
        raise RuntimeError(
            "Phase 2 evidence must contain non-empty operating_thresholds. "
            "Run research_pipeline_A_final.py --final-evaluation first."
        )
    thresholds = {str(name).upper(): float(value) for name, value in raw_thresholds.items()}
    metrics = data.get("metrics", data.get("selected_metrics", {}))
    if not isinstance(metrics, dict):
        metrics = {}

    def get_metric(name):
        if name in metrics:
            return metrics[name]
        upper = name.upper()
        if upper in metrics:
            return metrics[upper]
        return None

    return {
        "operating_thresholds": thresholds,
        "tp": get_metric("tp"),
        "tn": get_metric("tn"),
        "fp": get_metric("fp"),
        "fn": get_metric("fn"),
        "source": PHASE2_EVIDENCE,
    }


def threshold_for_scenario(decision, scenario):
    """Return the recorded Phase 2 operating threshold for this exact scenario."""
    normalized = str(scenario or "").upper().strip()
    aliases = {
        "S1": "S1_LABEL_FLIP",
        "S2": "S2_BACKDOOR",
        "S3": "S3_PYPI",
        "S4": "S4_MODEL_HUB",
        "S5": "S5_MLFLOW_TAMPERING",
    }
    normalized = aliases.get(normalized, normalized)
    thresholds = decision["operating_thresholds"]
    if normalized not in thresholds:
        raise RuntimeError(
            f"No Phase 2 operating threshold is recorded for scenario {scenario!r}. "
            f"Available scenarios: {sorted(thresholds)}"
        )
    return normalized, float(thresholds[normalized])


# ============================================================
# MLFLOW CONNECTION
# ============================================================

def connect_mlflow():

    print("\nConnecting to MLflow...")
    print(f"Tracking URI: {MLFLOW_URI}")

    mlflow.set_tracking_uri(MLFLOW_URI)

    client = MlflowClient(
        tracking_uri=MLFLOW_URI
    )

    # Test connection
    client.search_registered_models()

    print("MLflow connection: OK")

    return client


# ============================================================
# FIND REGISTERED MODEL
# ============================================================

def get_model_versions(client):
    try:
        versions = list(client.search_model_versions(f"name='{MODEL_NAME}'"))
    except Exception as e:
        raise RuntimeError(
            f"Unable to retrieve MLflow model '{MODEL_NAME}': {e}"
        ) from e

    if not versions:
        raise RuntimeError(
            f"Registered model '{MODEL_NAME}' exists but has no model versions "
            f"visible through {MLFLOW_URI}."
        )
    return versions


def get_effective_model_tags(client, version):
    """Return registry-version tags, falling back to the originating run tags.

    Phase 1 stores provenance/security tags on the MLflow run. Older versions of
    this framework did not copy every tag onto the registered ModelVersion.
    Reading both locations makes the existing evidence usable and preserves the
    exact run provenance instead of inventing metadata.
    """
    tags = dict(getattr(version, "tags", {}) or {})
    run_id = getattr(version, "run_id", None)
    if run_id:
        try:
            run_tags = dict(client.get_run(run_id).data.tags or {})
            for key, value in run_tags.items():
                tags.setdefault(key, value)
        except Exception:
            # Registry tags remain authoritative if the originating run cannot
            # be retrieved; callers will fail closed when required metadata is
            # absent.
            pass
    return tags


def sync_run_tags_to_registry_version(client, version, tags):
    """Copy missing security/provenance tags onto the ModelVersion."""
    changed = {}
    registry_tags = dict(getattr(version, "tags", {}) or {})
    for key in (
        "research_phase",
        "framework",
        "scenario",
        "attack_scenario",
        "status",
        "deployment_status",
        "model_sha256",
        "model_artifact_path",
        "poison_rate",
        "repetition",
        "attack_started_at",
        "integrity_gate_score",
        "integrity_gate_decision",
        "integrity_gate_control",
        "integrity_gate_evidence_path",
        "integrity_gate_evidence_sha256",
        "controlled_artifact_path",
        "controlled_artifact_sha256",
        "mlflow_artifact_sha256",
        "verified_clean_artifact_path",
        "expected_artifact_sha256",
        "observed_artifact_sha256",
        "artifact_hash_match",
    ):
        value = tags.get(key)
        if value is None:
            continue
        if registry_tags.get(key) != str(value):
            client.set_model_version_tag(
                MODEL_NAME, str(version.version), key, str(value)
            )
            changed[key] = str(value)
    return changed


# ============================================================
# SELECT CURRENT MODEL
# ============================================================

def select_current_model(client, versions):
    """Select the latest explicitly suspect model using registry/run provenance."""
    suspect_versions = []
    for version in versions:
        tags = get_effective_model_tags(client, version)
        if (
            str(tags.get("status", "")).lower() == "suspect"
            and str(tags.get("deployment_status", "")).upper() in {"STAGING", "PRODUCTION"}
        ):
            suspect_versions.append(version)
    if not suspect_versions:
        raise RuntimeError(
            f"No suspect versions found for MLflow model '{MODEL_NAME}'."
        )
    return max(suspect_versions, key=lambda x: int(x.version))


# ============================================================
# FIND VERIFIED CLEAN VERSION
# ============================================================

def find_verified_clean_version(client, versions):
    """Find the newest explicitly verified-clean baseline version."""
    clean_candidates = []

    for version in versions:
        tags = get_effective_model_tags(client, version)
        status = str(tags.get("status", "")).lower()
        attack = str(tags.get("attack_scenario", "")).lower()

        if status == "verified-clean" and attack == "none":
            clean_candidates.append(version)

    if not clean_candidates:
        return None

    clean_candidates = sorted(
        clean_candidates,
        key=lambda x: int(x.version),
        reverse=True
    )

    return clean_candidates[0]


# ============================================================
# MLflow BLOCK
# ============================================================

def block_model(client, version):

    start = time.perf_counter()

    version_number = str(version.version)

    print("\n[RESPONSE 1] BLOCK")

    try:

        # Add security tags first.
        client.set_model_version_tag(
            MODEL_NAME,
            version_number,
            "security_status",
            "BLOCKED"
        )
        client.set_model_version_tag(
            MODEL_NAME,
            version_number,
            "security_block_status",
            "BLOCKED"
        )

        client.set_model_version_tag(
            MODEL_NAME,
            version_number,
            "blocked_by",
            "phase3_automated_response"
        )

        client.set_model_version_tag(
            MODEL_NAME,
            version_number,
            "blocked_at",
            iso_now()
        )
        client.set_model_version_tag(
            MODEL_NAME,
            version_number,
            "deployment_status",
            "BLOCKED"
        )

        # Research deployment gate: promotion wrappers can consult this manifest
        # before allowing any model to become active. This is a real gate artifact,
        # not a claim that a serving process was stopped by MLflow itself.
        gate_file = os.path.join(RESULTS_DIR, "deployment_blocklist.jsonl")
        with open(gate_file, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "model_name": MODEL_NAME,
                "model_version": version_number,
                "blocked_at": iso_now(),
                "reason": "Phase 2 malicious detection",
            }) + "\n")

        success = True
        error = None

    except Exception as e:

        success = False
        error = str(e)

    elapsed = time.perf_counter() - start

    print(f"Success: {success}")
    print(f"Time: {elapsed:.4f}s")

    return {
        "action": "BLOCK",
        "success": success,
        "elapsed_seconds": elapsed,
        "error": error
    }


# ============================================================
# MLflow QUARANTINE
# ============================================================

def quarantine_model(client, version):

    start = time.perf_counter()

    version_number = str(version.version)

    print("\n[RESPONSE 2] QUARANTINE")

    try:

        client.set_model_version_tag(
            MODEL_NAME,
            version_number,
            "security_quarantine_status",
            "QUARANTINED"
        )

        client.set_model_version_tag(
            MODEL_NAME,
            version_number,
            "quarantine_reason",
            "Phase 2 malicious detection"
        )

        client.set_model_version_tag(
            MODEL_NAME,
            version_number,
            "quarantined_at",
            iso_now()
        )
        client.set_model_version_tag(
            MODEL_NAME,
            version_number,
            "deployment_status",
            "QUARANTINED"
        )

        success = True
        error = None

    except Exception as e:

        success = False
        error = str(e)

    elapsed = time.perf_counter() - start

    print(f"Success: {success}")
    print(f"Time: {elapsed:.4f}s")

    return {
        "action": "QUARANTINE",
        "success": success,
        "elapsed_seconds": elapsed,
        "error": error
    }


# ============================================================
# ROLLBACK
# ============================================================

def rollback_model(client, clean_version):

    start = time.perf_counter()

    print("\n[RESPONSE 3] ROLLBACK")

    if clean_version is None:

        print(
            "No verified-clean MLflow version "
            "was found. Rollback NOT performed."
        )

        return {
            "action": "ROLLBACK",
            "success": False,
            "elapsed_seconds": 0,
            "error": "No verified-clean model version found"
        }

    clean_version_number = str(
        clean_version.version
    )

    try:

        client.set_model_version_tag(
            MODEL_NAME,
            clean_version_number,
            "security_status",
            "RESTORED_PRODUCTION"
        )

        client.set_model_version_tag(
            MODEL_NAME,
            clean_version_number,
            "restored_at",
            iso_now()
        )

        client.set_model_version_tag(
            MODEL_NAME,
            clean_version_number,
            "restored_by",
            "phase3_automated_response"
        )
        client.set_model_version_tag(
            MODEL_NAME,
            clean_version_number,
            "deployment_status",
            "PRODUCTION"
        )

        # A registry alias is an explicit deployment pointer. Moving the
        # production alias to the verified-clean version is stronger evidence
        # than tags alone and can be consumed by deployment wrappers.
        client.set_registered_model_alias(
            MODEL_NAME,
            ACTIVE_DEPLOYMENT_ALIAS,
            clean_version_number,
        )
        alias_target = client.get_model_version_by_alias(
            MODEL_NAME,
            ACTIVE_DEPLOYMENT_ALIAS,
        )
        if str(alias_target.version) != clean_version_number:
            raise RuntimeError(
                f"Rollback alias verification failed: expected v{clean_version_number}, "
                f"got v{alias_target.version}."
            )

        success = True
        error = None

    except Exception as e:

        success = False
        error = str(e)

    elapsed = time.perf_counter() - start

    print(
        f"Restored version: "
        f"{clean_version_number}"
    )

    print(f"Success: {success}")
    print(f"Time: {elapsed:.4f}s")

    return {
        "action": "ROLLBACK",
        "success": success,
        "elapsed_seconds": elapsed,
        "restored_version": clean_version_number,
        "deployment_alias": ACTIVE_DEPLOYMENT_ALIAS,
        "deployment_alias_verified": bool(success),
        "error": error
    }


# ============================================================
# WAZUH EVENT
#
# We create a structured JSON event locally first.
# Wazuh can ingest the event once the corresponding
# integration/decoder is configured.
# ============================================================

def create_wazuh_event(
    model_version,
    detection_score,
    detection_time,
    response_results,
    mttd,
    mttd_basis,
    attack_start_timestamp,
    detection_wall_timestamp,
    detection_compute_seconds,
    mttr,
    detection_threshold
):

    event = {

        "event_type":
            "MLFLOW_MODEL_SECURITY_INCIDENT",

        "timestamp":
            iso_now(),

        "hostname":
            socket.gethostname(),

        "framework":
            "secure_mlflow_framework",

        "model_name":
            MODEL_NAME,

        "model_version":
            str(model_version),

        "detection_score":
            detection_score,

        "detection_threshold":
            detection_threshold,

        "detection_decision":
            "MALICIOUS",

        "mttd_seconds":
            mttd,
        "mttd_definition": "attack_start_to_detection_decision",
        "mttd_basis": mttd_basis,
        "attack_start_timestamp": attack_start_timestamp,
        "detection_timestamp": detection_wall_timestamp,
        "detection_compute_seconds": detection_compute_seconds,

        "mttr_seconds":
            mttr,

        "response":
            response_results,

        "severity":
            12,

        "research_phase":
            "PHASE_3",

        "source":
            "Phase 2 detection + automated response"
    }

    with open(
        WAZUH_EVENT_FILE,
        "a",
        encoding="utf-8"
    ) as f:

        f.write(
            json.dumps(event) + "\n"
        )

    return event


# ============================================================
# OPTIONAL WAZUH API TEST
# ============================================================

def wazuh_tls_configuration():
    """Return requests' verify value and fail closed for insecure production mode."""
    if WAZUH_PRODUCTION_MODE and (not WAZUH_VERIFY_TLS or not WAZUH_CA_BUNDLE):
        raise RuntimeError(
            "Production Wazuh mode requires WAZUH_VERIFY_TLS=true and WAZUH_CA_BUNDLE."
        )
    if not WAZUH_VERIFY_TLS:
        return False
    if WAZUH_CA_BUNDLE:
        if not os.path.isfile(WAZUH_CA_BUNDLE):
            raise RuntimeError(f"WAZUH_CA_BUNDLE does not exist: {WAZUH_CA_BUNDLE}")
        return WAZUH_CA_BUNDLE
    return True


def wazuh_request(method, url, **kwargs):
    """HTTP request with bounded exponential backoff."""
    last_error = None
    for attempt in range(1, WAZUH_RETRY_ATTEMPTS + 1):
        try:
            response = requests.request(method, url, **kwargs)
            response.raise_for_status()
            return response, attempt
        except Exception as exc:
            last_error = exc
            if attempt < WAZUH_RETRY_ATTEMPTS:
                time.sleep(WAZUH_RETRY_BACKOFF_SECONDS * (2 ** (attempt - 1)))
    raise RuntimeError(
        f"Wazuh request failed after {WAZUH_RETRY_ATTEMPTS} attempts: {last_error}"
    ) from last_error

def test_wazuh_api():
    """Authenticate to the real Wazuh API and query manager info.

    Credentials come only from environment variables. TLS verification is on by
    default. A self-signed lab certificate can be explicitly allowed with
    WAZUH_VERIFY_TLS=false; the result records that security downgrade.
    """
    print("\nTesting Wazuh API authentication...")

    if not WAZUH_USER or not WAZUH_PASSWORD:
        return {
            "reachable": False,
            "authenticated": False,
            "verified": False,
            "reason": "WAZUH_USER/WAZUH_PASSWORD not configured in environment",
        }

    login_url = WAZUH_API.rstrip("/") + "/security/user/authenticate"
    try:
        tls_verify = wazuh_tls_configuration()
        response, auth_attempts = wazuh_request(
            "POST",
            login_url,
            auth=(WAZUH_USER, WAZUH_PASSWORD),
            params={"raw": "true"},
            verify=tls_verify,
            timeout=WAZUH_AUTH_TIMEOUT_SECONDS,
        )
        token = response.text.strip().strip('"')
        if not token:
            try:
                token = response.json()["data"]["token"]
            except Exception:
                token = ""
        if not token:
            raise RuntimeError("Wazuh authentication response did not contain a JWT.")
    except Exception as exc:
        return {
            "reachable": False,
            "authenticated": False,
            "verified": bool(WAZUH_VERIFY_TLS),
            "production_mode": WAZUH_PRODUCTION_MODE,
            "ca_bundle": WAZUH_CA_BUNDLE or None,
            "authentication_error": str(exc),
        }

    # Authentication and the optional manager-information query are separate
    # checks. A slow information endpoint must not erase a successful login.
    result = {
        "reachable": True,
        "authenticated": True,
        "verified": bool(WAZUH_VERIFY_TLS),
        "production_mode": WAZUH_PRODUCTION_MODE,
        "ca_bundle": WAZUH_CA_BUNDLE or None,
        "authentication_attempts": auth_attempts,
        "authentication_http_status": response.status_code,
        "manager_info_reachable": False,
    }
    try:
        info, info_attempts = wazuh_request(
            "GET",
            WAZUH_API.rstrip("/") + "/",
            headers={"Authorization": f"Bearer {token}"},
            verify=tls_verify,
            timeout=WAZUH_API_TIMEOUT_SECONDS,
        )
        result.update({
            "manager_info_reachable": True,
            "manager_info_http_status": info.status_code,
            "manager_info_attempts": info_attempts,
        })
    except Exception as exc:
        result["manager_info_error"] = str(exc)

    return result


def discover_wazuh_container():
    """Discover a running Wazuh manager container when env config is absent."""
    if WAZUH_DOCKER_CONTAINER:
        return WAZUH_DOCKER_CONTAINER
    try:
        completed = subprocess.run(
            ["docker", "ps", "--format", "{{.Names}}"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if completed.returncode != 0:
            return ""
        names = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
        for name in names:
            lowered = name.lower()
            if "wazuh.manager" in lowered or "wazuh-manager" in lowered or lowered.endswith("wazuh_manager"):
                return name
    except Exception:
        pass
    return ""


def ingest_event_to_wazuh_container(event):
    """Send the structured event to a Wazuh manager JSON localfile when available."""
    container = discover_wazuh_container()
    if not container:
        return {
            "enabled": False,
            "ingested": False,
            "reason": "No running Wazuh manager container configured/discovered",
        }

    payload = json.dumps(event, separators=(",", ":")) + "\n"
    log_path = WAZUH_MANAGER_LOG_PATH
    script = (
        f"mkdir -p {shlex.quote(os.path.dirname(log_path) or '/')}; "
        f"touch {shlex.quote(log_path)}; "
        f"chown wazuh:wazuh {shlex.quote(log_path)} 2>/dev/null || true; "
        f"chmod 0640 {shlex.quote(log_path)} 2>/dev/null || true; "
        f"cat >> {shlex.quote(log_path)}"
    )
    last_error = None
    for attempt in range(1, WAZUH_RETRY_ATTEMPTS + 1):
        try:
            completed = subprocess.run(
                ["docker", "exec", "-i", container, "bash", "-lc", script],
                input=payload,
                capture_output=True,
                text=True,
                timeout=WAZUH_INGEST_TIMEOUT_SECONDS,
                check=False,
            )
            if completed.returncode == 0:
                return {
                    "enabled": True,
                    "ingested": True,
                    "container": container,
                    "manager_log_path": log_path,
                    "attempts": attempt,
                    "stdout": completed.stdout[-1000:],
                    "stderr": completed.stderr[-1000:],
                }
            last_error = completed.stderr[-1000:]
        except Exception as exc:
            last_error = str(exc)
        if attempt < WAZUH_RETRY_ATTEMPTS:
            time.sleep(WAZUH_RETRY_BACKOFF_SECONDS * (2 ** (attempt - 1)))
    return {
        "enabled": True,
        "ingested": False,
        "container": container,
        "attempts": WAZUH_RETRY_ATTEMPTS,
        "infrastructure_failure": True,
        "error": last_error,
    }


def verify_recent_wazuh_alert(model_version, timeout_seconds=None, poll_seconds=2):
    """Poll Wazuh until the matching Phase 3 alert is observed or timeout occurs."""

    container = discover_wazuh_container()

    if not container:
        return {
            "checked": False,
            "alert_observed": False,
            "reason": "No running Wazuh manager container configured/discovered",
        }

    timeout_seconds = WAZUH_ALERT_TIMEOUT_SECONDS if timeout_seconds is None else timeout_seconds
    deadline = time.monotonic() + timeout_seconds
    last_error = None

    while time.monotonic() < deadline:
        try:
            completed = subprocess.run(
                [
                    "docker",
                    "exec",
                    container,
                    "bash",
                    "-lc",
                    "tail -n 500 /var/ossec/logs/alerts/alerts.json 2>/dev/null",
                ],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )

            if completed.returncode != 0:
                last_error = completed.stderr[-1000:]
            else:
                matching = []

                for line in completed.stdout.splitlines():
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    data = record.get("data", {})

                    event_type = (
                        data.get("event_type")
                        or record.get("event_type")
                    )

                    candidate_version = (
                        data.get("model_version")
                        or record.get("model_version")
                    )

                    if (
                        event_type == "MLFLOW_MODEL_SECURITY_INCIDENT"
                        and str(candidate_version) == str(model_version)
                    ):
                        rule = record.get("rule", {})

                        rule_id = rule.get("id") if isinstance(rule, dict) else None
                        rule_level = rule.get("level") if isinstance(rule, dict) else None
                        if (
                            str(rule_id) != str(WAZUH_EXPECTED_RULE_ID)
                            or int(rule_level or -1) != WAZUH_EXPECTED_RULE_LEVEL
                        ):
                            continue

                        matching.append({
                            "rule_id": rule_id,
                            "rule_level": rule_level,
                            "timestamp": record.get("timestamp"),
                        })

                if matching:
                    return {
                        "checked": True,
                        "alert_observed": True,
                        "container": container,
                        "model_version": str(model_version),
                        "matching_alerts": matching[-5:],
                    }

        except Exception as exc:
            last_error = str(exc)

        time.sleep(poll_seconds)

    return {
        "checked": True,
        "alert_observed": False,
        "container": container,
        "model_version": str(model_version),
        "timeout_seconds": timeout_seconds,
        "expected_rule_id": WAZUH_EXPECTED_RULE_ID,
        "expected_rule_level": WAZUH_EXPECTED_RULE_LEVEL,
        "infrastructure_failure": bool(last_error),
        "error": last_error,
    }
# ============================================================
# SAVE FINAL EVIDENCE
# ============================================================

def _write_json(path, value):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, default=str)


def _evidence_run_filename(evidence):
    timestamp = str(evidence.get("timestamp", iso_now()))
    timestamp = re.sub(r"[^0-9A-Za-z]+", "-", timestamp).strip("-")
    scenario = re.sub(r"[^0-9A-Za-z_]+", "_", str(evidence.get("attack_scenario", "UNKNOWN")))
    version = re.sub(r"[^0-9A-Za-z_]+", "_", str(evidence.get("suspect_version", "UNKNOWN")))
    return f"phase3_response_{scenario}_v{version}_{timestamp}.json"


def _preserve_legacy_latest_evidence():
    """Preserve the pre-per-run evidence file from an earlier Phase 3 run once."""
    if not os.path.exists(LATEST_EVIDENCE_FILE):
        return
    try:
        with open(LATEST_EVIDENCE_FILE, "r", encoding="utf-8") as f:
            legacy = json.load(f)
        if not isinstance(legacy, dict) or legacy.get("phase") != "PHASE_3":
            return
        destination = os.path.join(RUN_EVIDENCE_DIR, _evidence_run_filename(legacy))
        if not os.path.exists(destination):
            _write_json(destination, legacy)
    except (OSError, json.JSONDecodeError):
        # A legacy file is supplementary evidence; a bad legacy file must not
        # prevent preservation of the current response evidence.
        return


def save_evidence(evidence):
    """Save immutable per-run evidence and a convenient latest-result copy."""
    _preserve_legacy_latest_evidence()
    base_path = os.path.join(RUN_EVIDENCE_DIR, _evidence_run_filename(evidence))
    path = base_path
    suffix = 2
    while os.path.exists(path):
        stem, extension = os.path.splitext(base_path)
        path = f"{stem}_{suffix}{extension}"
        suffix += 1
    _write_json(path, evidence)
    _write_json(LATEST_EVIDENCE_FILE, evidence)
    return path


def _numeric_or_none(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def phase3_summary_main():
    """Create reproducible per-run and aggregate MTTD/MTTR evidence."""
    _preserve_legacy_latest_evidence()
    records = []
    for path in sorted(Path(RUN_EVIDENCE_DIR).glob("phase3_response_*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict) or data.get("phase") != "PHASE_3":
            continue
        response = data.get("response", {}) if isinstance(data.get("response"), dict) else {}
        stored_mttd = _numeric_or_none(data.get("mttd_seconds"))
        detection_compute = _numeric_or_none(data.get("detection_compute_seconds"))
        mttd_definition = data.get("mttd_definition")

        if mttd_definition == "attack_start_to_detection_decision":
            # Evidence generated after the timing fix already stores decision MTTD.
            decision_mttd = stored_mttd
            detector_start_mttd = (
                max(0.0, stored_mttd - detection_compute)
                if stored_mttd is not None and detection_compute is not None
                else None
            )
            reconstruction = "DIRECT_DECISION_MTTD"
        else:
            # Historical Phase 3 evidence captured the wall-clock timestamp
            # immediately before detector computation. Preserve that original
            # interval and reconstruct attack-to-decision time by adding the
            # independently measured detector-compute duration.
            detector_start_mttd = stored_mttd
            decision_mttd = (
                stored_mttd + detection_compute
                if stored_mttd is not None and detection_compute is not None
                else stored_mttd
            )
            reconstruction = "LEGACY_START_MTTD_PLUS_DETECTION_COMPUTE"
        records.append({
            "evidence_file": str(path),
            "timestamp": data.get("timestamp"),
            "scenario": data.get("attack_scenario"),
            "suspect_version": str(data.get("suspect_version")),
            "mttd_seconds": decision_mttd,
            "mttd_to_detector_start_seconds": detector_start_mttd,
            "mttd_reconstruction": reconstruction,
            "mttd_basis": data.get("mttd_basis"),
            "detection_compute_seconds": detection_compute,
            "mttr_seconds": _numeric_or_none(data.get("mttr_seconds")),
            "block_success": bool(response.get("block", {}).get("success", False)),
            "quarantine_success": bool(response.get("quarantine", {}).get("success", False)),
            "rollback_success": bool(response.get("rollback", {}).get("success", False)),
            "wazuh_alert_observed": bool(data.get("wazuh_alert_verification", {}).get("alert_observed", False)),
        })

    if not records:
        raise RuntimeError(f"No per-run Phase 3 evidence files found in {RUN_EVIDENCE_DIR}.")

    fields = list(records[0].keys())
    with open(PHASE3_SUMMARY_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)

    grouped = {}
    for record in records:
        grouped.setdefault(record["scenario"] or "UNKNOWN", []).append(record)

    grouped_by_basis = {}
    for record in records:
        key = f"{record['scenario'] or 'UNKNOWN'}|{record['mttd_basis'] or 'UNKNOWN'}"
        grouped_by_basis.setdefault(key, []).append(record)

    def describe(values):
        values = [float(v) for v in values if v is not None]
        if not values:
            return {"count": 0, "mean": None, "std_dev": None, "min": None, "max": None}
        return {
            "count": len(values),
            "mean": float(statistics.fmean(values)),
            "std_dev": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
            "min": float(min(values)),
            "max": float(max(values)),
        }

    by_scenario = {}
    for scenario, rows in grouped.items():
        by_scenario[scenario] = {
            "runs": len(rows),
            "mttd_seconds": describe([r["mttd_seconds"] for r in rows]),
            "mttd_to_detector_start_seconds": describe(
                [r["mttd_to_detector_start_seconds"] for r in rows]
            ),
            "detection_compute_seconds": describe([r["detection_compute_seconds"] for r in rows]),
            "mttr_seconds": describe([r["mttr_seconds"] for r in rows]),
            "all_mlflow_response_actions_successful": sum(
                r["block_success"] and r["quarantine_success"] and r["rollback_success"] for r in rows
            ),
            "verified_wazuh_alerts": sum(r["wazuh_alert_observed"] for r in rows),
        }

    by_scenario_and_mttd_basis = {}
    for key, rows in grouped_by_basis.items():
        by_scenario_and_mttd_basis[key] = {
            "scenario": rows[0]["scenario"],
            "mttd_basis": rows[0]["mttd_basis"],
            "runs": len(rows),
            "mttd_seconds": describe([r["mttd_seconds"] for r in rows]),
            "mttd_to_detector_start_seconds": describe(
                [r["mttd_to_detector_start_seconds"] for r in rows]
            ),
            "detection_compute_seconds": describe([r["detection_compute_seconds"] for r in rows]),
            "mttr_seconds": describe([r["mttr_seconds"] for r in rows]),
        }

    summary = {
        "phase": "PHASE_3_RESPONSE_REPETITION_SUMMARY",
        "generated_at": iso_now(),
        "run_evidence_directory": RUN_EVIDENCE_DIR,
        "total_runs": len(records),
        "by_scenario": by_scenario,
        "by_scenario_and_mttd_basis": by_scenario_and_mttd_basis,
        "notes": {
            "mttd": "mttd_seconds is attack-start to detection-decision. For historical evidence it is reconstructed as the saved attack-to-detector-start interval plus detection_compute_seconds. The original interval is preserved as mttd_to_detector_start_seconds. Compare only runs with the same mttd_basis.",
            "mttr": "Current MTTR covers MLflow block, quarantine, and rollback; live Wazuh alert verification is reported separately.",
            "target_repetitions": 5,
        },
        "files": {"runs_csv": PHASE3_SUMMARY_CSV},
    }
    _write_json(PHASE3_SUMMARY_JSON, summary)
    print(f"Phase 3 per-run CSV: {PHASE3_SUMMARY_CSV}")
    print(f"Phase 3 summary JSON: {PHASE3_SUMMARY_JSON}")
    return summary


def phase3_clean_control_main(clean_version_number=None):
    """Record a non-destructive Phase 3 clean control.

    This command never calls block_model, quarantine_model, or rollback_model.
    It proves that an explicitly verified-clean model has no block/quarantine
    state before recording the result as separate evidence.
    """
    print_section("PHASE 3 — CLEAN RESPONSE CONTROL")
    client = connect_mlflow()
    versions = get_model_versions(client)
    if clean_version_number is not None:
        matches = [v for v in versions if int(v.version) == int(clean_version_number)]
        if not matches:
            raise RuntimeError(f"Requested clean version {clean_version_number} was not found.")
        clean_version = matches[0]
    else:
        clean_version = find_verified_clean_version(client, versions)
    if clean_version is None:
        raise RuntimeError("No explicitly verified-clean MLflow version is available for the control.")

    tags = get_effective_model_tags(client, clean_version)
    if str(tags.get("status", "")).lower() != "verified-clean":
        raise RuntimeError("Clean control requires status=verified-clean; refusing to test an unverified version.")
    blocked = str(tags.get("security_block_status", "")).upper() == "BLOCKED"
    quarantined = str(tags.get("security_quarantine_status", "")).upper() == "QUARANTINED"
    deployment_status = str(tags.get("deployment_status", "")).upper()
    unsafe_deployment_state = deployment_status in {"BLOCKED", "QUARANTINED"}
    passed = not (blocked or quarantined or unsafe_deployment_state)

    alias_version = None
    alias_error = None
    try:
        alias_version = str(client.get_model_version_by_alias(MODEL_NAME, ACTIVE_DEPLOYMENT_ALIAS).version)
    except Exception as exc:
        alias_error = str(exc)

    evidence = {
        "phase": "PHASE_3_CLEAN_CONTROL",
        "timestamp": iso_now(),
        "model_name": MODEL_NAME,
        "clean_version": str(clean_version.version),
        "clean_status": tags.get("status"),
        "deployment_status": deployment_status,
        "security_block_status": tags.get("security_block_status"),
        "security_quarantine_status": tags.get("security_quarantine_status"),
        "production_alias_version": alias_version,
        "production_alias_error": alias_error,
        "response_actions_attempted": [],
        "block_attempted": False,
        "quarantine_attempted": False,
        "rollback_attempted": False,
        "passed": passed,
        "failure_reason": None if passed else "Clean model had a block/quarantine deployment state.",
    }
    filename = f"phase3_clean_control_v{clean_version.version}_{re.sub(r'[^0-9A-Za-z]+', '-', evidence['timestamp']).strip('-')}.json"
    evidence_path = os.path.join(CLEAN_CONTROL_DIR, filename)
    _write_json(evidence_path, evidence)
    print(f"Clean model: v{clean_version.version}")
    print(f"No response actions were executed: {passed}")
    print(f"Evidence: {evidence_path}")
    if not passed:
        raise RuntimeError(evidence["failure_reason"])
    return evidence


def phase3_final_results_main():
    """Produce one dissertation-ready response table including clean controls."""
    summary = phase3_summary_main()
    controls = []
    for path in sorted(Path(CLEAN_CONTROL_DIR).glob("phase3_clean_control_*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if data.get("phase") == "PHASE_3_CLEAN_CONTROL":
            controls.append({
                "evidence_file": str(path),
                "timestamp": data.get("timestamp"),
                "clean_version": data.get("clean_version"),
                "passed": bool(data.get("passed", False)),
                "block_attempted": bool(data.get("block_attempted", False)),
                "quarantine_attempted": bool(data.get("quarantine_attempted", False)),
                "rollback_attempted": bool(data.get("rollback_attempted", False)),
            })
    if controls:
        with open(PHASE3_CLEAN_CONTROL_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(controls[0].keys()))
            writer.writeheader()
            writer.writerows(controls)
    final = {
        "phase": "PHASE_3_FINAL_RESULTS",
        "generated_at": iso_now(),
        "malicious_response_summary": summary,
        "clean_controls": {
            "runs": len(controls),
            "passed": sum(bool(c["passed"]) for c in controls),
            "false_response_runs": sum(
                bool(c["block_attempted"]) or bool(c["quarantine_attempted"]) or bool(c["rollback_attempted"])
                for c in controls
            ),
            "csv": PHASE3_CLEAN_CONTROL_CSV if controls else None,
        },
    }
    path = os.path.join(RESULTS_DIR, "phase3_final_results.json")
    _write_json(path, final)
    print(f"Phase 3 final results: {path}")
    return final


# ============================================================
# REAL SUSPECT-MODEL DETECTION SCORE
# ============================================================

def compute_real_detection_score(client, version, threshold):
    """
    Re-run the real Phase 2 detection logic on the exact suspect model
    represented by the MLflow version. No hard-coded detection score.

    Currently supports the two model-based attacks generated by Phase 1:
    S1_LABEL_FLIP and S2_BACKDOOR. Other registered attack versions are
    rejected rather than assigned a fabricated score.
    """
    from joblib import load
    import numpy as np
    import pandas as pd
    import research_pipeline_A_final as pm

    tags = get_effective_model_tags(client, version)
    scenario = str(tags.get("attack_scenario", "")).upper()
    poison_rate = tags.get("poison_rate")

    test_path = os.path.join(PROJECT_ROOT, "data", "processed", "split", "clean_test.csv")
    test_df = pd.read_csv(test_path)
    features = [c for c in test_df.columns if c not in {"label", "row_id"}]

    if scenario == "S2_BACKDOOR":
        if poison_rate is None:
            raise RuntimeError("S2 suspect version has no poison_rate tag.")
        rate = float(poison_rate)
        token = f"{int(round(rate * 100)):02d}"
        repetition_raw = tags.get("repetition")
        if repetition_raw is None:
            raise RuntimeError("S2 suspect version has no repetition tag; exact run binding is required.")
        repetition = int(repetition_raw)
        model_path = os.path.join(
            PROJECT_ROOT, "models", "S2_backdoor",
            f"backdoor_model_rate{token}_run{repetition:02d}.joblib"
        )
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Exact S2 run model not found: {model_path}")

        if tags.get("model_sha256"):
            local_hash = sha256_file(model_path)
            if tags["model_sha256"] != local_hash:
                raise RuntimeError(
                    f"MLflow artifact SHA-256 mismatch for version {version.version}: "
                    f"registry={tags['model_sha256']}, local={local_hash}"
                )
        bundle = load(model_path)
        model = bundle["model"]
        scaler = bundle["scaler"]
        model_features = bundle.get("features", features)
        X_frame = test_df[model_features].replace([np.inf, -np.inf], np.nan).fillna(0).astype("float32")
        X = scaler.transform(X_frame)

        cfg = pm.load_cfg()
        nc = pm.neural_cleanse_style_score(
            model,
            X,
            cfg["attacks"]["target_label"],
            cfg["detection"]["nc_steps"],
            cfg["detection"]["nc_learning_rate"],
        )
        strip = pm.strip_entropy_score(
            model,
            X,
            cfg["detection"]["strip_samples"],
        )
        cluster = pm.activation_clustering_score(
            model,
            X,
            cfg["detection"]["cluster_min_samples"],
        )
        fusion = pm.fuse(
            nc["score"], strip["score"], cluster["score"], cfg
        )
        return float(fusion["risk_score"]), {
            "scenario": scenario,
            "poison_rate": rate,
            "model_path": model_path,
            "neural_cleanse": nc,
            "strip": strip,
            "activation_clustering": cluster,
            "fusion": fusion,
        }

    if scenario == "S1_LABEL_FLIP":
        if poison_rate is None:
            raise RuntimeError("S1 suspect version has no poison_rate tag.")
        rate = float(poison_rate)
        token = f"{int(round(rate * 100)):02d}"
        repetition_raw = tags.get("repetition")
        if repetition_raw is None:
            raise RuntimeError("S1 suspect version has no repetition tag; exact run binding is required.")
        repetition = int(repetition_raw)
        model_path = os.path.join(
            PROJECT_ROOT, "models", "S1_label_flip",
            f"label_flip_model_rate{token}_run{repetition:02d}.joblib"
        )
        samples_path = os.path.join(
            PROJECT_ROOT, "evidence", "phase1", "S1_label_flip",
            f"label_flip_samples_rate{token}_run{repetition:02d}.csv"
        )
        training_path = os.path.join(
            PROJECT_ROOT, "evidence", "phase1", "S1_label_flip",
            f"label_flip_training_rate{token}_run{repetition:02d}.csv"
        )
        clean_training_path = os.path.join(
            PROJECT_ROOT, "data", "processed", "split", "clean_train.csv"
        )
        for required in (model_path, samples_path, training_path, clean_training_path):
            if not os.path.exists(required):
                raise FileNotFoundError(f"Exact S1 run artifact not found: {required}")

        result = pm.s1_detector(
            threshold,
            samples_path=samples_path,
            model_path=model_path,
            training_path=training_path,
            clean_training_path=clean_training_path,
        )
        result["model_path"] = model_path
        result["samples_path"] = samples_path
        result["training_path"] = training_path
        result["repetition"] = repetition
        return float(result["risk_score"]), result

    # S3/S4/S5 are integrity-gate scenarios, not neural-network poisoning
    # attacks.  Their score comes only from a saved, hash-verified gate result;
    # this deliberately refuses to invent a detector score from registry tags.
    gate_control_by_scenario = {
        "S3_PYPI": "SBOM",
        "S4_MODEL_HUB": "PICKLE",
        "S5_MLFLOW_TAMPERING": "MLFLOW_HASH",
    }
    if scenario in gate_control_by_scenario:
        required_control = gate_control_by_scenario[scenario]
        control = str(tags.get("integrity_gate_control", "")).upper()
        decision = str(tags.get("integrity_gate_decision", "")).upper()
        evidence_path = str(tags.get("integrity_gate_evidence_path", ""))
        expected_hash = str(tags.get("integrity_gate_evidence_sha256", "")).lower()
        if control != required_control:
            raise RuntimeError(
                f"{scenario} requires integrity_gate_control={required_control}; got {control!r}."
            )
        if decision in {"", "MATCH", "CLEAN", "SAFE", "NORMAL"}:
            raise RuntimeError(
                f"{scenario} is missing a malicious integrity-gate decision; got {decision!r}."
            )
        if not evidence_path or not os.path.isfile(evidence_path):
            raise RuntimeError("Integrity-gate evidence file is missing; Phase 3 fails closed.")
        actual_hash = sha256_file(evidence_path)
        if not expected_hash or expected_hash != actual_hash:
            raise RuntimeError(
                "Integrity-gate evidence SHA-256 is missing or does not match the saved evidence file."
            )
        artifact_path = str(tags.get("controlled_artifact_path", ""))
        artifact_hash = str(tags.get("controlled_artifact_sha256", "")).lower()
        expected_artifact_hash = str(tags.get("expected_artifact_sha256", "")).lower()
        registered_observed_hash = str(tags.get("observed_artifact_sha256", "")).lower()
        mlflow_registered_hash = str(tags.get("mlflow_artifact_sha256", "")).lower()
        hash_match_tag = str(tags.get("artifact_hash_match", "")).lower()

        if scenario == "S5_MLFLOW_TAMPERING":
            if decision != "HASH_MISMATCH":
                raise RuntimeError(
                    f"S5 requires integrity_gate_decision=HASH_MISMATCH; got {decision!r}."
                )
            if not expected_artifact_hash or not registered_observed_hash:
                raise RuntimeError("S5 expected/observed artifact SHA-256 tags are missing.")
            if hash_match_tag not in {"false", "0"}:
                raise RuntimeError("S5 artifact_hash_match must explicitly record false.")

            # Download the exact artifact referenced by the MLflow ModelVersion.
            # This avoids trusting only the original local controlled-artifact path.
            downloaded_path = mlflow.artifacts.download_artifacts(
                artifact_uri=str(version.source),
                tracking_uri=MLFLOW_URI,
            )
            if not os.path.isfile(downloaded_path):
                raise RuntimeError(
                    f"S5 MLflow ModelVersion source did not resolve to a file: {downloaded_path}"
                )
            observed_artifact_hash = sha256_file(downloaded_path).lower()
            if observed_artifact_hash != registered_observed_hash:
                raise RuntimeError(
                    "S5 downloaded MLflow artifact hash does not match the registered observed hash."
                )
            if mlflow_registered_hash and observed_artifact_hash != mlflow_registered_hash:
                raise RuntimeError(
                    "S5 downloaded MLflow artifact hash does not match the registration-time MLflow hash."
                )
            if observed_artifact_hash == expected_artifact_hash:
                raise RuntimeError(
                    "S5 clean and observed hashes match; tampering was not demonstrated."
                )
            artifact_path = downloaded_path
            artifact_hash = observed_artifact_hash
        else:
            if not artifact_path or not os.path.isfile(artifact_path):
                raise RuntimeError("Controlled integrity artifact is missing; Phase 3 fails closed.")
            if not artifact_hash or sha256_file(artifact_path) != artifact_hash:
                raise RuntimeError("Controlled integrity artifact SHA-256 is missing or does not match.")
        score_raw = tags.get("integrity_gate_score")
        try:
            score = float(score_raw)
        except (TypeError, ValueError) as exc:
            raise RuntimeError("Integrity-gate score is missing or invalid.") from exc
        if not 0.0 <= score <= 1.0:
            raise RuntimeError("Integrity-gate score must be in [0, 1].")
        if score < threshold:
            raise RuntimeError(
                f"Integrity-gate score {score:.6f} is below the recorded threshold {threshold:.6f}."
            )
        return score, {
            "scenario": scenario,
            "detector_type": "HASH_VERIFIED_BINARY_INTEGRITY_GATE",
            "integrity_gate_control": control,
            "integrity_gate_decision": decision,
            "integrity_gate_evidence_path": evidence_path,
            "integrity_gate_evidence_sha256": actual_hash,
            "controlled_artifact_path": artifact_path,
            "controlled_artifact_sha256": artifact_hash,
            "expected_artifact_sha256": expected_artifact_hash or None,
            "observed_artifact_sha256": (
                registered_observed_hash
                if scenario == "S5_MLFLOW_TAMPERING"
                else artifact_hash
            ),
            "artifact_hash_match": (
                False if scenario == "S5_MLFLOW_TAMPERING" else None
            ),
            "mlflow_model_version_source": (
                str(version.source)
                if scenario == "S5_MLFLOW_TAMPERING"
                else None
            ),
        }

    raise RuntimeError(
        f"Phase 3 cannot derive a real detection score for attack scenario '{scenario}'."
    )


# ============================================================
# MAIN
# ============================================================


def parse_attack_start_at(explicit_value, client, version):
    """Resolve an attack-start timestamp without fabricating one.

    Priority:
      1. explicit --attack-start-at supplied by the controlled experiment;
      2. MLflow tag attack_started_at if present;
      3. originating MLflow run start time, explicitly marked as a proxy.
    """
    if explicit_value:
        try:
            dt = datetime.fromisoformat(str(explicit_value).replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc), "EXPLICIT_ATTACK_START"
        except ValueError as exc:
            raise ValueError(f"Invalid --attack-start-at ISO-8601 timestamp: {explicit_value}") from exc

    tags = get_effective_model_tags(client, version)
    tagged = tags.get("attack_started_at")
    if tagged:
        try:
            dt = datetime.fromisoformat(str(tagged).replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc), "MLFLOW_ATTACK_START_TAG"
        except ValueError:
            pass

    run_id = getattr(version, "run_id", None)
    if run_id:
        run = client.get_run(run_id)
        if getattr(run.info, "start_time", None) is not None:
            return datetime.fromtimestamp(run.info.start_time / 1000.0, tz=timezone.utc), "MLFLOW_RUN_START_PROXY"

    raise RuntimeError("True MTTD cannot be calculated: no explicit attack_started_at timestamp or MLflow run start time is available.")


def calculate_pipeline_overhead(baseline_seconds, secured_seconds):
    """Calculate pipeline overhead percentage only when both timings are supplied."""
    if baseline_seconds is None or secured_seconds is None:
        return {"available": False, "reason": "Both baseline and security-enabled pipeline wall times are required."}
    baseline = float(baseline_seconds)
    secured = float(secured_seconds)
    if baseline <= 0:
        raise ValueError("Baseline pipeline seconds must be > 0.")
    return {
        "available": True,
        "baseline_pipeline_seconds": baseline,
        "security_enabled_pipeline_seconds": secured,
        "overhead_seconds": secured - baseline,
        "overhead_percent": ((secured - baseline) / baseline) * 100.0,
    }


def phase3_master_main(suspect_version=None, attack_start_at=None, baseline_pipeline_seconds=None):

    print_section(
        "PHASE 3 — INTEGRATED AUTOMATED RESPONSE"
    )

    overall_start = time.perf_counter()

    # --------------------------------------------------------
    # 1. LOAD PHASE 2
    # --------------------------------------------------------

    print_section(
        "STEP 1 — LOAD PHASE 2 FINAL EVIDENCE"
    )

    phase2 = load_phase2_evidence()

    decision = extract_phase2_decision(
        phase2
    )

    print("Phase 2 threshold policy: scenario-specific operating thresholds")

    print(
        f"Phase 2 TP: {decision['tp']}"
    )

    print(
        f"Phase 2 TN: {decision['tn']}"
    )

    print(
        f"Phase 2 FP: {decision['fp']}"
    )

    print(
        f"Phase 2 FN: {decision['fn']}"
    )

    # --------------------------------------------------------
    # 2. MLflow
    # --------------------------------------------------------

    print_section(
        "STEP 2 — CONNECT TO MLFLOW"
    )

    client = connect_mlflow()

    versions = get_model_versions(
        client
    )

    print(
        f"Registered model: {MODEL_NAME}"
    )

    print(
        f"Model versions found: "
        f"{len(versions)}"
    )

    for v in versions:

        print(
            f"  Version {v.version} "
            f"stage={getattr(v, 'current_stage', 'N/A')}"
        )

    # --------------------------------------------------------
    # 3. CURRENT MODEL
    # --------------------------------------------------------

    print_section(
        "STEP 3 — IDENTIFY SUSPECT MODEL"
    )

    if suspect_version is not None:
        matches = [v for v in versions if int(v.version) == int(suspect_version)]
        if not matches:
            raise RuntimeError(f"Requested MLflow version {suspect_version} was not found.")
        current_model = matches[0]
        current_tags = get_effective_model_tags(client, current_model)
        if (
            str(current_tags.get("status", "")).lower() != "suspect"
            or str(current_tags.get("deployment_status", "")).upper() not in {"STAGING", "PRODUCTION"}
        ):
            raise RuntimeError(
                "Requested MLflow version is not a deployed/staging suspect model. "
                f"Effective tags: {current_tags}"
            )
        # Backfill missing registry-version tags from the originating Phase 1 run.
        sync_run_tags_to_registry_version(client, current_model, current_tags)
    else:
        current_model = select_current_model(client, versions)
        current_tags = get_effective_model_tags(client, current_model)
        sync_run_tags_to_registry_version(client, current_model, current_tags)

    print(
        f"Suspect version: "
        f"{current_model.version}"
    )

    scenario, operating_threshold = threshold_for_scenario(
        decision,
        current_tags.get("attack_scenario"),
    )
    print(f"Attack scenario : {scenario}")
    print(f"Operating threshold: {operating_threshold:.8f}")

    # --------------------------------------------------------
    # 4. CLEAN VERSION
    # --------------------------------------------------------

    clean_version = find_verified_clean_version(
        client, versions
    )

    if clean_version:

        print(
            f"Verified-clean rollback target: "
            f"v{clean_version.version}"
        )

    else:

        print(
            "WARNING: No verified-clean "
            "rollback target found."
        )

    # --------------------------------------------------------
    # 5. DETECTION EVENT
    # --------------------------------------------------------

    attack_start_datetime, mttd_basis = parse_attack_start_at(
        attack_start_at, client, current_model
    )
    detection_started_wall_datetime = utc_now()
    detection_timestamp = time.perf_counter()

    detection_score, detection_details = compute_real_detection_score(
        client,
        current_model,
        operating_threshold,
    )

    if detection_score < operating_threshold:
        raise RuntimeError(
            f"Selected suspect model does not meet Phase 2 operating threshold: "
            f"score={detection_score:.6f}, threshold={operating_threshold:.6f}"
        )

    detection_decision_wall_datetime = utc_now()
    detection_time = detection_decision_wall_datetime.isoformat()

    print_section(
        "STEP 4 — MALICIOUS MODEL DETECTED"
    )

    print(
        f"Model: {MODEL_NAME}"
    )

    print(
        f"Version: {current_model.version}"
    )

    print(
        f"Detection score: {detection_score:.6f}"
    )

    print(
        f"Threshold: {operating_threshold:.8f}"
    )

    print(
        "Decision: MALICIOUS"
    )

    # --------------------------------------------------------
    # 6. MTTD
    # --------------------------------------------------------

    response_start = time.perf_counter()

    # Wall-clock MTTD is attack-start -> completed detection decision.
    mttd = (
        detection_decision_wall_datetime -
        attack_start_datetime
    ).total_seconds()

    # Preserve detector execution time as a separate measurement.
    detection_compute_seconds = response_start - detection_timestamp

    # --------------------------------------------------------
    # 7. BLOCK
    # --------------------------------------------------------

    block_result = block_model(
        client,
        current_model
    )

    # --------------------------------------------------------
    # 8. QUARANTINE
    # --------------------------------------------------------

    quarantine_result = quarantine_model(
        client,
        current_model
    )

    # --------------------------------------------------------
    # 9. ROLLBACK
    # --------------------------------------------------------

    rollback_result = rollback_model(
        client,
        clean_version
    )

    # --------------------------------------------------------
    # 10. RESPONSE COMPLETE
    # --------------------------------------------------------

    response_end = time.perf_counter()

    mttr = (
        response_end -
        response_start
    )

    response_results = {

        "block": block_result,

        "quarantine":
            quarantine_result,

        "rollback":
            rollback_result
    }

    # --------------------------------------------------------
    # 11. WAZUH
    # --------------------------------------------------------

    print_section(
        "STEP 5 — WAZUH SIEM EVENT"
    )

    wazuh_event = create_wazuh_event(
        model_version=current_model.version,
        detection_score=detection_score,
        detection_time=detection_time,
        response_results=response_results,
        mttd=mttd,
        mttd_basis=mttd_basis,
        attack_start_timestamp=attack_start_datetime.isoformat(),
        detection_wall_timestamp=detection_decision_wall_datetime.isoformat(),
        detection_compute_seconds=detection_compute_seconds,
        mttr=mttr,
        detection_threshold=operating_threshold
    )

    print(
        f"Wazuh event saved: "
        f"{WAZUH_EVENT_FILE}"
    )

    wazuh_ingest = ingest_event_to_wazuh_container(wazuh_event)
    wazuh_alert = verify_recent_wazuh_alert(current_model.version)

    # --------------------------------------------------------
    # 12. API TEST
    # --------------------------------------------------------

    wazuh_api_status = test_wazuh_api()

    # --------------------------------------------------------
    # 13. FINAL EVIDENCE
    # --------------------------------------------------------

    total_time = (
        time.perf_counter() -
        overall_start
    )

    pipeline_overhead = calculate_pipeline_overhead(
        baseline_pipeline_seconds,
        total_time if baseline_pipeline_seconds is not None else None,
    )

    evidence = {

        "phase":
            "PHASE_3",

        "timestamp":
            iso_now(),

        "phase2_source":
            PHASE2_EVIDENCE,

        "phase2_threshold": operating_threshold,
        "phase2_operating_thresholds": decision["operating_thresholds"],
        "attack_scenario": scenario,

        "model_name":
            MODEL_NAME,

        "suspect_version":
            str(current_model.version),

        "rollback_version":
            (
                str(clean_version.version)
                if clean_version
                else None
            ),

        "detection_score":
            detection_score,

        "detection_details":
            detection_details,

        "detection_decision":
            "MALICIOUS",

        "mttd_seconds":
            mttd,
        "mttd_definition": "attack_start_to_detection_decision",
        "mttd_basis": mttd_basis,
        "attack_start_timestamp": attack_start_datetime.isoformat(),
        "detection_start_timestamp": detection_started_wall_datetime.isoformat(),
        "detection_timestamp": detection_decision_wall_datetime.isoformat(),
        "detection_compute_seconds": detection_compute_seconds,

        "mttr_seconds":
            mttr,
        "pipeline_overhead": pipeline_overhead,

        "response":
            response_results,

        "wazuh_api_status": wazuh_api_status,
        "wazuh_ingest": wazuh_ingest,
        "wazuh_alert_verification": wazuh_alert,

        "wazuh_event_file":
            WAZUH_EVENT_FILE,

        "total_execution_seconds":
            total_time,

        "response_claim_boundary": {
            "block": "MLflow promotion/deployment gate metadata + blocklist manifest; no serving process is forcibly terminated.",
            "quarantine": "MLflow security metadata marks the suspect version QUARANTINED.",
            "rollback": "Verified-clean registry version is tagged RESTORED_PRODUCTION and the production alias is moved then read-back verified.",
            "wazuh": "Event-file creation is distinct from verified alert ingestion; ingestion/alert status is recorded separately."
        },
        "research_note":
            "Phase 3 integrates Phase 2 detection with MLflow registry controls and Wazuh evidence."
    }

    evidence_path = save_evidence(
        evidence
    )
    summary = phase3_summary_main()

    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

    print_section(
        "PHASE 3 COMPLETE"
    )

    print(
        f"Detection decision : MALICIOUS"
    )

    print(
        f"Suspect model      : "
        f"{MODEL_NAME} v{current_model.version}"
    )

    print(
        f"MTTD               : "
        f"{mttd:.6f} seconds ({mttd_basis})"
    )

    print(
        f"MTTR               : "
        f"{mttr:.6f} seconds"
    )

    print(
        f"Block successful   : "
        f"{block_result['success']}"
    )

    print(
        f"Quarantine success : "
        f"{quarantine_result['success']}"
    )

    print(
        f"Rollback success   : "
        f"{rollback_result['success']}"
    )

    print(
        f"Wazuh reachable    : "
        f"{wazuh_api_status}"
    )

    print(
        "\nEvidence:"
    )

    print(evidence_path)
    print(f"Completed Phase 3 response runs: {summary['total_runs']}")




# ===== EMBEDDED registry_transition_hook.py =====
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def load_state(path: Path) -> dict:
    if not path.exists():
        return {"seen_versions": []}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {"seen_versions": []}
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid hook state file: {path}: {exc}") from exc


def save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def find_new_suspects(client, model_name: str, seen: set[str]):
    versions = list(client.search_model_versions(f"name='{model_name}'"))
    candidates = []
    for version in versions:
        tags = get_effective_model_tags(client, version)
        status = str(tags.get("status", "")).lower()
        deployment = str(tags.get("deployment_status", "")).upper()
        if status != "suspect" or deployment not in {"STAGING", "PRODUCTION"}:
            continue

        # Persist recovered provenance to the registry version so future scans
        # no longer depend solely on run-level tags.
        sync_run_tags_to_registry_version(client, version, tags)

        key = f"{model_name}:{version.version}:{tags.get('model_sha256', '')}"
        if key not in seen:
            candidates.append((version, key))
    return sorted(candidates, key=lambda item: int(item[0].version))


def invoke_phase3(project_root: Path, version_number: int) -> dict:
    # Invoke this exact final response script, rather than an older duplicate.
    script = Path(__file__).resolve()
    if not script.exists():
        raise FileNotFoundError(f"Phase 3 master not found: {script}")
    completed = subprocess.run(
        [sys.executable, str(script), "--phase3", "--suspect-version", str(version_number)],
        cwd=str(project_root),
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "returncode": completed.returncode,
        "stdout": completed.stdout[-5000:],
        "stderr": completed.stderr[-5000:],
    }


def registry_hook_main():
    parser = argparse.ArgumentParser(description="MLflow registry transition watcher for Secure MLflow Framework")
    parser.add_argument("--model-name", default="nbaiot_detector")
    parser.add_argument("--tracking-uri", default="http://127.0.0.1:5000")
    parser.add_argument("--state-file", default="evidence/phase3/registry_hook_state.json")
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    if args.poll_seconds <= 0:
        raise ValueError("--poll-seconds must be > 0")

    try:
        import mlflow
        from mlflow import MlflowClient
    except ImportError as exc:
        raise RuntimeError("MLflow is required for the registry hook. Install requirements_final.txt first.") from exc

    project_root = Path(__file__).resolve().parent
    state_file = project_root / args.state_file
    state = load_state(state_file)
    seen = set(str(v) for v in state.get("seen_versions", []))

    mlflow.set_tracking_uri(args.tracking_uri)
    client = MlflowClient(tracking_uri=args.tracking_uri)

    while True:
        for version, key in find_new_suspects(client, args.model_name, seen):
            print(f"[HOOK] New suspect registry version: {args.model_name} v{version.version}")
            result = invoke_phase3(project_root, int(version.version))
            print(json.dumps(result, indent=2))
            if result["returncode"] != 0:
                state["last_triggered_version"] = str(version.version)
                state["last_result"] = result
                save_state(state_file, state)
                raise RuntimeError(f"Phase 3 response failed for MLflow version {version.version}")
            seen.add(key)
            state["seen_versions"] = sorted(seen)
            state["last_triggered_version"] = str(version.version)
            state["last_result"] = result
            save_state(state_file, state)

        if args.once:
            break
        time.sleep(args.poll_seconds)




# ============================================================
# PHASE B ORCHESTRATION
# ============================================================

def main_response():
    parser = argparse.ArgumentParser(
        description="Two-phase Secure MLflow automated response pipeline — Phase B"
    )
    parser.add_argument("--phase3", action="store_true", help="Run Phase 3 response once")
    parser.add_argument("--suspect-version", type=int, default=None, help="Exact MLflow suspect version")
    parser.add_argument("--watch-registry", action="store_true", help="Watch MLflow registry transitions")
    parser.add_argument("--attack-start-at", default=None, help="Explicit UTC attack-start timestamp (ISO-8601) for true MTTD")
    parser.add_argument("--baseline-pipeline-seconds", type=float, default=None, help="Baseline pipeline wall time for overhead calculation")
    parser.add_argument("--summarize-phase3", action="store_true", help="Build MTTD/MTTR summary from saved Phase 3 response runs")
    parser.add_argument("--phase3-clean-control", action="store_true", help="Record a non-destructive verified-clean response control")
    parser.add_argument("--clean-version", type=int, default=None, help="Exact verified-clean version for --phase3-clean-control")
    parser.add_argument("--phase3-final-results", action="store_true", help="Create final Phase 3 tables from response and clean-control evidence")
    parser.add_argument("--once", action="store_true", help="Process current registry state once")
    args = parser.parse_args()

    if args.summarize_phase3:
        phase3_summary_main()
        return

    if args.phase3_clean_control:
        phase3_clean_control_main(args.clean_version)
        return

    if args.phase3_final_results:
        phase3_final_results_main()
        return

    if args.watch_registry or args.once:
        # --once is a convenient one-shot registry scan even without --watch-registry.
        original_argv = sys.argv[:]
        try:
            sys.argv = [sys.argv[0], "--once"] if args.once else [sys.argv[0]]
            registry_hook_main()
        finally:
            sys.argv = original_argv
        return

    if args.phase3:
        phase3_master_main(
            args.suspect_version,
            attack_start_at=args.attack_start_at,
            baseline_pipeline_seconds=args.baseline_pipeline_seconds,
        )
        return

    parser.print_help()

if __name__ == "__main__":
    main_response()
