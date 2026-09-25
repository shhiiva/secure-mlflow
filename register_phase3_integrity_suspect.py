#!/usr/bin/env python3
"""Register a controlled S3/S4/S5 Phase-3 integrity incident.

The artifact is logged for provenance only; it is never imported or
deserialised.  Gate evidence must be produced by Phase 2 first.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import sys
import mlflow
from mlflow import MlflowClient

# MLflow prints Unicode run-status symbols on Windows.  Force UTF-8 so
# registration completes instead of failing while closing the run.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
MODEL_NAME = "nbaiot_detector"
MLFLOW_URI = "http://127.0.0.1:5000"
CONTROL_BY_SCENARIO = {
    "S3_PYPI": "SBOM",
    "S4_MODEL_HUB": "PICKLE",
    "S5_MLFLOW_TAMPERING": "MLFLOW_HASH",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Register a hash-verified Phase-3 integrity suspect")
    parser.add_argument("--scenario", choices=sorted(CONTROL_BY_SCENARIO), required=True)
    parser.add_argument("--artifact", required=True)
    parser.add_argument(
        "--gate-evidence",
        default=None,
        help="Existing Phase-2 gate evidence for S3/S4. S5 generates hash-gate evidence automatically.",
    )
    parser.add_argument(
        "--expected-artifact",
        default=None,
        help="Verified-clean reference artifact required for S5. Its SHA-256 is the expected hash.",
    )
    parser.add_argument("--control", default="modified")
    args = parser.parse_args()

    artifact = Path(args.artifact).resolve()
    if not artifact.is_file():
        raise FileNotFoundError(f"Artifact not found: {artifact}")
    required_gate = CONTROL_BY_SCENARIO[args.scenario]

    expected_artifact = None
    expected_artifact_hash = None
    observed_artifact_hash = sha256_file(artifact)

    if args.scenario == "S5_MLFLOW_TAMPERING":
        if not args.expected_artifact:
            raise RuntimeError("S5 requires --expected-artifact pointing to the verified-clean reference file.")
        expected_artifact = Path(args.expected_artifact).resolve()
        if not expected_artifact.is_file():
            raise FileNotFoundError(f"Expected clean artifact not found: {expected_artifact}")
        expected_artifact_hash = sha256_file(expected_artifact)
        malicious = expected_artifact_hash != observed_artifact_hash
        decision = "HASH_MISMATCH" if malicious else "MATCH"
        if not malicious:
            raise RuntimeError(
                "S5 registration refused: suspect and verified-clean artifacts have identical SHA-256 hashes."
            )

        evidence_dir = ROOT / "results" / "phase3" / "s5_hash_gates"
        evidence_dir.mkdir(parents=True, exist_ok=True)
        evidence_path = evidence_dir / (
            "s5_hash_gate_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".json"
        )
        evidence = {
            "phase": "PHASE_3_S5_HASH_GATE",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "scenario": args.scenario,
            "gate": required_gate,
            "control": args.control,
            "verified_clean_artifact_path": str(expected_artifact),
            "suspect_artifact_path": str(artifact),
            "expected_sha256": expected_artifact_hash,
            "observed_sha256": observed_artifact_hash,
            "hash_match": False,
            "decision": decision,
            "score": 1.0,
            "passed": True,
        }
        evidence_path.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    else:
        if not args.gate_evidence:
            raise RuntimeError(f"{args.scenario} requires --gate-evidence.")
        evidence_path = Path(args.gate_evidence).resolve()
        if not evidence_path.is_file():
            raise FileNotFoundError(f"Gate evidence not found: {evidence_path}")
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        matches = [
            item for item in evidence.get("controls", [])
            if str(item.get("gate", "")).upper() == required_gate
            and str(item.get("control", "")).lower() == args.control.lower()
            and bool(item.get("passed", False))
        ]
        if len(matches) != 1:
            raise RuntimeError(f"Expected one passed {required_gate}/{args.control} control in gate evidence.")

        gate_result = matches[0].get("result", {})
        if required_gate == "PICKLE":
            malicious = not bool(gate_result.get("safe", True))
            decision = "UNSAFE_PICKLE_OPCODE"
        else:
            decision = str(gate_result.get("status", "")).upper()
            malicious = decision not in {"", "MATCH", "CLEAN", "SAFE", "NORMAL"}
        if not malicious:
            raise RuntimeError("Selected control is not a malicious gate result; registration refused.")

    artifact_hash = observed_artifact_hash
    evidence_hash = sha256_file(evidence_path)
    attack_started_at = datetime.now(timezone.utc).isoformat()
    mlflow.set_tracking_uri(MLFLOW_URI)
    client = MlflowClient(tracking_uri=MLFLOW_URI)
    client.get_registered_model(MODEL_NAME)

    with mlflow.start_run(run_name=f"PHASE3_{args.scenario}_INTEGRITY_CONTROL") as run:
        mlflow.set_tags({"attack_scenario": args.scenario, "status": "suspect", "attack_started_at": attack_started_at})
        mlflow.log_artifact(str(artifact), artifact_path="controlled_integrity_artifact")
        source = f"runs:/{run.info.run_id}/controlled_integrity_artifact/{artifact.name}"
        version = client.create_model_version(MODEL_NAME, source=source, run_id=run.info.run_id)

    # Read the copy stored by MLflow and prove that the registered version is
    # bound to the observed suspect hash rather than only to a local file.
    downloaded_artifact = Path(
        client.download_artifacts(
            run.info.run_id,
            f"controlled_integrity_artifact/{artifact.name}",
        )
    )
    mlflow_artifact_hash = sha256_file(downloaded_artifact)
    if mlflow_artifact_hash != observed_artifact_hash:
        raise RuntimeError("The MLflow-stored artifact hash differs from the registered suspect artifact hash.")

    tags = {
        "research_phase": "PHASE_3_CONTROLLED_INTEGRITY_INCIDENT",
        "attack_scenario": args.scenario,
        "status": "suspect",
        "deployment_status": "STAGING",
        "security_status": "SUSPECT",
        "attack_started_at": attack_started_at,
        "integrity_gate_control": required_gate,
        "integrity_gate_decision": decision,
        "integrity_gate_score": "1.0",
        "integrity_gate_evidence_path": str(evidence_path),
        "integrity_gate_evidence_sha256": evidence_hash,
        "controlled_artifact_path": str(artifact),
        "controlled_artifact_sha256": artifact_hash,
        "mlflow_artifact_sha256": mlflow_artifact_hash,
    }
    if args.scenario == "S5_MLFLOW_TAMPERING":
        tags.update({
            "verified_clean_artifact_path": str(expected_artifact),
            "expected_artifact_sha256": expected_artifact_hash,
            "observed_artifact_sha256": observed_artifact_hash,
            "artifact_hash_match": "false",
        })
    for key, value in tags.items():
        client.set_model_version_tag(MODEL_NAME, str(version.version), key, value)
    print(f"Registered controlled integrity suspect: {MODEL_NAME} v{version.version}")
    print(f"Run: python .\\3phase_response_FINAL.py --phase3 --suspect-version {version.version}")


if __name__ == "__main__":
    main()
