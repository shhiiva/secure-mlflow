#!/usr/bin/env python3
"""Register an exact existing S1 or S2 Phase-1 model as a fresh Phase-3 suspect.

No training and no attack generation occurs here.  Phase 3 subsequently derives
the score again from the canonical Phase-1 files named by rate and repetition.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Register an exact S1/S2 model for Phase 3")
    parser.add_argument("--scenario", choices=["S1_LABEL_FLIP", "S2_BACKDOOR"], required=True)
    parser.add_argument("--poison-rate", type=float, choices=[0.01, 0.05, 0.1], required=True)
    parser.add_argument("--repetition", type=int, choices=range(1, 6), required=True)
    args = parser.parse_args()
    token = f"{int(round(args.poison_rate * 100)):02d}"
    if args.scenario == "S1_LABEL_FLIP":
        artifact = ROOT / "models" / "S1_label_flip" / f"label_flip_model_rate{token}_run{args.repetition:02d}.joblib"
    else:
        artifact = ROOT / "models" / "S2_backdoor" / f"backdoor_model_rate{token}_run{args.repetition:02d}.joblib"
    if not artifact.is_file():
        raise FileNotFoundError(f"Exact Phase-1 model not found: {artifact}")

    artifact_hash = sha256_file(artifact)
    attack_started_at = datetime.now(timezone.utc).isoformat()
    mlflow.set_tracking_uri(MLFLOW_URI)
    client = MlflowClient(tracking_uri=MLFLOW_URI)
    client.get_registered_model(MODEL_NAME)
    with mlflow.start_run(run_name=f"PHASE3_REGISTRATION_{args.scenario}_RATE{token}_RUN{args.repetition:02d}") as run:
        mlflow.set_tags({
            "attack_scenario": args.scenario,
            "status": "suspect",
            "poison_rate": str(args.poison_rate),
            "repetition": str(args.repetition),
            "attack_started_at": attack_started_at,
        })
        mlflow.log_artifact(str(artifact), artifact_path="model_artifact")
        source = f"runs:/{run.info.run_id}/model_artifact/{artifact.name}"
        version = client.create_model_version(MODEL_NAME, source=source, run_id=run.info.run_id)
    for key, value in {
        "research_phase": "PHASE_3_CONTROLLED_MODEL_INCIDENT",
        "attack_scenario": args.scenario,
        "status": "suspect",
        "deployment_status": "STAGING",
        "security_status": "SUSPECT",
        "poison_rate": str(args.poison_rate),
        "repetition": str(args.repetition),
        "attack_started_at": attack_started_at,
        "model_sha256": artifact_hash,
        "model_artifact_path": str(artifact),
    }.items():
        client.set_model_version_tag(MODEL_NAME, str(version.version), key, value)
    print(f"Registered controlled suspect: {MODEL_NAME} v{version.version}")
    print(f"Run: python .\\3phase_response_FINAL.py --phase3 --suspect-version {version.version}")


if __name__ == "__main__":
    main()
