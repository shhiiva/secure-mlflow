# Secure MLflow Framework — 3 Implementation Phases

This implementation follows the approved research proposal:

1. Code Phase 1 — MLflow + N-BaIoT + neural network + five controlled attack scenarios.
2. Code Phase 2 — five-component detection framework; Neural Cleanse + STRIP + Activation Clustering are fused for the neural/backdoor detector.
3. Code Phase 3 — MLflow registry response, block, quarantine, rollback, Wazuh alerting, 5x5 experiments, metrics and threshold analysis.

## Research mapping

- RQ1: attack propagation and observable artefacts.
- RQ2: effectiveness of SBOM hash verification, recursive Pickle analysis, Neural Cleanse, STRIP and Activation Clustering.
- RQ3: automated block, quarantine, rollback and SIEM response; MTTD/MTTR.
- RQ4: accuracy/FPR trade-off and threshold optimisation.

## Important experimental rule

Do not report placeholder/smoke-test numbers as dissertation results. Run the N-BaIoT experiments with the final configuration and preserve raw JSON/CSV evidence.

## Data

Put the approved N-BaIoT CSV in `data/raw/nbaiot.csv` and set `dataset.target_column` in `configs/config.yaml`.

A synthetic smoke-test mode exists only to verify that the pipeline wiring works before the real dataset is loaded. It is not a substitute for the N-BaIoT experiment.

## Security scope

All attack scenarios are controlled local simulations. Do not attack real PyPI, Hugging Face, MLflow or production infrastructure.
