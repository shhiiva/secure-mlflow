"""Regenerate Phase 3 timing statistics from the pre-specified primary run set."""

from pathlib import Path

from finalize_research_evidence import Audit, phase3_analysis, write_csv


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT = PROJECT_ROOT / "results" / "phase3" / "phase3_confidence_intervals.csv"
DETAIL_OUTPUT = PROJECT_ROOT / "results" / "phase3" / "phase3_primary_timing_statistics.csv"


def main() -> None:
    audit = Audit()
    _, timing_rows, summary = phase3_analysis(
        PROJECT_ROOT,
        audit,
        resamples=20_000,
        seed=20260910,
    )
    write_csv(OUTPUT, timing_rows)
    write_csv(DETAIL_OUTPUT, timing_rows)
    print(f"Saved: {OUTPUT}")
    print(f"Saved: {DETAIL_OUTPUT}")
    print(
        "Primary timing runs: "
        f"{summary['primary_timing_runs']}; supplementary alert-validation runs: "
        f"{summary['supplementary_alert_validation_runs']}"
    )


if __name__ == "__main__":
    main()
