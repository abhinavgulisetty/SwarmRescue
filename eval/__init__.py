"""Fair, paired evaluation of EMBER and classical baselines."""

from eval.harness import (
    DEVELOPMENT_SEEDS,
    HELD_OUT_SEEDS,
    EvaluationConfig,
    EvaluationReport,
    MethodRun,
    SeedRun,
    evaluate,
    evaluate_seed,
    format_summary,
    parse_seed_spec,
    write_report,
    write_runs_csv,
)

__all__ = [
    "DEVELOPMENT_SEEDS",
    "HELD_OUT_SEEDS",
    "EvaluationConfig",
    "EvaluationReport",
    "MethodRun",
    "SeedRun",
    "evaluate",
    "evaluate_seed",
    "format_summary",
    "parse_seed_spec",
    "write_report",
    "write_runs_csv",
]
