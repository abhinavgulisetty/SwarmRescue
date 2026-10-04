"""Command-line entry point for paired EMBER evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

from ember.manifest import write_source_bundle
from eval.harness import (
    EvaluationConfig,
    evaluate,
    format_summary,
    parse_seed_spec,
    write_report,
    write_runs_csv,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m eval", description=__doc__)
    parser.add_argument(
        "--seeds",
        default="0:199",
        help="comma-separated seeds and inclusive ranges (default: 0:199)",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=2_000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--bootstrap-seed", type=int, default=0xE4B3)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/eval/development.json"),
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("outputs/eval/development.csv"),
    )
    parser.add_argument(
        "--source-bundle",
        type=Path,
        help="archive the binary Git diff and untracked files for reconstruction",
    )
    parser.add_argument(
        "--allow-held-out",
        action="store_true",
        help="unlock sealed seeds 10000..10049 after thresholds are frozen",
    )
    args = parser.parse_args(argv)
    try:
        seeds = parse_seed_spec(args.seeds)
        config = EvaluationConfig(
            seeds=seeds,
            bootstrap_samples=args.bootstrap_samples,
            confidence_level=args.confidence,
            bootstrap_seed=args.bootstrap_seed,
            allow_held_out=args.allow_held_out,
        )
    except ValueError as error:
        parser.error(str(error))

    report = evaluate(config)
    bundle_hash = None
    if args.source_bundle is not None:
        bundle_hash = write_source_bundle(args.source_bundle)
    write_report(report, args.out)
    write_runs_csv(report, args.csv)
    print(format_summary(report))
    print(f"report: {args.out}")
    print(f"runs:   {args.csv}")
    if args.source_bundle is not None:
        print(f"source: {args.source_bundle} (sha256 {bundle_hash})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
