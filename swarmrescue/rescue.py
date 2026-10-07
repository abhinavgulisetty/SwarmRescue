"""Run the complete search, local contact and physical evacuation workflow."""

from cascade.demo import main as mission_main

DEFAULT_ARGUMENTS = (
    "--seed",
    "42",
    "--robots",
    "8",
    "--compact",
    "--live-disaster",
    "--distributed",
    "--region-cells",
    "4",
    "--latest-evidence",
    "--range-gbp",
    "--range-radio",
    "--evacuate",
    "--no-render",
)


def main(argv=None):
    import sys

    return mission_main([*DEFAULT_ARGUMENTS, *(sys.argv[1:] if argv is None else argv)])


if __name__ == "__main__":
    raise SystemExit(main())
