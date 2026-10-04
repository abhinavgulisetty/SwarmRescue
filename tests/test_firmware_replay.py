"""Freestanding harness acceptance and explicit rejection of wrong reference data."""

import os
import re
import shutil

import pytest

from eval.firmware_replay import generate_header, main, run_replay


def test_fixture_header_deterministic_and_covers_both_kernel_families():
    header = generate_header((0, 1))
    assert header == generate_header((0, 1))
    assert "#define REPLAY_LOG_CASES 4" in header
    assert "#define REPLAY_DEADLINE_CASES 6" in header
    assert "#define REPLAY_MAX_N 500" in header
    assert "l0_expected" in header and "l1_actions" in header


def test_host_harness_accepts_independent_python_reference(tmp_path):
    if not shutil.which("cc"):
        pytest.skip("C compiler unavailable")
    result = run_replay(tmp_path, (0, 1))
    assert result["failures"] == 0 and result["comparisons"] > 1000
    assert result["log_cases"] == 4 and result["deadline_cases"] == 6
    assert len(result["executable_sha256"]) == 64


def test_harness_rejects_corrupted_expected_field(monkeypatch, tmp_path):
    if not shutil.which("cc"):
        pytest.skip("C compiler unavailable")
    import eval.firmware_replay as replay

    original = replay.generate_header

    def corrupt(seeds):
        return re.sub(r"(l0_expected\[\] = \{)\d+", r"\g<1>0", original(seeds), count=1)

    monkeypatch.setattr(replay, "generate_header", corrupt)
    with pytest.raises(RuntimeError, match="kernel replay failed"):
        replay.run_replay(tmp_path, (0,))


@pytest.mark.parametrize("seeds", ["10000", "-1"])
def test_cli_rejects_non_development_seeds(seeds):
    with pytest.raises(SystemExit) as result:
        main(["--seeds", seeds])
    assert result.value.code == 2


def test_cortex_m4_replay_when_toolchain_configured(tmp_path):
    root = os.environ.get("EMBER_ARM_TOOL_ROOT")
    if not root and not (
        shutil.which("arm-none-eabi-gcc") and shutil.which("qemu-system-arm")
    ):
        pytest.skip("Cortex-M4 compiler/QEMU unavailable; set EMBER_ARM_TOOL_ROOT")
    result = run_replay(tmp_path, (0, 1), tool_root=root, arm=True)
    assert result["failures"] == 0 and result["comparisons"] > 1000
    assert (
        0 < result["observed_stack_high_water_bytes"] < result["stack_reserved_bytes"]
    )
    assert result["kernel_harness_ram_with_reserved_stack_bytes"] < 200 * 1024
    assert result["text_bytes"] > 16388
