"""Installer protections and execution of a launcher with shell-sensitive paths."""

import json
import shutil
import subprocess
import sys
import zipfile

import pytest

from swarmrescue.deploy import install, write_launcher


def test_missing_wheel_does_not_create_target(tmp_path):
    target = tmp_path / "deployment"
    with pytest.raises(ValueError, match="wheel is missing"):
        install(tmp_path / "missing.whl", target)
    assert not target.exists()


def test_installer_cannot_replace_its_running_environment(tmp_path, monkeypatch):
    wheel = tmp_path / "release.whl"
    wheel.write_bytes(b"placeholder")
    prefix = tmp_path / "deployment"
    monkeypatch.setattr(sys, "prefix", str(prefix / "venv"))
    with pytest.raises(ValueError, match="outside the target"):
        install(wheel, prefix)
    assert not prefix.exists()


def test_wheel_without_runtime_lock_does_not_create_target(tmp_path, monkeypatch):
    wheel = tmp_path / "release.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("unrelated.txt", "incomplete release")
    monkeypatch.setattr(shutil, "which", lambda name: "/unused/uv")
    prefix = tmp_path / "deployment"
    with pytest.raises(KeyError, match="runtime-lock"):
        install(wheel, prefix)
    assert not prefix.exists()


@pytest.mark.parametrize("arm", [False, True])
def test_launcher_preserves_paths_arguments_cache_and_working_directory(tmp_path, arm):
    prefix = tmp_path / "deploy with spaces ' and $(touch injected)"
    prefix.mkdir()
    cli = prefix / "fake cli"
    cli.write_text(
        "#!" + sys.executable + "\n"
        "import json, os, sys\n"
        "print(json.dumps({'args': sys.argv[1:], 'cwd': os.getcwd(), "
        "'cache': os.environ['SWARMRESCUE_CACHE_DIR']}))\n"
    )
    cli.chmod(0o755)
    image = prefix / "firmware image.elf"
    tools = prefix / "tool root"
    launcher = write_launcher(
        prefix,
        cli,
        firmware={"executable": str(image)} if arm else None,
        tool_root=tools,
    )
    result = subprocess.run(
        [str(launcher), "--seed", "7", "--report", "a 'quoted' report.json"],
        cwd=tmp_path,
        check=True,
        text=True,
        capture_output=True,
    )
    decoded = json.loads(result.stdout)
    assert decoded["cwd"] == str(prefix)
    assert decoded["cache"] == str(prefix / "cache")
    assert decoded["args"][-4:] == ["--seed", "7", "--report", "a 'quoted' report.json"]
    if arm:
        index = decoded["args"].index("--firmware-image")
        assert decoded["args"][index + 1] == str(image)
    else:
        assert "--firmware-robots" not in decoded["args"]
    assert not (tmp_path / "injected").exists()
