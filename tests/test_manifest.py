"""Tests for reproducible experiment manifests."""

from __future__ import annotations

import json
import subprocess
import tarfile

from ember.manifest import (
    capture_manifest,
    write_manifest,
    write_source_bundle,
)


def test_config_hash_is_stable_across_key_order(tmp_path) -> None:
    first = capture_manifest(7, {"beta": 1, "alpha": 2}, {}, root=tmp_path)
    second = capture_manifest(7, {"alpha": 2, "beta": 1}, {}, root=tmp_path)

    assert first.config_hash == second.config_hash
    assert first.versions["python"]


def test_manifest_round_trip_contains_provenance_and_metrics(tmp_path) -> None:
    (tmp_path / "uv.lock").write_text("locked", encoding="utf-8")
    manifest = capture_manifest(
        11,
        {"robots": 4},
        {"accepted": True, "radio_bytes": 244},
        root=tmp_path,
    )
    output = tmp_path / "run" / "manifest.json"

    write_manifest(manifest, output)
    decoded = json.loads(output.read_text(encoding="utf-8"))

    assert decoded["schema_version"] == 2
    assert decoded["seed"] == 11
    assert len(decoded["config_hash"]) == 64
    assert len(decoded["lockfile_hash"]) == 64
    assert decoded["dirty_diff_hash"] is None
    assert decoded["metrics"]["accepted"] is True


def test_source_bundle_contains_binary_diff_and_untracked_files(tmp_path) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()

    def git(*arguments: str) -> None:
        subprocess.run(
            ("git", *arguments),
            cwd=repository,
            check=True,
            capture_output=True,
        )

    git("init")
    git("config", "user.email", "evaluation@example.invalid")
    git("config", "user.name", "Evaluation Test")
    tracked = repository / "tracked.py"
    tracked.write_text("value = 1\n", encoding="utf-8")
    git("add", "tracked.py")
    git("commit", "-m", "base")

    tracked.write_text("value = 2\n", encoding="utf-8")
    untracked = repository / "src" / "new.py"
    untracked.parent.mkdir()
    untracked.write_text("new = True\n", encoding="utf-8")
    bundle = tmp_path / "source.tar.gz"

    bundle_hash = write_source_bundle(bundle, root=repository)

    assert len(bundle_hash) == 64
    with tarfile.open(bundle, mode="r:gz") as archive:
        assert {"bundle.json", "tracked.diff", "untracked/src/new.py"} <= set(
            archive.getnames()
        )
        diff_stream = archive.extractfile("tracked.diff")
        metadata_stream = archive.extractfile("bundle.json")
        assert diff_stream is not None
        assert metadata_stream is not None
        assert b"value = 2" in diff_stream.read()
        metadata = json.load(metadata_stream)

    assert len(metadata["dirty_diff_hash"]) == 64
    assert metadata["untracked_files"][0]["path"] == "src/new.py"
