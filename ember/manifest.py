"""Reproducible, machine-readable manifests for experiment runs."""

from __future__ import annotations

import hashlib
import io
import json
import os
import platform
import subprocess
import tarfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class RunManifest:
    schema_version: int
    created_at_utc: str
    seed: int
    git_sha: str
    git_dirty: bool
    dirty_diff_hash: str | None
    lockfile_hash: str | None
    config_hash: str
    config: dict[str, Any]
    versions: dict[str, str]
    metrics: dict[str, int | float | str | bool]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True) + "\n"


def _canonical_hash(value: dict[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dirty_hash(root: Path) -> str | None:
    try:
        diff = subprocess.run(
            ("git", "diff", "--binary", "HEAD"),
            cwd=root,
            check=True,
            capture_output=True,
        ).stdout
        untracked = subprocess.run(
            ("git", "ls-files", "--others", "--exclude-standard", "-z"),
            cwd=root,
            check=True,
            capture_output=True,
        ).stdout.split(b"\0")
    except (OSError, subprocess.CalledProcessError):
        return None
    digest = hashlib.sha256(diff)
    for encoded in sorted(path for path in untracked if path):
        digest.update(b"\0untracked\0")
        digest.update(encoded)
        digest.update(b"\0")
        digest.update((root / encoded.decode()).read_bytes())
    return digest.hexdigest()


def _git_state(root: Path) -> tuple[str, bool, str | None]:
    try:
        sha = subprocess.run(
            ("git", "rev-parse", "HEAD"),
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ("git", "status", "--porcelain"),
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return "unknown", False, None
    dirty = bool(status)
    return sha, dirty, _dirty_hash(root) if dirty else None


def _versions() -> dict[str, str]:
    packages = ("numpy", "mujoco", "pillow", "imageio")
    found = {"python": platform.python_version()}
    for package in packages:
        try:
            found[package] = version(package)
        except PackageNotFoundError:
            found[package] = "not-installed"
    return found


def capture_manifest(
    seed: int,
    config: dict[str, Any],
    metrics: dict[str, int | float | str | bool],
    *,
    root: Path | None = None,
) -> RunManifest:
    """Capture seed, source state, normalized config and runtime versions."""

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    normalized = json.loads(json.dumps(config, sort_keys=True))
    source_root = root or Path.cwd()
    git_sha, git_dirty, dirty_diff_hash = _git_state(source_root)
    return RunManifest(
        schema_version=2,
        created_at_utc=datetime.now(UTC).isoformat(),
        seed=seed,
        git_sha=git_sha,
        git_dirty=git_dirty,
        dirty_diff_hash=dirty_diff_hash,
        lockfile_hash=_sha256_file(source_root / "uv.lock"),
        config_hash=_canonical_hash(normalized),
        config=normalized,
        versions=_versions(),
        metrics=dict(metrics),
    )


def write_source_bundle(path: Path, *, root: Path | None = None) -> str:
    """Archive the binary Git diff and untracked files needed to rebuild a run."""

    requested_root = (root or Path.cwd()).resolve()
    try:
        repository_root = Path(
            subprocess.run(
                ("git", "rev-parse", "--show-toplevel"),
                cwd=requested_root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        ).resolve()
        git_sha = subprocess.run(
            ("git", "rev-parse", "HEAD"),
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        tracked_diff = subprocess.run(
            ("git", "diff", "--binary", "HEAD"),
            cwd=repository_root,
            check=True,
            capture_output=True,
        ).stdout
        untracked = subprocess.run(
            ("git", "ls-files", "--others", "--exclude-standard", "-z"),
            cwd=repository_root,
            check=True,
            capture_output=True,
        ).stdout.split(b"\0")
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError(
            "source bundle root must be inside a Git repository"
        ) from error

    destination = path.resolve()
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    payloads: list[tuple[str, bytes]] = []
    untracked_metadata: list[dict[str, str]] = []
    for encoded in sorted(item for item in untracked if item):
        relative = Path(os.fsdecode(encoded))
        source = (repository_root / relative).resolve()
        if not source.is_relative_to(repository_root):
            raise ValueError("untracked source path escapes the repository")
        if source in (destination, temporary.resolve()) or not source.is_file():
            continue
        payload = source.read_bytes()
        archive_name = f"untracked/{relative.as_posix()}"
        payloads.append((archive_name, payload))
        untracked_metadata.append(
            {"path": relative.as_posix(), "sha256": hashlib.sha256(payload).hexdigest()}
        )

    metadata = {
        "schema_version": 1,
        "git_sha": git_sha,
        "dirty_diff_hash": _dirty_hash(repository_root),
        "lockfile_hash": _sha256_file(repository_root / "uv.lock"),
        "tracked_diff_file": "tracked.diff",
        "untracked_files": untracked_metadata,
    }
    metadata_bytes = (json.dumps(metadata, indent=2, sort_keys=True) + "\n").encode()

    def add_bytes(archive: tarfile.TarFile, name: str, payload: bytes) -> None:
        info = tarfile.TarInfo(name)
        info.size = len(payload)
        info.mode = 0o644
        info.mtime = 0
        archive.addfile(info, io.BytesIO(payload))

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(temporary, mode="w:gz") as archive:
        add_bytes(archive, "bundle.json", metadata_bytes)
        add_bytes(archive, "tracked.diff", tracked_diff)
        for name, payload in payloads:
            add_bytes(archive, name, payload)
    temporary.replace(destination)
    bundle_hash = _sha256_file(destination)
    if bundle_hash is None:
        raise RuntimeError("source bundle was not written")
    return bundle_hash


def write_manifest(manifest: RunManifest, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(manifest.to_json(), encoding="utf-8")
    temporary.replace(path)


__all__ = [
    "RunManifest",
    "capture_manifest",
    "write_manifest",
    "write_source_bundle",
]
