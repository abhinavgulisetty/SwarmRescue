"""Installed source identity and cache behavior without checkout assumptions."""

import hashlib
import json
import tarfile
from types import SimpleNamespace

import pytest

from ember import manifest


def installed_package(tmp_path, monkeypatch):
    root = tmp_path / "site-packages"
    content = {
        "ember/manifest.py": b"# shipped source\n",
        "firmware/log_lmdp.c": b"/* integer kernel */\n",
        "swarmrescue/cli.py": b"# entrypoint\n",
        "swarmrescue/runtime-lock.txt": b"numpy==2.5.1\n",
        "swarmrescue-0.2.0.dist-info/WHEEL": b"Wheel-Version: 1.0\n",
    }
    for name, payload in content.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    package = SimpleNamespace(
        version="0.2.0", files=list(content), locate_file=lambda name: root / name
    )
    monkeypatch.setattr(manifest, "distribution", lambda name: package)
    monkeypatch.setattr(manifest, "__file__", str(root / "ember/manifest.py"))
    return root, package


def test_source_build_inventory_does_not_change_checkout_source_identity(
    tmp_path, monkeypatch
):
    root, package = installed_package(tmp_path, monkeypatch)
    package.files.remove("swarmrescue-0.2.0.dist-info/WHEEL")
    package.files.append("swarmrescue.egg-info/PKG-INFO")
    assert manifest.installed_source_files() is None
    monkeypatch.setattr(manifest, "_git_state", lambda root: ("a" * 40, False, None))
    path = root / "swarmrescue/templates/cases.html"
    path.parent.mkdir(parents=True)
    path.write_text("newly inventoried source template")
    before = manifest.capture_manifest(7, {}, {}, root=root)
    package.files.append("swarmrescue/templates/cases.html")
    after = manifest.capture_manifest(7, {}, {}, root=root)
    assert before.source_kind == after.source_kind == "git"
    assert before.source_identity == after.source_identity


def test_installed_manifest_tracks_shipped_sources_instead_of_cwd_git(
    tmp_path, monkeypatch
):
    root, _ = installed_package(tmp_path, monkeypatch)
    monkeypatch.chdir(tmp_path)
    before = manifest.capture_manifest(7, {}, {})
    assert before.source_kind == "installed_distribution" and before.schema_version == 3
    assert before.git_sha == "unknown" and not before.git_dirty
    assert before.package_version == "0.2.0"
    assert before.lockfile_hash == hashlib.sha256(b"numpy==2.5.1\n").hexdigest()
    (root / "firmware/log_lmdp.c").write_text("/* changed */\n")
    after = manifest.capture_manifest(7, {}, {})
    assert before.source_fingerprint != after.source_fingerprint


def test_runtime_archive_contains_the_sources_that_were_fingerprinted(
    tmp_path, monkeypatch
):
    _, _ = installed_package(tmp_path, monkeypatch)
    before = manifest.capture_manifest(7, {}, {})
    output = tmp_path / "run/source.tar.gz"
    assert len(manifest.write_runtime_source_bundle(output)) == 64
    with tarfile.open(output) as archive:
        metadata = json.load(archive.extractfile("bundle.json"))
        assert metadata["source_fingerprint"] == before.source_fingerprint
        for item in metadata["source_files"]:
            payload = archive.extractfile("installed/" + item["path"]).read()
            assert hashlib.sha256(payload).hexdigest() == item["sha256"]


def test_installed_source_may_not_escape_distribution_root(tmp_path, monkeypatch):
    root, _ = installed_package(tmp_path, monkeypatch)
    outside = tmp_path / "outside.py"
    outside.write_text("external")
    (root / "ember/manifest.py").unlink()
    (root / "ember/manifest.py").symlink_to(outside)
    monkeypatch.setattr(manifest, "__file__", str(outside))
    with pytest.raises(ValueError, match="escapes package root"):
        manifest.installed_source_files()


def test_cache_override_and_installed_xdg_directory(tmp_path, monkeypatch):
    from cascade import collapse

    monkeypatch.setattr(
        collapse, "__file__", str(tmp_path / "site/cascade/collapse.py")
    )
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "writable"))
    monkeypatch.delenv("SWARMRESCUE_CACHE_DIR", raising=False)
    assert collapse.default_cache_dir() == tmp_path / "writable/swarmrescue/collapse"
    monkeypatch.setenv("SWARMRESCUE_CACHE_DIR", str(tmp_path / "explicit"))
    assert collapse.default_cache_dir() == tmp_path / "explicit"


def test_checkout_cache_remains_compatible(tmp_path, monkeypatch):
    from cascade import collapse

    (tmp_path / "pyproject.toml").write_text("[project]\n")
    monkeypatch.setattr(collapse, "__file__", str(tmp_path / "cascade/collapse.py"))
    monkeypatch.delenv("SWARMRESCUE_CACHE_DIR", raising=False)
    assert collapse.default_cache_dir() == tmp_path / "cascade/cache"


def test_source_identity_includes_installed_content_and_dependency_versions(tmp_path):
    from dataclasses import replace

    value = manifest.capture_manifest(7, {}, {}, root=tmp_path)
    assert (
        value.source_identity
        == replace(value, created_at_utc="another time").source_identity
    )
    assert (
        value.source_identity
        != replace(value, source_fingerprint="changed").source_identity
    )
    assert (
        value.source_identity
        != replace(value, versions={"numpy": "different"}).source_identity
    )


def test_installed_presentation_template_is_fingerprinted_and_archived(
    tmp_path, monkeypatch
):
    root, package = installed_package(tmp_path, monkeypatch)
    path = root / "swarmrescue/templates/cases.html"
    path.parent.mkdir(parents=True)
    path.write_text("<html>case player</html>")
    package.files.append("swarmrescue/templates/cases.html")
    before = manifest.capture_manifest(7, {}, {})
    path.write_text("<html>changed player</html>")
    after = manifest.capture_manifest(7, {}, {})
    assert before.source_fingerprint != after.source_fingerprint
    output = tmp_path / "source.tar.gz"
    manifest.write_runtime_source_bundle(output)
    with tarfile.open(output) as archive:
        assert (
            archive.extractfile("installed/swarmrescue/templates/cases.html").read()
            == path.read_bytes()
        )
