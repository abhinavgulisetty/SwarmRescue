"""Backend tests run in fresh interpreters because MuJoCo binds GL on import."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]


def _python(code: str, *, backend: str | None) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    for variable in ("DISPLAY", "WAYLAND_DISPLAY"):
        env.pop(variable, None)
    if backend is None:
        env.pop("MUJOCO_GL", None)
    else:
        env["MUJOCO_GL"] = backend
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="EGL is Linux-only")
def test_package_selects_egl_before_mujoco_and_renders_without_a_display() -> None:
    result = _python(
        """
import os
import cascade
assert os.environ["MUJOCO_GL"] == "egl"
import mujoco

model = mujoco.MjModel.from_xml_string(
    '<mujoco><worldbody><geom type="sphere" size=".1"/></worldbody></mujoco>'
)
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model, height=32, width=32)
try:
    renderer.update_scene(data)
    assert renderer.render().shape == (32, 32, 3)
finally:
    renderer.close()
""",
        backend=None,
    )
    assert result.returncode == 0, result.stderr


def test_package_preserves_an_explicit_backend_override() -> None:
    result = _python(
        """
import os
import cascade
assert os.environ["MUJOCO_GL"] == "glfw"
""",
        backend="glfw",
    )
    assert result.returncode == 0, result.stderr
