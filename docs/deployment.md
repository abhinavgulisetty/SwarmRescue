# Local deployment — 2026-10-06

Version 0.2.0 is an installable SwarmRescue release candidate. The wheel includes portable firmware sources/headers, emulator replay/linker assets, the frozen evaluation protocol, an exact runtime dependency lock and the `swarmrescue` CLI. Native routing therefore works outside the source checkout. Installed runs use writable collapse caches and archive the actual shipped Python/C sources with a content fingerprint rather than attributing them to an unrelated working-directory Git repository.

The local deployment prefix is `outputs/deployment/venv`, with the wheel in `outputs/releases/0.2.0/`. No system packages were installed, no Git push or public package publishing was performed. This target runs the simulator and portable kernel tools; it does not establish completion of the research or physical-board acceptance gates.

```bash
uv build --wheel --out-dir outputs/releases/0.2.0
uv venv --python 3.12 outputs/deployment/venv
uv pip install --python outputs/deployment/venv/bin/python --require-hashes -r swarmrescue/runtime-lock.txt
uv pip install --python outputs/deployment/venv/bin/python --no-deps outputs/releases/0.2.0/swarmrescue-0.2.0-py3-none-any.whl
outputs/deployment/venv/bin/swarmrescue doctor
```

The health check compiles/runs a native integer route and advances a minimal MuJoCo model; it also checks packaged firmware assets and Python support. Commands include `mission`, `tactical`, `view`, `evaluate`, `range-evaluation` and `firmware-replay`, delegating to their existing command-line interfaces.

Set `SWARMRESCUE_CACHE_DIR` when choosing a deployment cache explicitly. Installed packages otherwise use the XDG/user cache directory; source checkouts keep the existing `cascade/cache` path. Generated caches are never part of the runtime source fingerprint.

## Installation validation

All 665 regression tests and runtime Ruff pass, with the configured local ARM tools enabling the Cortex-M4 integration test. Fresh installation into a separate environment was tested from `/tmp`, so neither import resolution nor native C compilation depended on the checkout. Installed health checks, the tactical mission and a native firmware replay all pass. An installed eight-robot seed-42 physical mission reaches 8/8 sites and passes its mission acceptance checks; its source archive matches its installed-source fingerprint. Site arrival remains distinct from physical patient evacuation.

Artifacts include `outputs/releases/0.2.0/installed-mission.json`, its installed-source archive, `installed-tactical.json` and `installed-host-replay/`. These installation checkpoints may precede the final rebuilt release artifact; each report records its own installed fingerprint and package version. Do not infer that two builds with the same unpublished release-candidate version necessarily have the same source hash.

Current open gates are listed in [COMPLETION_GATES.md](../COMPLETION_GATES.md). Live firmware transport and deployment integration are the next checkpoint; whole-robot firmware, real sensor calibration, measured MCU timing/RAM, learned acceleration and patient evacuation remain open.
