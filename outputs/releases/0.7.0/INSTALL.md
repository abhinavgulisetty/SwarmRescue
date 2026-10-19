# SwarmRescue 0.7.0 local installation

Python 3.12 on Linux is the validated environment. Extract the delivery bundle,
then run these commands from its directory with `uv` installed:

```bash
uv venv --python 3.12 bootstrap
uv pip install --python bootstrap/bin/python --require-hashes -r runtime-lock.txt
uv pip install --python bootstrap/bin/python --no-deps wheel/swarmrescue-0.7.0-py3-none-any.whl
bootstrap/bin/swarmrescue deploy --wheel wheel/swarmrescue-0.7.0-py3-none-any.whl --prefix ./deployment
./deployment/run-mission
```

That creates a standalone native simulator deployment. The launcher selects the
complete-rescue assignment workflow. `--rescue-assignment distance` selects its
matched legacy assignment control. `--render` enables a recorded simulation.

For Cortex-M4 instruction emulation, rerun `deploy` with `--tool-root` pointing
to a directory containing `usr/bin/arm-none-eabi-gcc` and
`usr/bin/qemu-system-arm`, or `/` when those tools are installed system-wide.
The installer builds and probes the service, then creates a launcher using two
workers. No physical board flashing is performed.

Reproduce the independent mathematical study from the installed distribution:

```bash
bootstrap/bin/swarmrescue certificate-evaluation --output-dir ./certificate-evidence
```

The bundled original-author receipt uses a separately downloaded, pinned
reference checkout. Its reproduction instructions and limitations are in
`docs/research/external-rescue/reproduction.md`. That source is not vendored.

The current machine's ready-to-run deployment is separate from this portable
bundle at `outputs/deployment/complete-rescue-0.7.0/`. Its absolute-path launcher
is not a relocatable executable; use the installer above on another machine.

Read `RESEARCH_DELIVERY.md` for proved model properties and open historical,
calibration and whole-robot hardware claims. Simulation completion does not
certify human physiology or an arbitrary physical fire.
