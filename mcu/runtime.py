"""Persistent host/UART Cortex-M4 integer routing service and build tooling."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import select
import shutil
import struct
import subprocess
import tempfile
import threading
import time
import zlib
from contextlib import ExitStack
from pathlib import Path

import numpy as np

from mcu import INT32_MAX, MCUArena
from mcu.log_lmdp import LogQ16Model, LogQ16Solution, _array

MAX_NODES = 2048
MAX_EDGES = 8192
_HEADER = struct.Struct("<4sIIII")
ROOT = Path(__file__).resolve().parents[1]


def tool_environment(tool_root=None):
    environment = os.environ.copy()
    if tool_root:
        root = Path(tool_root).resolve()
        environment["PATH"] = (
            str(root / "usr/bin") + os.pathsep + environment.get("PATH", "")
        )
        environment["LD_LIBRARY_PATH"] = str(root / "usr/lib/x86_64-linux-gnu")
    return environment


def build_runtime(output, *, arm=False, tool_root=None):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    environment = tool_environment(tool_root)
    compiler = shutil.which(
        "arm-none-eabi-gcc" if arm else "cc", path=environment["PATH"]
    )
    if not compiler:
        raise RuntimeError("required firmware compiler is unavailable")
    executable = output / ("runtime.elf" if arm else "runtime-host")
    runtime = ROOT / "firmware/runtime"
    command = [
        compiler,
        "-std=c99",
        "-O2",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-I",
        str(ROOT / "firmware"),
        str(runtime / "service.c"),
        str(runtime / ("mps2.c" if arm else "host.c")),
        str(ROOT / "firmware/log_lmdp.c"),
    ]
    if arm:
        command += [
            "-mcpu=cortex-m4",
            "-mthumb",
            "-mfloat-abi=soft",
            "-ffreestanding",
            "-fno-builtin",
            "-fno-common",
            "-nostdlib",
            "-Wl,-T," + str(runtime / "mps2.ld"),
            "-Wl,-Map," + str(output / "runtime.map"),
            "-lgcc",
        ]
    command += ["-o", str(executable)]
    result = subprocess.run(
        command, capture_output=True, text=True, env=environment, timeout=120
    )
    if result.returncode:
        raise RuntimeError(result.stderr)
    metadata = {
        "schema_version": 1,
        "platform": "cortex-m4-qemu" if arm else "host-service",
        "max_nodes": MAX_NODES,
        "max_edges": MAX_EDGES,
        "executable_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
        "compiler": subprocess.run(
            [compiler, "--version"],
            capture_output=True,
            text=True,
            check=True,
            env=environment,
        ).stdout.splitlines()[0],
        "whole_robot_firmware": False,
        "hardware_cycles_measured": False,
    }
    if arm:
        size = shutil.which("arm-none-eabi-size", path=environment["PATH"])
        values = (
            subprocess.run(
                [size, str(executable)],
                capture_output=True,
                text=True,
                check=True,
                env=environment,
            )
            .stdout.splitlines()[1]
            .split()
        )
        (
            metadata["text_bytes"],
            metadata["data_bytes"],
            metadata["bss_and_stack_bytes"],
        ) = map(int, values[:3])
        metadata["kernel_service_ram_bytes"] = int(values[1]) + int(values[2])
        metadata["reserved_stack_bytes"] = 16384
    (output / "runtime.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return executable, metadata


class FirmwareTransportError(RuntimeError):
    """A bounded live service rejected a model or failed its transport checks."""


class FirmwareBackend:
    """One private persistent integer planner; requests are serialized and timed.

    The C image streams model arrays into bounded storage without a full
    payload buffer. The host quantizes inputs, verifies response CRC/identity,
    and checks firmware actions against the same integer model. No host Bellman
    solve substitutes for a failed or capped firmware request.
    """

    def __init__(self, executable, *, arm=False, tool_root=None, timeout_s=60):
        if not np.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("firmware timeout must be finite and positive")
        if os.name != "posix":
            raise RuntimeError("live firmware transport requires POSIX process pipes")
        self.executable = Path(executable).resolve()
        if not self.executable.is_file():
            raise ValueError("firmware executable is missing")
        metadata_path = self.executable.parent / "runtime.json"
        self.metadata = (
            json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        )
        digest = hashlib.sha256(self.executable.read_bytes()).hexdigest()
        self.image_sha256 = digest
        if self.metadata.get("executable_sha256", digest) != digest:
            raise ValueError("firmware executable does not match its build manifest")
        environment = tool_environment(tool_root)
        if arm:
            qemu = shutil.which("qemu-system-arm", path=environment["PATH"])
            if not qemu:
                raise RuntimeError("QEMU is unavailable")
            self.metadata["emulator"] = subprocess.run(
                [qemu, "--version"],
                capture_output=True,
                text=True,
                check=True,
                env=environment,
            ).stdout.splitlines()[0]
            command = [
                qemu,
                "-M",
                "mps2-an386",
                "-display",
                "none",
                "-monitor",
                "none",
                "-serial",
                "stdio",
                "-semihosting-config",
                "enable=on,target=native",
                "-kernel",
                str(self.executable),
            ]
        else:
            command = [str(self.executable)]
        self.resources = ExitStack()
        self.stderr = self.resources.enter_context(
            tempfile.TemporaryFile()  # noqa: SIM115 - the backend owns its lifetime
        )
        try:
            self.process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self.stderr,
                env=environment,
                bufsize=0,
            )
        except Exception:
            self.resources.close()
            raise
        os.set_blocking(self.process.stdin.fileno(), False)
        os.set_blocking(self.process.stdout.fileno(), False)
        self.timeout_s = float(timeout_s)
        self.lock = threading.Lock()
        self.sequence = 0
        self.closed = False
        self.requests = 0
        self.transport_bytes = 0
        self.wall_time_s = 0.0
        self.platform = "cortex-m4-qemu" if arm else "host-service"
        try:
            reply = self._rpc(0, b"")
            if len(reply) != 16:
                raise FirmwareTransportError("invalid firmware capabilities")
            protocol, nodes, edges, ram = struct.unpack("<IIII", reply)
            if (protocol, nodes, edges) != (1, MAX_NODES, MAX_EDGES):
                raise FirmwareTransportError("firmware protocol or capacities differ")
            if arm and self.metadata.get("kernel_service_ram_bytes", ram) != ram:
                raise FirmwareTransportError(
                    "firmware RAM layout differs from manifest"
                )
            if arm:
                self.metadata["kernel_service_ram_bytes"] = ram
        except Exception:
            self.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def _write(self, payload, deadline):
        cursor = 0
        while cursor < len(payload):
            left = deadline - time.monotonic()
            if left <= 0 or not select.select([], [self.process.stdin], [], left)[1]:
                raise FirmwareTransportError("firmware write deadline expired")
            try:
                count = os.write(
                    self.process.stdin.fileno(), payload[cursor : cursor + 16384]
                )
            except (BrokenPipeError, OSError) as error:
                raise FirmwareTransportError("firmware input closed") from error
            cursor += count

    def _read(self, count, deadline):
        result = bytearray()
        while len(result) < count:
            left = deadline - time.monotonic()
            if left <= 0 or not select.select([self.process.stdout], [], [], left)[0]:
                raise FirmwareTransportError("firmware response deadline expired")
            data = os.read(self.process.stdout.fileno(), count - len(result))
            if not data:
                raise FirmwareTransportError("firmware response closed")
            result.extend(data)
        return bytes(result)

    def _rpc(self, command, payload, *, checksum=None):
        if self.closed:
            raise FirmwareTransportError("firmware backend is closed")
        if self.sequence >= 0xFFFFFFFF:
            raise OverflowError("firmware request IDs exhausted")
        self.sequence += 1
        deadline = time.monotonic() + self.timeout_s
        request = (
            _HEADER.pack(
                b"EMRQ",
                command,
                self.sequence,
                len(payload),
                zlib.crc32(payload) if checksum is None else checksum,
            )
            + payload
        )
        self._write(request, deadline)
        magic, sequence, status, length, crc = _HEADER.unpack(
            self._read(_HEADER.size, deadline)
        )
        if magic != b"EMRS" or sequence != self.sequence or length > 8 + 8 * MAX_NODES:
            raise FirmwareTransportError("invalid firmware response identity/length")
        response = self._read(length, deadline)
        if zlib.crc32(response) != crc:
            raise FirmwareTransportError("firmware response checksum failed")
        if status:
            raise FirmwareTransportError(f"firmware rejected request: status {status}")
        if command == 1:
            self.transport_bytes += len(request) + _HEADER.size + len(response)
        return response

    def solve(self, model, *, arena=None):
        if not 1 <= model.n_states <= MAX_NODES or len(model.col_idx) > MAX_EDGES:
            raise ValueError("model exceeds firmware node/edge capacity")
        if not 1 <= model.config.max_sweeps <= 10000:
            raise ValueError("firmware sweep cap must be in 1..10000")
        storage = arena if arena is not None else MCUArena(200 * 1024)
        quantized = LogQ16Model.from_float(model, storage)
        arrays = (
            quantized.row_ptr,
            quantized.col_idx,
            quantized.negative_log_passive,
            quantized.cost_over_temperature,
            quantized.exit_mask,
            quantized.death_mask,
        )
        payload = struct.pack(
            "<III", model.n_states, len(model.col_idx), model.config.max_sweeps
        )
        payload += b"".join(
            array.astype("<i4", copy=False).tobytes()
            if array.dtype == np.int32
            else array.tobytes()
            for array in arrays
        )
        with self.lock:
            started = time.monotonic()
            response = self._rpc(1, payload)
            self.wall_time_s += time.monotonic() - started
            self.requests += 1
        n = model.n_states
        if len(response) != 8 + 8 * n:
            raise FirmwareTransportError("wrong firmware field length")
        sweeps, fixed = struct.unpack_from("<II", response)
        if not 1 <= sweeps <= model.config.max_sweeps or fixed not in (0, 1):
            raise FirmwareTransportError("invalid firmware solve status")
        values = np.frombuffer(response[8 : 8 + 4 * n], dtype="<i4")
        actions = np.frombuffer(response[8 + 4 * n :], dtype="<i4")
        if np.any(values < 0) or np.any(values[quantized.exit_mask.astype(bool)] != 0):
            raise FirmwareTransportError("invalid firmware field boundaries")
        # Validate the returned firmware policy without recomputing its field.
        for state in range(n):
            options = []
            if values[state] != INT32_MAX:
                for edge in range(
                    int(quantized.row_ptr[state]), int(quantized.row_ptr[state + 1])
                ):
                    node = int(quantized.col_idx[edge])
                    passive = int(quantized.negative_log_passive[edge])
                    if passive != INT32_MAX and values[node] < values[state]:
                        options.append((int(values[node]) + passive, node))
            expected = min(options)[1] if options else -1
            if actions[state] != expected:
                raise FirmwareTransportError("firmware policy disagrees with its field")
        current = _array(storage, values, np.int32)
        _array(storage, np.empty(n, np.int32), np.int32)
        return LogQ16Solution(
            quantized, current, sweeps, bool(fixed), storage.high_water
        )

    def close(self):
        if self.closed:
            return
        try:
            with self.lock:
                if self.process.poll() is None:
                    self._rpc(2, b"")
            self.process.wait(timeout=5)
        except (FirmwareTransportError, subprocess.TimeoutExpired, OverflowError):
            if self.process.poll() is None:
                self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        finally:
            self.closed = True
            self.process.stdin.close()
            self.process.stdout.close()
            self.resources.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", action="store_true")
    parser.add_argument("--tool-root", type=Path)
    parser.add_argument("--out", type=Path, default=Path("outputs/firmware/runtime"))
    args = parser.parse_args(argv)
    executable, metadata = build_runtime(
        args.out, arm=args.arm, tool_root=args.tool_root
    )
    print(json.dumps({"executable": str(executable), **metadata}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
