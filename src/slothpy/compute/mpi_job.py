from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from slothpy.core.mpi_launch import resolve_mpi_launch_args
from slothpy.types.aliases import PathLike


@dataclass(frozen=True, slots=True)
class MPIJobResources:
    """
    Resources requested for one MPI-backed SlothPy computation.

    num_processes:
        MPI ranks.

    num_threads:
        Threads per MPI rank. This is propagated to common thread-control
        environment variables.
    """

    num_processes: int = 1
    num_threads: int = 1
    mpi_executable: str = "mpirun"
    python_executable: str = sys.executable
    extra_mpi_args: tuple[str, ...] = ()
    mpi_bind_to: str | None = "none"

    def __post_init__(self) -> None:
        if self.num_processes < 1:
            raise ValueError("num_processes must be >= 1.")
        if self.num_threads < 1:
            raise ValueError("num_threads must be >= 1.")


@dataclass(frozen=True, slots=True)
class MPIJobSpec:
    """
    Small JSON-serializable job description passed from the parent process to
    the MPI worker module.
    """

    worker_module: str
    shared_memory_manifest: str
    payload: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def read_json(cls, path: PathLike) -> MPIJobSpec:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_json_dict(data)

    def write_json(self, path: PathLike) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        temporary_path = path.with_suffix(path.suffix + ".tmp")
        temporary_path.write_text(
            json.dumps(self.to_json_dict(), indent=2),
            encoding="utf-8",
        )
        temporary_path.replace(path)

        return path

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json_dict(cls, data: Mapping[str, Any]) -> MPIJobSpec:
        payload = data.get("payload", {})
        if not isinstance(payload, dict):
            raise TypeError("MPIJobSpec payload must be a dictionary.")

        return cls(
            worker_module=str(data["worker_module"]),
            shared_memory_manifest=str(data["shared_memory_manifest"]),
            payload=payload,
        )


@dataclass(frozen=True, slots=True)
class MPIJobCompleted:
    """
    Completed MPI process information.
    """

    returncode: int
    stdout: str
    stderr: str
    command: tuple[str, ...]

    def check_returncode(self) -> None:
        if self.returncode != 0:
            raise RuntimeError(
                "MPI worker failed with return code "
                f"{self.returncode}.\n\n"
                f"Command:\n{' '.join(self.command)}\n\n"
                f"STDOUT:\n{self.stdout}\n\n"
                f"STDERR:\n{self.stderr}"
            )


@dataclass(slots=True)
class MPIJobRunner:
    """
    Parent-side launcher for one MPI worker process.
    """

    resources: MPIJobResources
    cwd: Path | None = None
    env: dict[str, str] = field(default_factory=dict)
    capture_output: bool = True

    def command(self, spec: MPIJobSpec, job_spec_path: PathLike) -> tuple[str, ...]:
        launch_args = resolve_mpi_launch_args(
            self.resources.extra_mpi_args,
            bind_to=self.resources.mpi_bind_to,
        )
        return (
            self.resources.mpi_executable,
            *launch_args,
            "-np",
            str(self.resources.num_processes),
            self.resources.python_executable,
            "-m",
            spec.worker_module,
            "--job-spec",
            str(job_spec_path),
        )

    def environment(self) -> dict[str, str]:
        env = dict(os.environ)
        env.update(self.env)

        threads = str(self.resources.num_threads)

        env.setdefault("OMP_NUM_THREADS", threads)
        env.setdefault("OPENBLAS_NUM_THREADS", threads)
        env.setdefault("MKL_NUM_THREADS", threads)
        env.setdefault("NUMEXPR_NUM_THREADS", threads)
        env.setdefault("NUMBA_NUM_THREADS", threads)

        return env

    def run(self, spec: MPIJobSpec, job_spec_path: PathLike) -> MPIJobCompleted:
        command = self.command(spec, job_spec_path)

        completed = subprocess.run(
            command,
            cwd=self.cwd,
            env=self.environment(),
            check=False,
            text=True,
            capture_output=self.capture_output,
        )

        result = MPIJobCompleted(
            returncode=completed.returncode,
            stdout=completed.stdout or "",
            stderr=completed.stderr or "",
            command=command,
        )
        result.check_returncode()
        return result


def read_mpi_job_spec_from_cli(
    argv: Sequence[str] | None = None,
) -> MPIJobSpec:
    """
    Worker-side helper.

    Every worker module can call this in main().
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-spec", required=True)
    args = parser.parse_args(argv)

    return MPIJobSpec.read_json(args.job_spec)


__all__ = [
    "MPIJobCompleted",
    "MPIJobResources",
    "MPIJobRunner",
    "MPIJobSpec",
    "read_mpi_job_spec_from_cli",
]
