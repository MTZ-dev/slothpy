from __future__ import annotations

import asyncio
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Mapping, Sequence
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from enum import IntEnum, StrEnum
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from slothpy.io.shared_memory import _open_shared_memory

if TYPE_CHECKING:
    from rich.console import Console
    from rich.panel import Panel

# ---------------------------------------------------------------------------
# Status enums
# ---------------------------------------------------------------------------


class SltJobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"
    FINISHED = "finished"
    FAILED = "failed"


class SltProgressStatus(IntEnum):
    UNKNOWN = 0
    QUEUED = 1
    RUNNING = 2
    CANCELLING = 3
    CANCELLED = 4
    FINISHED = 5
    FAILED = 6


# ---------------------------------------------------------------------------
# Progress shared-memory block
# ---------------------------------------------------------------------------


_PROGRESS_SIZE = 16

_PROGRESS_DONE = 0
_PROGRESS_TOTAL = 1
_PROGRESS_STATUS = 2
_PROGRESS_CANCEL_REQUESTED = 3
_PROGRESS_HEARTBEAT_NS = 4
_PROGRESS_ERROR_CODE = 5
_PROGRESS_RESERVED_0 = 6
_PROGRESS_RESERVED_1 = 7


@dataclass(frozen=True, slots=True)
class SltProgressSpec:
    """
    JSON-serializable description of a progress shared-memory block.

    This can be stored inside MPI job-spec payload or shared-memory
    manifest. Worker rank 0 can attach using this spec.
    """

    name: str
    shape: tuple[int, ...] = (_PROGRESS_SIZE,)
    dtype: str = np.dtype(np.int64).str

    def to_json_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["shape"] = list(self.shape)
        return data

    @classmethod
    def from_json_dict(cls, data: Mapping[str, Any]) -> SltProgressSpec:
        return cls(
            name=str(data["name"]),
            shape=tuple(int(size) for size in data.get("shape", [_PROGRESS_SIZE])),
            dtype=str(data.get("dtype", np.dtype(np.int64).str)),
        )


@dataclass(frozen=True, slots=True)
class SltProgressSnapshot:
    done: int
    total: int
    status: SltProgressStatus
    cancel_requested: bool
    heartbeat_ns: int
    error_code: int = 0

    @property
    def fraction(self) -> float | None:
        if self.total <= 0:
            return None
        return max(0.0, min(1.0, self.done / self.total))

    @property
    def percent(self) -> float | None:
        if self.fraction is None:
            return None
        return 100.0 * self.fraction


@dataclass(slots=True)
class SltProgressTracker:
    """
    Parent-side shared-memory progress tracker.

    Parent process owns this object and should release it. MPI rank 0 should
    attach with ``SltWorkerProgress.attach(spec, track=False)`` and close only.

    The progress memory layout is an int64 array:

        [0] done
        [1] total
        [2] status
        [3] cancel_requested
        [4] heartbeat_ns
        [5] error_code
        [6:] reserved
    """

    spec: SltProgressSpec
    _shm: SharedMemory
    _array: np.ndarray
    owns_memory: bool = True
    _closed: bool = False
    _unlinked: bool = False

    @classmethod
    def create(
        cls,
        *,
        total: int = 0,
        status: SltProgressStatus = SltProgressStatus.QUEUED,
        track: bool = True,
    ) -> SltProgressTracker:
        if total < 0:
            raise ValueError("total must be >= 0.")

        nbytes = _PROGRESS_SIZE * np.dtype(np.int64).itemsize
        shm = _open_shared_memory(create=True, size=nbytes, track=track)

        array = np.ndarray((_PROGRESS_SIZE,), dtype=np.int64, buffer=shm.buf)
        array[...] = 0
        array[_PROGRESS_TOTAL] = int(total)
        array[_PROGRESS_STATUS] = int(status)
        array[_PROGRESS_HEARTBEAT_NS] = time.monotonic_ns()

        return cls(
            spec=SltProgressSpec(name=shm.name),
            _shm=shm,
            _array=array,
            owns_memory=True,
        )

    @classmethod
    def attach(
        cls,
        spec: SltProgressSpec,
        *,
        track: bool = False,
    ) -> SltProgressTracker:
        shm = _open_shared_memory(name=spec.name, create=False, track=track)
        array = np.ndarray(spec.shape, dtype=np.dtype(spec.dtype), buffer=shm.buf)

        return cls(
            spec=spec,
            _shm=shm,
            _array=array,
            owns_memory=False,
        )

    @property
    def array(self) -> np.ndarray:
        if self._closed:
            raise RuntimeError(f"Progress shared memory {self.spec.name!r} is closed.")
        return self._array

    def snapshot(self) -> SltProgressSnapshot:
        array = self.array
        return SltProgressSnapshot(
            done=int(array[_PROGRESS_DONE]),
            total=int(array[_PROGRESS_TOTAL]),
            status=SltProgressStatus(int(array[_PROGRESS_STATUS])),
            cancel_requested=bool(array[_PROGRESS_CANCEL_REQUESTED]),
            heartbeat_ns=int(array[_PROGRESS_HEARTBEAT_NS]),
            error_code=int(array[_PROGRESS_ERROR_CODE]),
        )

    def set_total(self, total: int) -> None:
        if total < 0:
            raise ValueError("total must be >= 0.")
        self.array[_PROGRESS_TOTAL] = np.int64(total)
        self.touch()

    def set_done(self, done: int) -> None:
        self.array[_PROGRESS_DONE] = np.int64(max(done, 0))
        self.touch()

    def advance(self, delta: int = 1) -> None:
        self.array[_PROGRESS_DONE] += np.int64(delta)
        self.touch()

    def set_status(self, status: SltProgressStatus) -> None:
        self.array[_PROGRESS_STATUS] = np.int64(status)
        self.touch()

    def set_running(self) -> None:
        self.set_status(SltProgressStatus.RUNNING)

    def set_finished(self) -> None:
        total = np.int64(self.array[_PROGRESS_TOTAL])
        if total > 0:
            self.array[_PROGRESS_DONE] = total
        self.set_status(SltProgressStatus.FINISHED)

    def set_failed(self, *, error_code: int = 1) -> None:
        self.array[_PROGRESS_ERROR_CODE] = np.int64(error_code)
        self.set_status(SltProgressStatus.FAILED)

    def request_cancel(self) -> None:
        self.array[_PROGRESS_CANCEL_REQUESTED] = 1
        self.set_status(SltProgressStatus.CANCELLING)

    def set_cancelled(self) -> None:
        self.array[_PROGRESS_CANCEL_REQUESTED] = 1
        self.set_status(SltProgressStatus.CANCELLED)

    def cancel_requested(self) -> bool:
        return bool(self.array[_PROGRESS_CANCEL_REQUESTED])

    def touch(self) -> None:
        self.array[_PROGRESS_HEARTBEAT_NS] = np.int64(time.monotonic_ns())

    def close(self) -> None:
        if self._closed:
            return

        self._closed = True
        self._array = np.ndarray((0,), dtype=np.int64)
        self._shm.close()

    def unlink(self) -> None:
        if self._unlinked:
            return

        if not self.owns_memory:
            raise RuntimeError(
                f"Progress shared memory {self.spec.name!r} is not owned here."
            )

        self._unlinked = True
        self._shm.unlink()

    def release(self) -> None:
        if self.owns_memory and not self._unlinked:
            self.unlink()
        self.close()

    def __enter__(self) -> SltProgressTracker:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.release() if self.owns_memory else self.close()


SltWorkerProgress = SltProgressTracker


def update_mpi_progress(
    *,
    comm: Any,
    progress: SltWorkerProgress | None,
    local_done: int,
    total: int | None = None,
    root: int = 0,
) -> bool:
    """
    Worker-side helper for MPI progress aggregation.

    All ranks call this. ``local_done`` is reduced to rank 0. Rank 0 writes
    the aggregate into the parent-visible shared progress block.

    Returns
    -------
    bool
        True if cancellation has been requested by the parent.
    """
    rank = comm.Get_rank()

    done_total = comm.reduce(int(local_done), op=None, root=root)

    if rank == root:
        if progress is None:
            cancel = False
        else:
            if total is not None:
                progress.set_total(int(total))
            progress.set_done(int(done_total))
            progress.set_running()
            cancel = progress.cancel_requested()
    else:
        cancel = False

    return bool(comm.bcast(cancel, root=root))


# ---------------------------------------------------------------------------
# Resource model
# ---------------------------------------------------------------------------


def _normalize_hostname(name: str) -> str:
    return name.strip().lower().split(".")[0]


def resolve_control_node_name(
    nodes: Sequence[SltNodeResources],
    *,
    control_node_name: str | None = None,
) -> str:
    """
    Resolve the node where the session parent runs (MPI rank 0 must land here).

    Rank 0 attaches to POSIX shared memory created by the parent process, so it
    must be scheduled on the same machine.
    """
    names = {node.name for node in nodes}

    if control_node_name is not None:
        if control_node_name not in names:
            raise ValueError(
                f"control_node_name {control_node_name!r} is not in the resource pool: "
                f"{sorted(names)}."
            )
        return control_node_name

    host_short = _normalize_hostname(socket.gethostname())
    for node in nodes:
        if _normalize_hostname(node.name) == host_short:
            return node.name

    fqdn = socket.getfqdn()
    if fqdn in names:
        return fqdn

    if "localhost" in names:
        return "localhost"

    return nodes[0].name


@dataclass(frozen=True, slots=True)
class SltNodeResources:
    """
    Static description of one node.

    Parameters
    ----------
    name
        Hostname as understood by MPI.
    cores
        Logical CPU cores available to SlothPy on this node.
    """

    name: str
    cores: int

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Node name cannot be empty.")

        if self.cores < 1:
            raise ValueError("cores must be >= 1.")


@dataclass(frozen=True, slots=True)
class SltResourceRequest:
    """
    Resource request for one computation.

    num_processes:
        Total number of MPI ranks.

    num_threads:
        CPU threads per MPI rank.

    num_nodes:
        Exact number of nodes to use. If None, resources are packed greedily.

    exclusive_nodes:
        If True, a touched node is treated as fully occupied.
    """

    num_processes: int = 1
    num_threads: int = 1
    num_nodes: int | None = None
    exclusive_nodes: bool = False

    def __post_init__(self) -> None:
        if self.num_processes < 1:
            raise ValueError("num_processes must be >= 1.")

        if self.num_threads < 1:
            raise ValueError("num_threads must be >= 1.")

        if self.num_nodes is not None:
            if self.num_nodes < 1:
                raise ValueError("num_nodes must be >= 1 when provided.")
            if self.num_processes < self.num_nodes:
                raise ValueError("num_processes must be >= num_nodes.")

    @property
    def total_cores(self) -> int:
        return self.num_processes * self.num_threads


@dataclass(frozen=True, slots=True)
class SltNodeAllocation:
    node_name: str
    ranks: int
    threads_per_rank: int
    reserved_cores: int
    exclusive: bool = False


@dataclass(frozen=True, slots=True)
class SltAllocation:
    """
    Concrete resources reserved for one computation.

    The first node is always the session control node. OpenMPI assigns global
    rank 0 to the first slot on the first host in the hostfile, so rank 0 runs
    where the parent can share memory.
    """

    request: SltResourceRequest
    nodes: tuple[SltNodeAllocation, ...]
    control_node_name: str

    def __post_init__(self) -> None:
        if not self.nodes:
            raise ValueError("SltAllocation requires at least one node.")

        if self.nodes[0].node_name != self.control_node_name:
            raise ValueError(
                "The first allocated node must be the session control node "
                f"({self.control_node_name!r}) so MPI rank 0 can attach to "
                "parent shared memory."
            )

        if self.nodes[0].ranks < 1:
            raise ValueError(
                f"Control node {self.control_node_name!r} must host at least one "
                "MPI rank."
            )

    @property
    def rank0_node_name(self) -> str:
        return self.nodes[0].node_name

    @property
    def num_processes(self) -> int:
        return sum(node.ranks for node in self.nodes)

    @property
    def num_threads(self) -> int:
        return self.request.num_threads

    @property
    def total_reserved_cores(self) -> int:
        return sum(node.reserved_cores for node in self.nodes)

    @property
    def node_names(self) -> tuple[str, ...]:
        return tuple(node.node_name for node in self.nodes)

    @property
    def is_localhost_only(self) -> bool:
        return self.node_names == ("localhost",)

    def write_openmpi_hostfile(self, path: str | Path) -> Path:
        """
        Write an OpenMPI hostfile.

        Nodes are written in allocation order; the first host receives MPI
        rank 0 (the session control node).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        lines = [
            f"{node.node_name} slots={node.ranks}"
            for node in self.nodes
            if node.ranks > 0
        ]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def openmpi_extra_args(self, hostfile: str | Path) -> tuple[str, ...]:
        """
        OpenMPI hostfile / mapping options for multi-node launches.

        Uses ``--bind-to none`` so each rank sees the node CPUs; per-rank thread
        counts come from the job environment (see :func:`build_mpi_environment`).
        """
        from slothpy.core.mpi_launch import openmpi_bind_extra_args

        return (
            "--hostfile",
            str(hostfile),
            "--map-by",
            f"slot:PE={self.num_threads}",
            *openmpi_bind_extra_args("none"),
        )


@dataclass(frozen=True, slots=True)
class SltResourceNodeSnapshot:
    node_name: str
    total_cores: int
    used_cores: int
    free_cores: int
    exclusive: bool


class SltResourcePool:
    """
    Thread-safe resource allocator.

    The control node is where the session parent runs. MPI rank 0 is always
    placed on that node so it can attach to parent-owned shared memory.

    MPI workers are launched with ``--bind-to none`` (see
    :mod:`slothpy.core.mpi_launch`); this pool tracks how many logical cores are
    reserved on each node. Per-rank thread limits are set in the worker environment
    by :func:`build_mpi_environment`.
    """

    def __init__(
        self,
        nodes: Sequence[SltNodeResources],
        *,
        control_node_name: str | None = None,
        parent_reserved_cores: int = 1,
    ) -> None:
        if not nodes:
            raise ValueError("SltResourcePool requires at least one node.")

        names = [node.name for node in nodes]
        if len(names) != len(set(names)):
            raise ValueError("Node names must be unique.")

        if parent_reserved_cores < 0:
            raise ValueError("parent_reserved_cores must be >= 0.")

        self._nodes = tuple(nodes)
        self._control_node_name = resolve_control_node_name(
            self._nodes,
            control_node_name=control_node_name,
        )
        self._parent_reserved_cores = parent_reserved_cores
        self._used_cores: dict[str, int] = {node.name: 0 for node in self._nodes}
        self._exclusive_nodes: set[str] = set()
        self._lock = threading.Lock()

        control_node = self._node_by_name(self._control_node_name)
        if control_node.cores < parent_reserved_cores + 1:
            raise ValueError(
                f"Control node {self._control_node_name!r} must have enough cores "
                f"for the session parent (reserved {parent_reserved_cores}) and at "
                "least one MPI rank."
            )

    @classmethod
    def local(
        cls,
        *,
        cores: int | None = None,
        parent_reserved_cores: int = 1,
    ) -> SltResourcePool:
        # Machine-wide logical CPUs (not the parent's affinity mask).
        detected = os.cpu_count() or 1
        return cls(
            [
                SltNodeResources(
                    name="localhost",
                    cores=detected if cores is None else cores,
                )
            ],
            control_node_name="localhost",
            parent_reserved_cores=parent_reserved_cores,
        )

    @classmethod
    def from_tuples(
        cls,
        nodes: Sequence[tuple[str, int]],
        *,
        control_node_name: str | None = None,
        parent_reserved_cores: int = 1,
    ) -> SltResourcePool:
        return cls(
            [SltNodeResources(name, cores) for name, cores in nodes],
            control_node_name=control_node_name,
            parent_reserved_cores=parent_reserved_cores,
        )

    @property
    def control_node_name(self) -> str:
        return self._control_node_name

    @property
    def nodes(self) -> tuple[SltNodeResources, ...]:
        return self._nodes

    def snapshot(self) -> tuple[SltResourceNodeSnapshot, ...]:
        with self._lock:
            return tuple(
                SltResourceNodeSnapshot(
                    node_name=node.name,
                    total_cores=node.cores,
                    used_cores=self._used_cores[node.name],
                    free_cores=self._free_cores_unlocked(node),
                    exclusive=node.name in self._exclusive_nodes,
                )
                for node in self._nodes
            )

    def try_acquire(self, request: SltResourceRequest) -> SltAllocation | None:
        with self._lock:
            allocation = self._try_allocate_unlocked(request)
            if allocation is None:
                return None

            for node_allocation in allocation.nodes:
                self._used_cores[node_allocation.node_name] += (
                    node_allocation.reserved_cores
                )
                if node_allocation.exclusive:
                    self._exclusive_nodes.add(node_allocation.node_name)

            return allocation

    def release(self, allocation: SltAllocation) -> None:
        with self._lock:
            for node_allocation in allocation.nodes:
                used = self._used_cores[node_allocation.node_name]
                used -= node_allocation.reserved_cores
                self._used_cores[node_allocation.node_name] = max(used, 0)

                if node_allocation.exclusive:
                    self._exclusive_nodes.discard(node_allocation.node_name)

    def _node_by_name(self, node_name: str) -> SltNodeResources:
        for node in self._nodes:
            if node.name == node_name:
                return node
        raise KeyError(node_name)

    def _nodes_in_allocation_order(self) -> tuple[SltNodeResources, ...]:
        control = self._node_by_name(self._control_node_name)
        others = tuple(node for node in self._nodes if node.name != control.name)
        return (control, *others)

    def _free_cores_unlocked(self, node: SltNodeResources) -> int:
        if node.name in self._exclusive_nodes:
            return 0

        free = node.cores - self._used_cores[node.name]
        if node.name == self._control_node_name:
            free -= self._parent_reserved_cores

        return max(free, 0)

    def _rank_capacity_unlocked(
        self,
        node: SltNodeResources,
        request: SltResourceRequest,
    ) -> int:
        return self._free_cores_unlocked(node) // request.num_threads

    def _try_allocate_unlocked(
        self,
        request: SltResourceRequest,
    ) -> SltAllocation | None:
        if request.num_nodes is None:
            return self._try_allocate_packed_unlocked(request)
        return self._try_allocate_exact_nodes_unlocked(request)

    def _try_allocate_packed_unlocked(
        self,
        request: SltResourceRequest,
    ) -> SltAllocation | None:
        ranks_left = request.num_processes
        allocations: list[SltNodeAllocation] = []

        for node in self._nodes_in_allocation_order():
            capacity = self._rank_capacity_unlocked(node, request)
            ranks_on_node = min(ranks_left, capacity)

            if ranks_on_node <= 0:
                continue

            reserved_cores = (
                node.cores
                if request.exclusive_nodes
                else ranks_on_node * request.num_threads
            )

            allocations.append(
                SltNodeAllocation(
                    node_name=node.name,
                    ranks=ranks_on_node,
                    threads_per_rank=request.num_threads,
                    reserved_cores=reserved_cores,
                    exclusive=request.exclusive_nodes,
                )
            )

            ranks_left -= ranks_on_node

            if ranks_left == 0:
                break

        if ranks_left != 0:
            return None

        return self._finalize_allocation(request, allocations)

    def _try_allocate_exact_nodes_unlocked(
        self,
        request: SltResourceRequest,
    ) -> SltAllocation | None:
        assert request.num_nodes is not None

        control = self._node_by_name(self._control_node_name)
        if self._rank_capacity_unlocked(control, request) < 1:
            return None

        candidate_nodes = [
            node
            for node in self._nodes
            if self._rank_capacity_unlocked(node, request) >= 1
        ]

        if control.name not in {node.name for node in candidate_nodes}:
            return None

        if len(candidate_nodes) < request.num_nodes:
            return None

        other_candidates = [
            node for node in candidate_nodes if node.name != control.name
        ]
        selected_nodes = (control, *other_candidates[: request.num_nodes - 1])

        capacities = {
            node.name: self._rank_capacity_unlocked(node, request)
            for node in selected_nodes
        }

        if sum(capacities.values()) < request.num_processes:
            return None

        ranks_by_node = {node.name: 1 for node in selected_nodes}
        ranks_left = request.num_processes - request.num_nodes

        for node in selected_nodes:
            if ranks_left == 0:
                break

            additional_capacity = capacities[node.name] - 1
            additional = min(ranks_left, additional_capacity)
            ranks_by_node[node.name] += additional
            ranks_left -= additional

        if ranks_left != 0:
            return None

        allocations: list[SltNodeAllocation] = []

        for node in selected_nodes:
            ranks_on_node = ranks_by_node[node.name]
            reserved_cores = (
                node.cores
                if request.exclusive_nodes
                else ranks_on_node * request.num_threads
            )

            allocations.append(
                SltNodeAllocation(
                    node_name=node.name,
                    ranks=ranks_on_node,
                    threads_per_rank=request.num_threads,
                    reserved_cores=reserved_cores,
                    exclusive=request.exclusive_nodes,
                )
            )

        return self._finalize_allocation(request, allocations)

    def _finalize_allocation(
        self,
        request: SltResourceRequest,
        allocations: list[SltNodeAllocation],
    ) -> SltAllocation:
        ordered = sorted(
            allocations,
            key=lambda node_allocation: (
                0 if node_allocation.node_name == self._control_node_name else 1,
                node_allocation.node_name,
            ),
        )
        return SltAllocation(
            request=request,
            nodes=tuple(ordered),
            control_node_name=self._control_node_name,
        )


# ---------------------------------------------------------------------------
# MPI process handling
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class MPIProcessHandle:
    """
    Handle for an externally launched MPI process.

    ``start_mpi_process`` starts the subprocess in a separate process group on
    POSIX systems, so terminate/kill signal the whole MPI-launched process tree.
    """

    process: subprocess.Popen[str]
    command: tuple[str, ...]

    @property
    def returncode(self) -> int | None:
        return self.process.poll()

    @property
    def running(self) -> bool:
        return self.returncode is None

    def wait(self, timeout: float | None = None) -> int:
        return self.process.wait(timeout=timeout)

    def communicate(self, timeout: float | None = None) -> tuple[str, str]:
        stdout, stderr = self.process.communicate(timeout=timeout)
        return stdout or "", stderr or ""

    def terminate(self, *, grace_seconds: float = 5.0) -> None:
        if not self.running:
            return

        self._send_signal(signal.SIGTERM)

        try:
            self.process.wait(timeout=grace_seconds)
        except subprocess.TimeoutExpired:
            self.kill()

    def kill(self) -> None:
        if not self.running:
            return

        self._send_signal(signal.SIGKILL)

    def _send_signal(self, sig: signal.Signals) -> None:
        if os.name == "posix":
            try:
                os.killpg(self.process.pid, sig)
            except ProcessLookupError:
                return
        else:
            if sig == signal.SIGTERM:
                self.process.terminate()
            else:
                self.process.kill()


def build_mpi_environment(
    *,
    num_threads: int,
    extra_env: dict[str, str] | None = None,
) -> dict[str, str]:
    env = dict(os.environ)
    threads = str(num_threads)

    env.setdefault("OMP_NUM_THREADS", threads)
    env.setdefault("OPENBLAS_NUM_THREADS", threads)
    env.setdefault("MKL_NUM_THREADS", threads)
    env.setdefault("NUMEXPR_NUM_THREADS", threads)
    env.setdefault("NUMBA_NUM_THREADS", threads)

    if extra_env is not None:
        env.update(extra_env)

    return env


def start_mpi_process(
    *,
    worker_module: str,
    job_spec_path: str | Path,
    allocation: SltAllocation | None = None,
    mpi_executable: str = "mpirun",
    python_executable: str = sys.executable,
    extra_mpi_args: Sequence[str] = (),
    mpi_bind_to: str | None = "none",
    cwd: str | Path | None = None,
    env: dict[str, str] | None = None,
    capture_output: bool = True,
) -> MPIProcessHandle:
    """
    Launch:

        mpirun --bind-to none -np N python -m worker_module --job-spec job_spec_path

    CPU placement is left to :class:`SltResourcePool` (cores reserved per job) and
  ``build_mpi_environment`` (thread caps per rank). Pass ``mpi_bind_to=None`` to
    skip ``--bind-to`` for non-OpenMPI launchers.
    """
    from slothpy.core.mpi_launch import resolve_mpi_launch_args
    if allocation is None:
        num_processes = 1
        num_threads = 1
    else:
        num_processes = allocation.num_processes
        num_threads = allocation.num_threads

    launch_args = resolve_mpi_launch_args(extra_mpi_args, bind_to=mpi_bind_to)

    command = (
        mpi_executable,
        *launch_args,
        "-np",
        str(num_processes),
        python_executable,
        "-m",
        worker_module,
        "--job-spec",
        str(job_spec_path),
    )

    process = subprocess.Popen(
        command,
        cwd=None if cwd is None else Path(cwd),
        env=build_mpi_environment(num_threads=num_threads, extra_env=env),
        text=True,
        stdout=subprocess.PIPE if capture_output else None,
        stderr=subprocess.PIPE if capture_output else None,
        start_new_session=True,
    )

    return MPIProcessHandle(process=process, command=command)


# ---------------------------------------------------------------------------
# Computation protocol
# ---------------------------------------------------------------------------


class SltRunnableComputation[ResultT](Protocol):
    """
    Minimal protocol expected by SltSession.

    Computation classes still own:
    - shared-memory staging,
    - MPI job-spec creation,
    - worker module selection,
    - result composition.
    """

    @property
    def resource_request(self) -> SltResourceRequest: ...

    def run_with_allocation(
        self, allocation: SltAllocation | None = None
    ) -> ResultT: ...

    def cancel(self, *, grace_seconds: float = 5.0) -> None: ...


def _computation_name(computation: object) -> str:
    name = getattr(computation, "computation_name", None)
    if isinstance(name, str):
        return name
    return type(computation).__name__


def _request_computation_cancel(computation: object) -> None:
    request_cancel = getattr(computation, "request_cancel", None)
    if callable(request_cancel):
        request_cancel()


def _computation_progress_snapshot(
    computation: object,
) -> SltProgressSnapshot | None:
    progress_snapshot = getattr(computation, "progress_snapshot", None)
    if callable(progress_snapshot):
        snapshot = progress_snapshot()
        if isinstance(snapshot, SltProgressSnapshot):
            return snapshot

    progress_tracker = getattr(computation, "progress_tracker", None)
    if isinstance(progress_tracker, SltProgressTracker):
        try:
            return progress_tracker.snapshot()
        except Exception:
            return None

    return None


# ---------------------------------------------------------------------------
# Snapshots
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SltJobSnapshot:
    job_id: str
    name: str
    status: SltJobStatus
    allocation: SltAllocation | None
    progress: SltProgressSnapshot | None
    exception: str | None = None

    @property
    def progress_fraction(self) -> float | None:
        if self.progress is None:
            if self.status == SltJobStatus.FINISHED:
                return 1.0
            return None
        return self.progress.fraction

    @property
    def progress_percent(self) -> float | None:
        fraction = self.progress_fraction
        if fraction is None:
            return None
        return 100.0 * fraction


@dataclass(frozen=True, slots=True)
class SltSessionSnapshot:
    resources: tuple[SltResourceNodeSnapshot, ...]
    jobs: tuple[SltJobSnapshot, ...]
    closed: bool

    @property
    def queued(self) -> int:
        return sum(job.status == SltJobStatus.QUEUED for job in self.jobs)

    @property
    def running(self) -> int:
        return sum(job.status == SltJobStatus.RUNNING for job in self.jobs)

    @property
    def cancelling(self) -> int:
        return sum(job.status == SltJobStatus.CANCELLING for job in self.jobs)

    @property
    def finished(self) -> int:
        return sum(job.status == SltJobStatus.FINISHED for job in self.jobs)

    @property
    def failed(self) -> int:
        return sum(job.status == SltJobStatus.FAILED for job in self.jobs)

    @property
    def cancelled(self) -> int:
        return sum(job.status == SltJobStatus.CANCELLED for job in self.jobs)


# ---------------------------------------------------------------------------
# Job handle
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class SltJob[ResultT]:
    job_id: str
    computation: SltRunnableComputation[ResultT]
    _future: Future[ResultT]
    _status: SltJobStatus = SltJobStatus.QUEUED
    _allocation: SltAllocation | None = None
    _cancel_requested: threading.Event = field(default_factory=threading.Event)
    _last_progress: SltProgressSnapshot | None = None
    _last_exception: BaseException | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def name(self) -> str:
        return _computation_name(self.computation)

    @property
    def status(self) -> SltJobStatus:
        with self._lock:
            return self._status

    @property
    def allocation(self) -> SltAllocation | None:
        with self._lock:
            return self._allocation

    @property
    def cancel_requested(self) -> bool:
        return self._cancel_requested.is_set()

    def _set_status(self, status: SltJobStatus) -> None:
        with self._lock:
            self._status = status

    def _set_allocation(self, allocation: SltAllocation | None) -> None:
        with self._lock:
            self._allocation = allocation

    def _set_last_exception(self, exc: BaseException | None) -> None:
        with self._lock:
            self._last_exception = exc

    def _set_last_progress(self, snapshot: SltProgressSnapshot | None) -> None:
        with self._lock:
            self._last_progress = snapshot

    def _safe_set_result(self, result: ResultT) -> None:
        if not self._future.done():
            self._future.set_result(result)

    def _safe_set_exception(self, exc: BaseException) -> None:
        if not self._future.done():
            self._future.set_exception(exc)

    def progress_snapshot(self) -> SltProgressSnapshot | None:
        snapshot = _computation_progress_snapshot(self.computation)
        if snapshot is not None:
            self._set_last_progress(snapshot)
            return snapshot

        with self._lock:
            return self._last_progress

    def snapshot(self) -> SltJobSnapshot:
        with self._lock:
            exception = (
                None if self._last_exception is None else repr(self._last_exception)
            )

        return SltJobSnapshot(
            job_id=self.job_id,
            name=self.name,
            status=self.status,
            allocation=self.allocation,
            progress=self.progress_snapshot(),
            exception=exception,
        )

    def request_cancel(self) -> None:
        """
        Soft cancellation.

        For queued jobs this completes the future as cancelled. For running jobs
        this sets the cancellation flag and asks the computation/progress tracker
        to exit cleanly. The MPI process is not forcibly killed here.
        """
        self._cancel_requested.set()

        if self.status == SltJobStatus.QUEUED:
            self._set_status(SltJobStatus.CANCELLED)
            self._safe_set_exception(CancelledError())
            return

        if self.status == SltJobStatus.RUNNING:
            self._set_status(SltJobStatus.CANCELLING)
            _request_computation_cancel(self.computation)

            progress_tracker = getattr(self.computation, "progress_tracker", None)
            if isinstance(progress_tracker, SltProgressTracker):
                try:
                    progress_tracker.request_cancel()
                except Exception:
                    pass

    def terminate(self, *, grace_seconds: float = 5.0) -> None:
        """
        Harder cancellation: ask the computation to terminate its MPI process.
        """
        self._cancel_requested.set()

        if self.status == SltJobStatus.QUEUED:
            self._set_status(SltJobStatus.CANCELLED)
            self._safe_set_exception(CancelledError())
            return

        self._set_status(SltJobStatus.CANCELLING)
        self.computation.cancel(grace_seconds=grace_seconds)

    def kill(self) -> None:
        """
        Strong cancellation.

        If the computation exposes ``kill()``, use it. Otherwise fall back to
        ``cancel(grace_seconds=0)``.
        """
        self._cancel_requested.set()
        self._set_status(SltJobStatus.CANCELLING)

        kill = getattr(self.computation, "kill", None)
        if callable(kill):
            kill()
        else:
            self.computation.cancel(grace_seconds=0.0)

    def cancel(self, *, hard: bool = False, grace_seconds: float = 5.0) -> None:
        if hard:
            self.terminate(grace_seconds=grace_seconds)
        else:
            self.request_cancel()

    def done(self) -> bool:
        return self._future.done()

    def result(self, timeout: float | None = None) -> ResultT:
        return self._future.result(timeout=timeout)

    async def result_async(self) -> ResultT:
        return await asyncio.wrap_future(self._future)

    def exception(self, timeout: float | None = None) -> BaseException | None:
        return self._future.exception(timeout=timeout)


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class SltSession[ResultT]:
    """
    Asynchronous SlothPy computation session.

    Use in notebooks as:

        session = SltSession()
        session

    For a live marimo dashboard, use a refresh cell:

        refresh = mo.ui.refresh(default_interval="500ms")
        mo.vstack([refresh, mo.Html(session.dashboard_html())])

    For terminal scripts, use :meth:`run_dashboard` (blocking) or
    :meth:`run_dashboard_async` (non-blocking refresh loop)::

        session.run_dashboard(exit_when_done=True)
        await session.run_dashboard_async(interval=0.25)

    Ctrl+C / SIGTERM (when :attr:`install_signal_handlers` is true, the default)
    hard-cancels all jobs via the MPI process group (SIGTERM, then SIGKILL on a
    second interrupt). Cooperative cancel alone is available through
    :meth:`request_cancel`; use :meth:`terminate`, :meth:`kill`, or
    :meth:`kill_all` when workers must stop immediately.

    MPI rank 0 always runs on :attr:`control_node_name` (the machine where this
    session creates shared memory). The resource pool reserves
    ``parent_reserved_cores`` on that node for the parent process.
    """

    resource_pool: SltResourcePool | None = None
    max_running_jobs: int = 16
    install_signal_handlers: bool = True
    interrupt_grace_seconds: float = 5.0

    _executor: ThreadPoolExecutor = field(init=False)
    _scheduler_thread: threading.Thread = field(init=False)
    _condition: threading.Condition = field(
        default_factory=threading.Condition,
        init=False,
    )
    _pending: list[SltJob[Any]] = field(default_factory=list, init=False)
    _jobs: dict[str, SltJob[Any]] = field(default_factory=dict, init=False)
    _closed: bool = field(default=False, init=False)
    _interrupt_handlers_registered: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if self.resource_pool is None:
            self.resource_pool = SltResourcePool.local()

        if self.max_running_jobs < 1:
            raise ValueError("max_running_jobs must be >= 1.")

        self._executor = ThreadPoolExecutor(
            max_workers=self.max_running_jobs,
            thread_name_prefix="slothpy-computation",
        )

        self._scheduler_thread = threading.Thread(
            target=self._scheduler_loop,
            name="slothpy-session-scheduler",
            daemon=True,
        )
        self._scheduler_thread.start()

        if self.install_signal_handlers:
            from slothpy.core.slt_interrupt import register_session_interrupt_handlers

            register_session_interrupt_handlers(
                self,
                grace_seconds=self.interrupt_grace_seconds,
            )
            self._interrupt_handlers_registered = True

    @classmethod
    def local(
        cls,
        *,
        cores: int | None = None,
        parent_reserved_cores: int = 1,
        max_running_jobs: int = 16,
        install_signal_handlers: bool = True,
        interrupt_grace_seconds: float = 5.0,
    ) -> SltSession:
        return cls(
            resource_pool=SltResourcePool.local(
                cores=cores,
                parent_reserved_cores=parent_reserved_cores,
            ),
            max_running_jobs=max_running_jobs,
            install_signal_handlers=install_signal_handlers,
            interrupt_grace_seconds=interrupt_grace_seconds,
        )

    @classmethod
    def from_nodes(
        cls,
        nodes: Sequence[SltNodeResources | tuple[str, int]],
        *,
        control_node_name: str | None = None,
        parent_reserved_cores: int = 1,
        max_running_jobs: int = 16,
        install_signal_handlers: bool = True,
        interrupt_grace_seconds: float = 5.0,
    ) -> SltSession:
        node_resources = [
            node if isinstance(node, SltNodeResources) else SltNodeResources(*node)
            for node in nodes
        ]

        return cls(
            resource_pool=SltResourcePool(
                node_resources,
                control_node_name=control_node_name,
                parent_reserved_cores=parent_reserved_cores,
            ),
            max_running_jobs=max_running_jobs,
            install_signal_handlers=install_signal_handlers,
            interrupt_grace_seconds=interrupt_grace_seconds,
        )

    @property
    def control_node_name(self) -> str:
        """
        Node where this session parent runs.

        MPI rank 0 is always scheduled on this node so it can attach to
        parent-owned shared memory.
        """
        assert self.resource_pool is not None
        return self.resource_pool.control_node_name

    def submit(
        self,
        computation: SltRunnableComputation[ResultT],
        *,
        job_id: str | None = None,
    ) -> SltJob[ResultT]:
        with self._condition:
            if self._closed:
                raise RuntimeError("Cannot submit to a closed SltSession.")

            final_job_id = job_id or f"slt-job-{uuid.uuid4().hex[:12]}"

            if final_job_id in self._jobs:
                raise KeyError(f"Job id {final_job_id!r} already exists.")

            future: Future[ResultT] = Future()

            job = SltJob(
                job_id=final_job_id,
                computation=computation,
                _future=future,
            )

            self._jobs[final_job_id] = job
            self._pending.append(job)
            self._condition.notify_all()

            return job

    def jobs(self) -> dict[str, SltJob[Any]]:
        with self._condition:
            return dict(self._jobs)

    def pending_jobs(self) -> list[SltJob[Any]]:
        with self._condition:
            return list(self._pending)

    def get_job(self, job_id: str) -> SltJob[Any]:
        with self._condition:
            return self._jobs[job_id]

    def status(self) -> dict[str, SltJobStatus]:
        with self._condition:
            return {job_id: job.status for job_id, job in self._jobs.items()}

    def resource_snapshot(self) -> tuple[SltResourceNodeSnapshot, ...]:
        assert self.resource_pool is not None
        return self.resource_pool.snapshot()

    def snapshot(self) -> SltSessionSnapshot:
        with self._condition:
            jobs = tuple(job.snapshot() for job in self._jobs.values())
            closed = self._closed

        return SltSessionSnapshot(
            resources=self.resource_snapshot(),
            jobs=jobs,
            closed=closed,
        )

    def request_cancel(self, job_id: str) -> None:
        job = self.get_job(job_id)
        job.request_cancel()

        with self._condition:
            self._condition.notify_all()

    def terminate(self, job_id: str, *, grace_seconds: float = 5.0) -> None:
        job = self.get_job(job_id)
        job.terminate(grace_seconds=grace_seconds)

        with self._condition:
            self._condition.notify_all()

    def kill(self, job_id: str) -> None:
        job = self.get_job(job_id)
        job.kill()

        with self._condition:
            self._condition.notify_all()

    def cancel(
        self,
        job_id: str,
        *,
        hard: bool = False,
        grace_seconds: float = 5.0,
    ) -> None:
        job = self.get_job(job_id)
        job.cancel(hard=hard, grace_seconds=grace_seconds)

        with self._condition:
            self._condition.notify_all()

    def cancel_all(
        self,
        *,
        hard: bool = False,
        grace_seconds: float = 5.0,
    ) -> None:
        with self._condition:
            jobs = list(self._jobs.values())

        for job in jobs:
            if not job.done():
                job.cancel(hard=hard, grace_seconds=grace_seconds)

        with self._condition:
            self._condition.notify_all()

    def kill_all(self) -> None:
        """
        Forcibly SIGKILL every MPI process group for non-finished jobs.

        Use on a second Ctrl+C or when cooperative / SIGTERM cancellation is too slow.
        """
        with self._condition:
            jobs = list(self._jobs.values())

        for job in jobs:
            if job.done():
                continue

            if job.status == SltJobStatus.QUEUED:
                job._cancel_requested.set()
                job._set_status(SltJobStatus.CANCELLED)
                job._safe_set_exception(CancelledError())
            else:
                job.kill()

        with self._condition:
            self._condition.notify_all()

    def prune_finished(self) -> None:
        with self._condition:
            removable = [
                job_id
                for job_id, job in self._jobs.items()
                if job.status
                in {
                    SltJobStatus.CANCELLED,
                    SltJobStatus.FINISHED,
                    SltJobStatus.FAILED,
                }
                and job.done()
            ]

            for job_id in removable:
                self._jobs.pop(job_id, None)

    def shutdown(
        self,
        *,
        cancel_running: bool = False,
        hard_cancel: bool = True,
        grace_seconds: float = 5.0,
        wait: bool = True,
    ) -> None:
        if self._interrupt_handlers_registered:
            from slothpy.core.slt_interrupt import unregister_session_interrupt_handlers

            unregister_session_interrupt_handlers(self)
            self._interrupt_handlers_registered = False

        if cancel_running:
            self.cancel_all(hard=hard_cancel, grace_seconds=grace_seconds)

        with self._condition:
            self._closed = True
            self._condition.notify_all()

        if wait:
            self._scheduler_thread.join()

        self._executor.shutdown(wait=wait, cancel_futures=True)

    def _scheduler_loop(self) -> None:
        assert self.resource_pool is not None

        while True:
            with self._condition:
                if self._closed:
                    return

                if self._running_count_unlocked() >= self.max_running_jobs:
                    self._condition.wait(timeout=0.25)
                    continue

                started_any = False

                for job in list(self._pending):
                    if self._running_count_unlocked() >= self.max_running_jobs:
                        break

                    if job.cancel_requested or job.status == SltJobStatus.CANCELLED:
                        self._pending.remove(job)
                        job._set_status(SltJobStatus.CANCELLED)
                        job._safe_set_exception(CancelledError())
                        started_any = True
                        continue

                    allocation = self.resource_pool.try_acquire(
                        job.computation.resource_request
                    )

                    if allocation is None:
                        continue

                    self._pending.remove(job)
                    job._set_allocation(allocation)
                    job._set_status(SltJobStatus.RUNNING)

                    self._executor.submit(self._run_job, job, allocation)
                    started_any = True

                if not started_any:
                    self._condition.wait(timeout=0.25)

    def _running_count_unlocked(self) -> int:
        return sum(
            job.status in {SltJobStatus.RUNNING, SltJobStatus.CANCELLING}
            for job in self._jobs.values()
        )

    def _run_job(
        self,
        job: SltJob[Any],
        allocation: SltAllocation,
    ) -> None:
        assert self.resource_pool is not None

        try:
            if job.cancel_requested:
                raise CancelledError()

            progress = _computation_progress_snapshot(job.computation)
            job._set_last_progress(progress)

            result = job.computation.run_with_allocation(allocation)

            progress = _computation_progress_snapshot(job.computation)
            job._set_last_progress(progress)

            if job.cancel_requested:
                raise CancelledError()

            job._safe_set_result(result)
            job._set_status(SltJobStatus.FINISHED)

        except CancelledError as exc:
            job._set_last_exception(exc)
            job._safe_set_exception(exc)
            job._set_status(SltJobStatus.CANCELLED)

        except BaseException as exc:
            job._set_last_exception(exc)

            if job.cancel_requested:
                cancelled = CancelledError()
                job._safe_set_exception(cancelled)
                job._set_status(SltJobStatus.CANCELLED)
            else:
                job._safe_set_exception(exc)
                job._set_status(SltJobStatus.FAILED)

        finally:
            progress = _computation_progress_snapshot(job.computation)
            job._set_last_progress(progress)

            self.resource_pool.release(allocation)
            job._set_allocation(None)

            with self._condition:
                self._condition.notify_all()

    # ------------------------------------------------------------------
    # Rich dashboard
    # ------------------------------------------------------------------

    def to_rich(self, *, width: int | None = None) -> Panel:
        from slothpy.core.slt_dashboard import session_snapshot_to_rich

        return session_snapshot_to_rich(self.snapshot(), width=width)

    def print_rich(
        self,
        *,
        console: Console | None = None,
        width: int | None = None,
    ) -> None:
        """
        Print the Rich terminal dashboard (tree/tables).

        Use this in terminals; notebooks use HTML via ``_repr_html_`` / ``dashboard()``.
        """
        from slothpy.core.slt_common import print_rich_renderable

        print_rich_renderable(self.to_rich(width=width), console=console)

    def show(
        self,
        *,
        console: Console | None = None,
        width: int | None = None,
    ) -> None:
        """Alias for :meth:`print_rich`."""
        self.print_rich(console=console, width=width)

    def dashboard_html(self) -> str:
        from slothpy.core.slt_dashboard import session_snapshot_to_html

        return session_snapshot_to_html(self.snapshot())

    def dashboard(self, refresh: Any = None) -> str:
        """
        Return dashboard HTML.

        The optional ``refresh`` argument is intentionally unused. It is useful
        in marimo to create a dependency on ``mo.ui.refresh``:

            refresh = mo.ui.refresh(default_interval="500ms")
            mo.Html(session.dashboard(refresh))
        """
        _ = refresh
        return self.dashboard_html()

    def dashboard_text(self) -> str:
        from slothpy.core.slt_dashboard import session_snapshot_to_text

        return session_snapshot_to_text(self.snapshot())

    def run_dashboard(
        self,
        *,
        interval: float = 0.5,
        once: bool = False,
        exit_when_done: bool = False,
        jobs: Sequence[SltJob[Any]] | None = None,
        console: Console | None = None,
        width: int | None = None,
    ) -> None:
        """
        Print a live Rich dashboard to the terminal (blocking).

        Parameters
        ----------
        interval
            Seconds between refreshes. Ignored when ``once=True``.
        once
            Print a single frame and return (no :class:`rich.live.Live` loop).
        exit_when_done
            Stop the live loop once tracked jobs have finished. When ``jobs``
            is omitted, all jobs known to the session are tracked.
        jobs
            Jobs to watch for ``exit_when_done``. Defaults to all session jobs.
        console
            Rich console for output. Defaults to a new :class:`~rich.console.Console`.
        width
            Optional width passed to :meth:`to_rich`.
        """
        from slothpy.core.slt_dashboard import run_session_dashboard

        run_session_dashboard(
            self,
            interval=interval,
            once=once,
            exit_when_done=exit_when_done,
            jobs=jobs,
            console=console,
            width=width,
        )

    async def run_dashboard_async(
        self,
        *,
        interval: float = 0.5,
        once: bool = False,
        exit_when_done: bool = False,
        jobs: Sequence[SltJob[Any]] | None = None,
        console: Console | None = None,
        width: int | None = None,
    ) -> None:
        """
        Print a live Rich dashboard to the terminal (async).

        Same options as :meth:`run_dashboard`, but uses ``asyncio.sleep`` between
        refreshes so other coroutines can run concurrently.
        """

        from slothpy.core.slt_dashboard import run_session_dashboard_async

        await run_session_dashboard_async(
            self,
            interval=interval,
            once=once,
            exit_when_done=exit_when_done,
            jobs=jobs,
            console=console,
            width=width,
        )

    def _repr_html_(self) -> str:
        return self.dashboard_html()

    def __rich__(self) -> Panel:
        return self.to_rich()

    def __str__(self) -> str:
        return self.dashboard_text()

    def __enter__(self) -> SltSession:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.shutdown(
            cancel_running=exc_type is not None,
            hard_cancel=True,
            wait=True,
        )


__all__ = [
    "MPIProcessHandle",
    "SltAllocation",
    "SltJob",
    "SltJobSnapshot",
    "SltJobStatus",
    "SltNodeAllocation",
    "SltNodeResources",
    "SltProgressSnapshot",
    "SltProgressSpec",
    "SltProgressStatus",
    "SltProgressTracker",
    "SltResourceNodeSnapshot",
    "SltResourcePool",
    "SltResourceRequest",
    "SltRunnableComputation",
    "SltSession",
    "SltSessionSnapshot",
    "SltWorkerProgress",
    "build_mpi_environment",
    "start_mpi_process",
    "update_mpi_progress",
]
