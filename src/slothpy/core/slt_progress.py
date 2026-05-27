"""
Shared-memory progress tracking for SlothPy sessions and MPI workers.

Parent processes own :class:`SltProgressTracker` blocks; MPI rank 0 attaches via
:class:`SltWorkerProgress`. Per-thread boards and MPI aggregation live in
:mod:`slothpy.compute.mpi_progress`.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from enum import IntEnum
from multiprocessing.shared_memory import SharedMemory
from typing import Any

import numpy as np

from slothpy.io.shared_memory import (
    _open_shared_memory,
    register_parent_owned_shared_block,
    unregister_parent_owned_shared_block,
)

_PROGRESS_SIZE = 16

_PROGRESS_DONE = 0
_PROGRESS_TOTAL = 1
_PROGRESS_STATUS = 2
_PROGRESS_CANCEL_REQUESTED = 3
_PROGRESS_HEARTBEAT_NS = 4
_PROGRESS_ERROR_CODE = 5
_PROGRESS_RESERVED_0 = 6
_PROGRESS_RESERVED_1 = 7


class SltProgressStatus(IntEnum):
    UNKNOWN = 0
    QUEUED = 1
    RUNNING = 2
    CANCELLING = 3
    CANCELLED = 4
    FINISHED = 5
    FAILED = 6


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
        track: bool = False,
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

        tracker = cls(
            spec=SltProgressSpec(name=shm.name),
            _shm=shm,
            _array=array,
            owns_memory=True,
        )
        register_parent_owned_shared_block(shm.name, tracker.release)
        return tracker

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
        if self.owns_memory:
            unregister_parent_owned_shared_block(self.spec.name)
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
    Worker-side helper for MPI progress aggregation (scalar done count).

    All ranks call this. ``local_done`` is reduced to rank 0. Rank 0 writes
    the aggregate into the parent-visible shared progress block.

    For per-thread boards use :func:`slothpy.compute.mpi_progress.publish_mpi_thread_progress`.

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


__all__ = [
    "SltProgressSnapshot",
    "SltProgressSpec",
    "SltProgressStatus",
    "SltProgressTracker",
    "SltWorkerProgress",
    "update_mpi_progress",
]
