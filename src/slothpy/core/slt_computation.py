from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from time import perf_counter
from traceback import format_exception
from typing import Any, ClassVar, Protocol, runtime_checkable

import xarray as xr
from pydantic import ConfigDict, validate_call

from slothpy.core.slt import (
    SltFile,
    create_slt_file,
    open_slt_file,
)
from slothpy.core.slt_group import SltGroup
from slothpy.core.slt_results import SltResults
from slothpy.types.aliases import PathLike

_VALIDATE_CONFIG = ConfigDict(arbitrary_types_allowed=True)


class SltComputationStatus(StrEnum):
    """
    Runtime status of a SlothPy computation object.
    """

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    FINISHED = "FINISHED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


@dataclass(frozen=True, slots=True)
class SltComputationResources:
    """
    Resource request attached to a computation.

    ``0`` means: resolve from SlothPy settings or from the active SltSession.
    The base class does not decide how MPI ranks or threads are mapped.
    """

    num_processes: int = 0
    num_threads: int = 0
    nodes: int = 1

    def __post_init__(self) -> None:
        if self.num_processes < 0:
            raise ValueError("num_processes must be >= 0.")
        if self.num_threads < 0:
            raise ValueError("num_threads must be >= 0.")
        if self.nodes < 1:
            raise ValueError("nodes must be >= 1.")


@dataclass(frozen=True, slots=True)
class SltComputationFailure:
    """
    Stored information about a failed computation.
    """

    exception_type: str
    message: str
    traceback: str


@runtime_checkable
class SltSessionProtocol(Protocol):
    """
    Minimal protocol expected from future SltSession objects.
    """

    def submit(self, computation: SltComputation[Any, Any]) -> Any: ...


@dataclass(slots=True)
class SltComputation[OptionsT](ABC):
    """
    Base class for SlothPy computations.

    A computation:

    1. holds input source(s), options, and resource request;
    2. computes an in-memory ``SltResults`` object;
    3. can optionally save that result as a SlothPy semantic group;
    4. can later be submitted to ``SltSession`` without changing the public API.

    Subclasses should usually implement only ``_compute`` and optionally
    ``_before_run`` / ``_after_success``.
    """

    source: Any
    options: OptionsT
    resources: SltComputationResources = field(default_factory=SltComputationResources)

    save_as: str | None = None
    output_slt: SltFile | PathLike | None = None
    overwrite: bool = False
    encoding: dict[str, Any] | None = None

    _status: SltComputationStatus = field(
        default=SltComputationStatus.PENDING,
        init=False,
    )
    _result: SltResults | None = field(default=None, init=False)
    _saved_group: SltGroup | None = field(default=None, init=False)
    _failure: SltComputationFailure | None = field(default=None, init=False)

    _progress: float | None = field(default=None, init=False)
    _message: str | None = field(default=None, init=False)
    _started_perf_counter: float | None = field(default=None, init=False)
    _finished_perf_counter: float | None = field(default=None, init=False)

    computation_name: ClassVar[str] = "SltComputation"

    @property
    def status(self) -> SltComputationStatus:
        return self._status

    @property
    def is_pending(self) -> bool:
        return self._status == SltComputationStatus.PENDING

    @property
    def is_running(self) -> bool:
        return self._status == SltComputationStatus.RUNNING

    @property
    def is_finished(self) -> bool:
        return self._status == SltComputationStatus.FINISHED

    @property
    def is_failed(self) -> bool:
        return self._status == SltComputationStatus.FAILED

    @property
    def is_cancelled(self) -> bool:
        return self._status == SltComputationStatus.CANCELLED

    @property
    def progress(self) -> float | None:
        """
        Progress in the inclusive range [0.0, 1.0], if known.
        """
        return self._progress

    @property
    def message(self) -> str | None:
        """
        Human-readable status message, if available.
        """
        return self._message

    @property
    def failure(self) -> SltComputationFailure | None:
        """
        Failure information after a failed run.
        """
        return self._failure

    @property
    def saved_group(self) -> SltGroup | None:
        """
        Saved group handle, if this computation has been persisted.
        """
        return self._saved_group

    @property
    def elapsed_seconds(self) -> float | None:
        """
        Runtime duration in seconds, if the computation has started.
        """
        if self._started_perf_counter is None:
            return None

        end = (
            perf_counter()
            if self._finished_perf_counter is None
            else self._finished_perf_counter
        )
        return end - self._started_perf_counter

    @property
    def result(self) -> SltResults:
        """
        Return computed in-memory results.

        Raises
        ------
        RuntimeError
            If the computation has not finished successfully.
        """
        if self._result is None:
            raise RuntimeError(
                f"{self.computation_name} has no result yet. "
                "Call run() first or submit it to a session."
            )
        return self._result

    @validate_call(config=_VALIDATE_CONFIG)
    def run(self, *, force: bool = False, save: bool | None = None) -> SltResults:
        """
        Execute the computation synchronously and return in-memory results.

        Parameters
        ----------
        force
            Re-run even if a previous result exists.
        save
            If ``True``, save after computing. If ``None``, save automatically
            when ``save_as`` was provided.
        """
        if self._status == SltComputationStatus.RUNNING:
            raise RuntimeError(f"{self.computation_name} is already running.")

        if self._status == SltComputationStatus.FINISHED and not force:
            result = self.result
            if save is True and self._saved_group is None:
                self.save()
            return result

        if self._status == SltComputationStatus.CANCELLED and not force:
            raise RuntimeError(
                f"{self.computation_name} was cancelled. "
                "Use run(force=True) to run it anyway."
            )

        self._reset_runtime_state()
        self._status = SltComputationStatus.RUNNING
        self._started_perf_counter = perf_counter()

        try:
            self._set_progress(0.0, "Starting computation.")
            self._before_run()

            result = self._compute()
            self._validate_result(result)

            self._result = result
            self._status = SltComputationStatus.FINISHED
            self._set_progress(1.0, "Computation finished.")

            self._after_success(result)

            should_save = self.save_as is not None if save is None else save
            if should_save:
                self.save()

            return result

        except BaseException as exc:
            self._status = SltComputationStatus.FAILED
            self._failure = SltComputationFailure(
                exception_type=type(exc).__name__,
                message=str(exc),
                traceback="".join(format_exception(type(exc), exc, exc.__traceback__)),
            )
            self._set_progress(None, "Computation failed.")
            raise

        finally:
            self._finished_perf_counter = perf_counter()

    @validate_call(config=_VALIDATE_CONFIG)
    def save(
        self,
        slt: SltFile | PathLike | None = None,
        group_name: str | None = None,
        *,
        overwrite: bool | None = None,
        encoding: dict[str, Any] | None = None,
    ) -> SltGroup:
        """
        Save computed results as a SlothPy semantic group.

        If the computation has not been run yet, it is run first with
        ``save=False`` to avoid recursive saving.
        """
        if self._result is None:
            self.run(save=False)

        results = self.result
        target_slt = self._resolve_output_slt(slt)

        final_group_name = group_name or self.save_as
        if final_group_name is None:
            raise ValueError(
                "No output group name was provided. Pass group_name=... "
                "or construct the computation with save_as=..."
            )

        final_overwrite = self.overwrite if overwrite is None else overwrite
        final_encoding = self.encoding if encoding is None else encoding

        group = target_slt._write_slothpy_group(
            final_group_name,
            results,
            overwrite=final_overwrite,
            encoding=final_encoding,
        )

        self._saved_group = group
        return group

    @validate_call(config=_VALIDATE_CONFIG)
    def submit(self, session: SltSessionProtocol) -> Any:
        """
        Submit this computation to a future SltSession-like scheduler.
        """
        return session.submit(self)

    def cancel(self) -> None:
        """
        Cancel a computation that has not started yet.

        Running MPI/process cancellation will be implemented by SltSession later.
        """
        if self._status == SltComputationStatus.RUNNING:
            raise RuntimeError(
                "Cannot cancel a running computation directly. "
                "Cancel it through the SltSession that owns the worker."
            )

        if self._status == SltComputationStatus.FINISHED:
            raise RuntimeError("Cannot cancel a finished computation.")

        self._status = SltComputationStatus.CANCELLED
        self._set_progress(None, "Computation cancelled.")

    def to_dataset(self, *, copy: bool = False) -> xr.Dataset:
        """
        Return the computed xarray Dataset.
        """
        dataset = self.result.dataset
        return dataset.copy(deep=True) if copy else dataset

    def to_xarray(self) -> xr.Dataset | xr.DataArray:
        """
        Return the primary xarray object from the computed result.
        """
        results = self.result
        dataset = results.dataset
        primary = results.primary

        if primary is None or primary == "__dataset__":
            return dataset

        if primary in dataset.data_vars:
            return dataset[primary]

        if primary in dataset.coords:
            return dataset.coords[primary]

        raise KeyError(
            f"Computed result declares primary={primary!r}, "
            "but this variable or coordinate is missing from the dataset."
        )

    def _reset_runtime_state(self) -> None:
        self._result = None
        self._saved_group = None
        self._failure = None
        self._progress = None
        self._message = None
        self._started_perf_counter = None
        self._finished_perf_counter = None

    def _set_progress(self, progress: float | None, message: str | None = None) -> None:
        if progress is not None and not 0.0 <= progress <= 1.0:
            raise ValueError("progress must be between 0.0 and 1.0.")

        self._progress = progress
        self._message = message

    def _before_run(self) -> None:
        """
        Optional subclass hook before computation starts.
        """
        return None

    def _after_success(self, result: SltResults) -> None:
        """
        Optional subclass hook after successful computation.
        """
        return None

    @abstractmethod
    def _compute(self) -> SltResults:
        """
        Perform the actual computation and return SltResults.

        Subclasses decide whether this is pure NumPy/Numba, MPI-backed,
        stream-based, or delegated to a worker process.
        """

    def _resolve_output_slt(self, slt: SltFile | PathLike | None) -> SltFile:
        if slt is not None:
            return self._coerce_slt_file(slt)

        if self.output_slt is not None:
            return self._coerce_slt_file(self.output_slt)

        inferred = self._infer_output_slt_from_source()
        if inferred is not None:
            return inferred

        raise ValueError(
            "Could not infer output .slt file. Pass output_slt=... when creating "
            "the computation or pass slt=... to save()."
        )

    @staticmethod
    def _coerce_slt_file(value: SltFile | PathLike) -> SltFile:
        if isinstance(value, SltFile):
            return value

        try:
            return open_slt_file(value)
        except FileNotFoundError:
            return create_slt_file(value)

    def _infer_output_slt_from_source(self) -> SltFile | None:
        file_paths = self._source_file_paths(self.source)

        if not file_paths:
            return None

        if len(file_paths) > 1:
            raise ValueError(
                "Computation uses groups from multiple .slt files. "
                "Pass output_slt=... explicitly."
            )

        return open_slt_file(next(iter(file_paths)))

    @classmethod
    def _source_file_paths(cls, source: Any) -> set[Path]:
        paths: set[Path] = set()

        if isinstance(source, SltGroup):
            paths.add(source.file_path)
            return paths

        if isinstance(source, Mapping):
            for value in source.values():
                paths.update(cls._source_file_paths(value))
            return paths

        if isinstance(source, Sequence) and not isinstance(source, str | bytes):
            for value in source:
                paths.update(cls._source_file_paths(value))
            return paths

        return paths

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"status={self.status.value!r}, "
            f"save_as={self.save_as!r}, "
            f"resources={self.resources!r})"
        )


__all__ = [
    "SltComputation",
    "SltComputationFailure",
    "SltComputationResources",
    "SltComputationStatus",
    "SltSessionProtocol",
]
