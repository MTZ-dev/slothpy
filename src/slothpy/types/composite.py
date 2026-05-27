from __future__ import annotations

import os
from enum import StrEnum
from typing import Annotated, Any, Final

from pydantic import AfterValidator, BeforeValidator, Field, TypeAdapter

from slothpy.types.primitive import NonNegativeInt, PositiveInt

# ---------------------------------------------------------------------------
# Composite type helpers
# ---------------------------------------------------------------------------


def _current_process_cpu_count() -> int:
    process_cpu_count = getattr(os, "process_cpu_count", None)
    if process_cpu_count is not None:
        value = process_cpu_count()
        if value is not None:
            return value

    value = os.cpu_count()
    return 1 if value is None else value


def _replace_zero_with_settings_num_threads(value: Any) -> Any:
    if value == 0:
        from slothpy.config.settings import settings

        return settings.num_threads
    return value


def _replace_zero_with_settings_num_processes(value: Any) -> Any:
    if value == 0:
        from slothpy.config.settings import settings

        return settings.num_processes
    return value


def _validate_num_threads(value: int) -> int:
    process_cpu_count = _current_process_cpu_count()
    if value > process_cpu_count:
        raise ValueError(
            f"num_threads={value} exceeds process_cpu_count={process_cpu_count}."
        )
    return value


# ---------------------------------------------------------------------------
# Composite types
# ---------------------------------------------------------------------------


class AutotuneDisplay(StrEnum):
    """How :meth:`~slothpy.core.slt_computation.SltComputation.autotune` reports progress."""

    PRINT = "print"
    RICH = "rich"
    HTML = "html"
    NONE = "none"


AUTOTUNE_DISPLAY_ADAPTER: Final[TypeAdapter[AutotuneDisplay]] = TypeAdapter(
    AutotuneDisplay
)


type NumProcesses = Annotated[
    PositiveInt,
    BeforeValidator(_replace_zero_with_settings_num_processes),
    Field(description="Number of MPI processes to use."),
]

type NumThreads = Annotated[
    PositiveInt,
    BeforeValidator(_replace_zero_with_settings_num_threads),
    Field(description="Number of threads per process to use."),
    AfterValidator(_validate_num_threads),
]

NUM_PROCESSES_ADAPTER: Final[TypeAdapter[NumProcesses]] = TypeAdapter(NumProcesses)
NUM_THREADS_ADAPTER: Final[TypeAdapter[NumThreads]] = TypeAdapter(NumThreads)


# class Precision(StrEnum):
#     """Floating-point precision mode used by SlothPy calculations."""

#     SINGLE = "single"
#     DOUBLE = "double"

__all__ = [
    "AUTOTUNE_DISPLAY_ADAPTER",
    "AutotuneDisplay",
    "NumProcesses",
    "NumThreads",
    "NUM_PROCESSES_ADAPTER",
    "NUM_THREADS_ADAPTER",
]
