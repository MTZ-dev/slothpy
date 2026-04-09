from __future__ import annotations

from typing import Annotated, Final

from pydantic import Field, TypeAdapter

from .primitive import PositiveInt

type NumProcesses = Annotated[
    PositiveInt,
    Field(description="Number of MPI processes to use."),
]

type NumThreads = Annotated[
    PositiveInt,
    Field(description="Number of threads per process to use."),
]

# type NumThreads = Annotated[
#     int,
#     BeforeValidator(_replace_zero_with_settings_num_threads),
#     Field(strict=True, gt=0, validate_default=True),
#     AfterValidator(_validate_num_threads),
# ]


# def _read_positive_env_int(*names: str) -> int | None:
#     for name in names:
#         raw = os.environ.get(name)
#         if raw is None:
#             continue
#         value = int(raw)
#         if value <= 0:
#             raise ValueError(f"{name} must be positive, got {raw!r}.")
#         return value
#     return None


# def _replace_zero_with_settings_num_threads(value: Any) -> Any:
#     if value == 0:
#         from slothpy.config.settings import settings
#         return settings.num_threads
#     return value


# def _replace_zero_with_settings_num_processes(value: Any) -> Any:
#     if value == 0:
#         from slothpy.config.settings import settings
#         return settings.num_processes
#     return value


# def _validate_num_threads(value: int) -> int:
#     cpus_on_node = _read_positive_env_int("SLURM_CPUS_ON_NODE")
#     if cpus_on_node is not None and value > cpus_on_node:
#         raise ValueError(
#             f"num_threads={value} exceeds SLURM_CPUS_ON_NODE={cpus_on_node}."
#         )
#     return value


# def _validate_num_processes(value: int) -> int:
#     max_processes = _read_positive_env_int("SLURM_NTASKS", "SLURM_NPROCS")
#     if max_processes is not None and value > max_processes:
#         raise ValueError(
#             f"num_processes={value} exceeds SLURM allocation={max_processes}."
#         )
#     return value


# type NumThreads = Annotated[
#     int,
#     BeforeValidator(_replace_zero_with_settings_num_threads),
#     Field(strict=True, gt=0),
#     AfterValidator(_validate_num_threads),
# ]

# type NumProcesses = Annotated[
#     int,
#     BeforeValidator(_replace_zero_with_settings_num_processes),
#     Field(strict=True, gt=0),
#     AfterValidator(_validate_num_processes),
# ]

NUM_PROCESSES_ADAPTER: Final[TypeAdapter[NumProcesses]] = TypeAdapter(NumProcesses)
NUM_THREADS_ADAPTER: Final[TypeAdapter[NumThreads]] = TypeAdapter(NumThreads)

# class Precision(StrEnum):
#     """Floating-point precision mode used by SlothPy calculations."""

#     SINGLE = "single"
#     DOUBLE = "double"

__all__ = [
    "NumProcesses",
    "NumThreads",
    "NUM_PROCESSES_ADAPTER",
    "NUM_THREADS_ADAPTER",
]
