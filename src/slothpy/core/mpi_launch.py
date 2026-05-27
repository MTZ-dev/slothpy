"""
OpenMPI launch helpers for SlothPy MPI workers.

SlothPy does not rely on the MPI launcher to pin ranks to CPU subsets. Ranks are
started with ``--bind-to none`` so worker processes see the node's logical CPUs;
:class:`~slothpy.core.slt_session.SltResourcePool` tracks how many cores are
reserved per job and ``build_mpi_environment`` sets per-rank thread caps
(``OMP_NUM_THREADS``, ``NUMBA_NUM_THREADS``, ...).
"""

from __future__ import annotations

from collections.abc import Sequence


def openmpi_bind_extra_args(bind_to: str | None = "none") -> tuple[str, ...]:
    """
    Return OpenMPI ``--bind-to`` arguments.

    Pass ``bind_to=None`` to omit binding flags (for launchers that do not support
    OpenMPI syntax).
    """
    if bind_to is None:
        return ()
    return ("--bind-to", bind_to)


def extra_mpi_args_contain_bind_to(extra_mpi_args: Sequence[str]) -> bool:
    for index, argument in enumerate(extra_mpi_args):
        if argument == "--bind-to":
            return True
        if argument.startswith("--bind-to="):
            return True
        if argument in {"-bind-to", "-bind-to-none", "-bind-to-core"}:
            return True
        if argument.startswith("-bind-to"):
            return True
    return False


def resolve_mpi_launch_args(
    extra_mpi_args: Sequence[str] = (),
    *,
    bind_to: str | None = "none",
) -> tuple[str, ...]:
    """
    Prepend default OpenMPI binding flags unless the caller already set ``--bind-to``.
    """
    if bind_to is None or extra_mpi_args_contain_bind_to(extra_mpi_args):
        return tuple(extra_mpi_args)
    return openmpi_bind_extra_args(bind_to) + tuple(extra_mpi_args)


__all__ = [
    "extra_mpi_args_contain_bind_to",
    "openmpi_bind_extra_args",
    "resolve_mpi_launch_args",
]
