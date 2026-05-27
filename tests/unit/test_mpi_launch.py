from __future__ import annotations

from slothpy.core.mpi_launch import (
    extra_mpi_args_contain_bind_to,
    resolve_mpi_launch_args,
)


def test_resolve_mpi_launch_args_prepends_bind_to_none() -> None:
    assert resolve_mpi_launch_args(()) == ("--bind-to", "none")


def test_resolve_mpi_launch_args_respects_existing_bind_to() -> None:
    existing = ("--hostfile", "hosts", "--bind-to", "core")
    assert resolve_mpi_launch_args(existing) == existing


def test_resolve_mpi_launch_args_skips_when_bind_to_disabled() -> None:
    assert resolve_mpi_launch_args((), bind_to=None) == ()


def test_extra_mpi_args_contain_bind_to() -> None:
    assert extra_mpi_args_contain_bind_to(("--bind-to", "none"))
    assert not extra_mpi_args_contain_bind_to(("--np", "4"))
