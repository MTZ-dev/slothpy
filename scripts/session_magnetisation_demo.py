#!/usr/bin/env python3
"""
Terminal demo for :class:`~slothpy.core.slt_session.SltSession` and magnetisation progress.

Mirrors ``notebooks/session_magnetisation_demo.py``: creates a demo Hamiltonian,
submits several slow placeholder magnetisation jobs, and prints a live Rich dashboard
to the terminal.

Example::

    uv run python scripts/session_magnetisation_demo.py

    uv run python scripts/session_magnetisation_demo.py --once

    uv run python scripts/session_magnetisation_demo.py --interval 0.25 --wait
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console

from slothpy.compute.magnetisation import (
    create_demo_hamiltonian_group,
    demo_magnetisation,
)
from slothpy.core.slt_session import SltJob, SltSession
from slothpy.groups.hamiltonian import SltHamiltonianGroup
from slothpy.specs.magnetisation import SltMagnetisationResult


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a local SltSession magnetisation demo with a terminal dashboard.",
    )
    parser.add_argument(
        "--slt-path",
        type=Path,
        default=Path("notebooks/demo_session.slt"),
        help="Path for the demo Hamiltonian .slt file.",
    )
    parser.add_argument(
        "--group-name",
        default="demo_hamiltonian",
        help="Hamiltonian group name inside the demo .slt file.",
    )
    parser.add_argument(
        "--cores",
        type=int,
        default=8,
        help="Logical cores exposed to the local resource pool.",
    )
    parser.add_argument(
        "--max-jobs",
        type=int,
        default=4,
        help="Maximum concurrently running session jobs.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.5,
        help="Dashboard refresh interval in seconds (ignored with --once).",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Print the dashboard once and exit without a live refresh loop.",
    )
    parser.add_argument(
        "--exit-when-done",
        action="store_true",
        help="Stop the live dashboard loop once every submitted job has finished.",
    )
    parser.add_argument(
        "--wait",
        action="store_true",
        help="After the dashboard loop, wait for job results and print a short summary.",
    )
    return parser.parse_args(argv)


def _submit_demo_jobs(
    session: SltSession,
    hamiltonian: SltHamiltonianGroup,
) -> list[SltJob[SltMagnetisationResult]]:
    jobs: list[SltJob[SltMagnetisationResult]] = []

    jobs.append(
        session.submit(
            demo_magnetisation(
                hamiltonian,
                n_fields=2,
                n_orientations=3,
                num_processes=2,
                num_threads=2,
                steps_per_task=12,
                sleep_seconds=0.12,
            ),
            job_id="mag-fast",
        )
    )
    jobs.append(
        session.submit(
            demo_magnetisation(
                hamiltonian,
                n_fields=3,
                n_orientations=4,
                num_processes=2,
                num_threads=2,
                steps_per_task=16,
                sleep_seconds=0.18,
            ),
            job_id="mag-medium",
        )
    )
    jobs.append(
        session.submit(
            demo_magnetisation(
                hamiltonian,
                n_fields=2,
                n_orientations=5,
                num_processes=1,
                num_threads=4,
                steps_per_task=20,
                sleep_seconds=0.25,
            ),
            job_id="mag-slow",
        )
    )

    return jobs


def _print_results(
    jobs: list[SltJob[SltMagnetisationResult]],
    *,
    console: Console,
) -> None:
    console.print("\n[bold]Results[/bold]")
    for job in jobs:
        if not job.done():
            console.print(f"[yellow]{job.job_id}[/yellow]: still running")
            continue

        try:
            view = job.result(timeout=0)
            mean_value = float(view.magnetisation.values.mean())
            console.print(
                f"[green]{job.job_id}[/green]: mean magnetisation = {mean_value:.6f}"
            )
        except Exception as exc:
            console.print(f"[red]{job.job_id}[/red]: {type(exc).__name__}: {exc}")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    if args.interval <= 0 and not args.once:
        print("--interval must be > 0 unless --once is set.", file=sys.stderr)
        return 2

    session = SltSession.local(cores=args.cores, max_running_jobs=args.max_jobs)
    console = Console()

    try:
        hamiltonian = create_demo_hamiltonian_group(
            args.slt_path,
            group_name=args.group_name,
            overwrite=True,
        )
        jobs = _submit_demo_jobs(session, hamiltonian)

        console.print(
            f"[bold]Submitted {len(jobs)} magnetisation jobs[/bold] "
            f"to {args.slt_path} ({args.group_name})."
        )
        for job in jobs:
            console.print(f"  - {job.job_id}: {job.name}")

        session.run_dashboard(
            console=console,
            once=args.once,
            interval=args.interval,
            exit_when_done=args.exit_when_done,
            jobs=jobs,
        )

        if args.wait:
            for job in jobs:
                if not job.done():
                    job.result()
            _print_results(jobs, console=console)

        return 0

    except KeyboardInterrupt:
        console.print("\n[bold yellow]Interrupted. Shutting down session.[/bold yellow]")
        from slothpy.core.slt_interrupt import interrupt_session

        interrupt_session(session, hard=True)
        return 130

    finally:
        session.shutdown(wait=True)


if __name__ == "__main__":
    raise SystemExit(main())
