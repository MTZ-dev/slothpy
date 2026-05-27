"""
POSIX helpers to terminate MPI / subprocess trees reliably.

OpenMPI and similar launchers often leave worker processes outside the initial
process group. Walking ``/proc`` descendants and signalling deepest children
first makes shutdown much more reliable than ``killpg`` alone.
"""

from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path


def _read_child_pids(pid: int) -> list[int]:
    if os.name != "posix":
        return []

    children_path = Path(f"/proc/{pid}/task/{pid}/children")
    if not children_path.is_file():
        return []

    text = children_path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    return [int(token) for token in text.split()]


def iter_descendant_pids(root_pid: int) -> tuple[int, ...]:
    """
  Return descendant PIDs in depth-first order (parents before children).
    """
    if root_pid < 1:
        return ()

    descendants: list[int] = []
    stack = list(reversed(_read_child_pids(root_pid)))

    while stack:
        pid = stack.pop()
        descendants.append(pid)
        stack.extend(reversed(_read_child_pids(pid)))

    return tuple(descendants)


def signal_process_tree(
    root_pid: int,
    sig: signal.Signals,
    *,
    include_root: bool = True,
) -> None:
    """
    Deliver ``sig`` to ``root_pid`` and its descendants.

    Children are signalled deepest-first so leaves stop before parents.
    """
    if root_pid < 1:
        return

    descendants = iter_descendant_pids(root_pid)
    for pid in reversed(descendants):
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            continue

    if not include_root:
        return

    if os.name == "posix":
        try:
            os.killpg(root_pid, sig)
            return
        except ProcessLookupError:
            pass

    try:
        os.kill(root_pid, sig)
    except ProcessLookupError:
        return


def terminate_process_tree(
    root_pid: int,
    *,
    grace_seconds: float = 5.0,
    poll_seconds: float = 0.05,
) -> None:
    """
    SIGTERM the tree, wait up to ``grace_seconds``, then SIGKILL survivors.
    """
    if root_pid < 1:
        return

    signal_process_tree(root_pid, signal.SIGTERM)

    deadline = time.monotonic() + max(grace_seconds, 0.0)
    while time.monotonic() < deadline:
        try:
            os.kill(root_pid, 0)
        except ProcessLookupError:
            return
        time.sleep(poll_seconds)

    signal_process_tree(root_pid, signal.SIGKILL)


def terminate_subprocess(
    process: subprocess.Popen[object],
    *,
    grace_seconds: float = 5.0,
) -> None:
    """
    Terminate a :class:`subprocess.Popen` and its descendants.
    """
    if process.poll() is not None:
        return

    terminate_process_tree(process.pid, grace_seconds=grace_seconds)

    try:
        process.wait(timeout=max(grace_seconds, 0.0))
    except subprocess.TimeoutExpired:
        signal_process_tree(process.pid, signal.SIGKILL)
        try:
            process.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            pass


__all__ = [
    "iter_descendant_pids",
    "signal_process_tree",
    "terminate_process_tree",
    "terminate_subprocess",
]
