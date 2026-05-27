"""
Runtime bookkeeping for in-flight MPI autotune benchmark trials.

Allows :class:`~slothpy.core.slt_session.SltSession` shutdown and notebook
interrupts to terminate stray ``mpirun`` processes and release parent-owned
shared memory even when a trial is aborted mid-loop.
"""

from __future__ import annotations

import subprocess
import threading
from collections.abc import Callable
from dataclasses import dataclass, field

_RELEASE_CALLBACK = Callable[[], None]


@dataclass(slots=True)
class _ActiveAutotuneTrial:
    process: subprocess.Popen[bytes] | None = None
    release_callbacks: list[_RELEASE_CALLBACK] = field(default_factory=list)

    def register_release(self, callback: _RELEASE_CALLBACK) -> None:
        self.release_callbacks.append(callback)

    def abort(self) -> None:
        process = self.process
        self.process = None
        if process is not None and process.poll() is None:
            from slothpy.core.process_tree import terminate_subprocess

            terminate_subprocess(process, grace_seconds=0.0)

        for callback in reversed(self.release_callbacks):
            try:
                callback()
            except Exception:
                pass
        self.release_callbacks.clear()


_LOCK = threading.Lock()
_ACTIVE_TRIAL: _ActiveAutotuneTrial | None = None


def begin_autotune_trial() -> _ActiveAutotuneTrial:
    """Mark the start of one MPI benchmark trial (not thread-safe for overlap)."""
    global _ACTIVE_TRIAL

    trial = _ActiveAutotuneTrial()
    with _LOCK:
        if _ACTIVE_TRIAL is not None:
            _ACTIVE_TRIAL.abort()
        _ACTIVE_TRIAL = trial
    return trial


def bind_autotune_process(process: subprocess.Popen[bytes]) -> None:
    with _LOCK:
        if _ACTIVE_TRIAL is not None:
            _ACTIVE_TRIAL.process = process


def end_autotune_trial(trial: _ActiveAutotuneTrial) -> None:
    """Clear the active trial without aborting (normal completion)."""
    global _ACTIVE_TRIAL

    with _LOCK:
        if _ACTIVE_TRIAL is trial:
            _ACTIVE_TRIAL = None


def cancel_active_autotune_trials() -> None:
    """Terminate the current benchmark subprocess and release its shared memory."""
    global _ACTIVE_TRIAL

    with _LOCK:
        trial = _ACTIVE_TRIAL
        _ACTIVE_TRIAL = None

    if trial is not None:
        trial.abort()


def abort_compute_runtime(
    *,
    session: object | None = None,
    cancel_session_jobs: bool = True,
) -> None:
    """
    Stop autotune benchmarks and optionally hard-cancel a session's MPI jobs.

    Intended for ``KeyboardInterrupt``, marimo cell stop, and process exit.
    """
    cancel_active_autotune_trials()

    if cancel_session_jobs and session is not None:
        shutdown = getattr(session, "shutdown", None)
        if callable(shutdown) and not getattr(session, "_closed", True):
            try:
                shutdown(
                    cancel_running=True,
                    hard_cancel=True,
                    grace_seconds=0.0,
                    wait=False,
                    cleanup_timeout=0.0,
                )
            except Exception:
                kill_all = getattr(session, "kill_all", None)
                if callable(kill_all):
                    try:
                        kill_all()
                    except Exception:
                        pass

    from slothpy.io.shared_memory import release_all_parent_owned_shared_blocks

    release_all_parent_owned_shared_blocks()


__all__ = [
    "abort_compute_runtime",
    "begin_autotune_trial",
    "bind_autotune_process",
    "cancel_active_autotune_trials",
    "end_autotune_trial",
]
