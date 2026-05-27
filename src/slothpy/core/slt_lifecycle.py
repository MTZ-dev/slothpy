"""
Process-wide cleanup for active :class:`~slothpy.core.slt_session.SltSession` instances.

Registers ``atexit`` and ``weakref.finalize`` hooks so notebooks and REPLs that
exit without an explicit ``shutdown()`` still cancel MPI workers and release
parent-owned shared memory.
"""

from __future__ import annotations

import atexit
import threading
import weakref
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from slothpy.core.slt_session import SltSession

_ACTIVE_SESSIONS: dict[int, weakref.ReferenceType[Any]] = {}
_ATEXIT_REGISTERED = False
_LOCK = threading.Lock()


def register_session_lifecycle(session: SltSession) -> None:
    """Track ``session`` for automatic shutdown on interpreter exit."""
    global _ATEXIT_REGISTERED

    session_id = id(session)
    with _LOCK:
        _ACTIVE_SESSIONS[session_id] = weakref.ref(session)
        if not _ATEXIT_REGISTERED:
            atexit.register(_shutdown_all_sessions)
            _ATEXIT_REGISTERED = True

    weakref.finalize(session, _finalize_session, session_id, weakref.ref(session))


def unregister_session_lifecycle(session: SltSession) -> None:
    """Stop tracking ``session`` after an explicit :meth:`~SltSession.shutdown`."""
    with _LOCK:
        _ACTIVE_SESSIONS.pop(id(session), None)


def _finalize_session(
    session_id: int,
    session_ref: weakref.ReferenceType[SltSession],
) -> None:
    with _LOCK:
        _ACTIVE_SESSIONS.pop(session_id, None)

    session = session_ref()
    if session is None:
        return

    _shutdown_session(session, wait=True, cleanup_timeout=5.0)


def _shutdown_all_sessions() -> None:
    with _LOCK:
        session_refs = list(_ACTIVE_SESSIONS.values())
        _ACTIVE_SESSIONS.clear()

    for session_ref in session_refs:
        session = session_ref()
        if session is not None:
            _shutdown_session(session, wait=True, cleanup_timeout=10.0)


def _shutdown_session(
    session: SltSession,
    *,
    wait: bool,
    cleanup_timeout: float = 10.0,
) -> None:
    if getattr(session, "_closed", True):
        return

    from slothpy.compute.autotune_runtime import cancel_active_autotune_trials

    cancel_active_autotune_trials()

    try:
        session.shutdown(
            cancel_running=True,
            hard_cancel=True,
            grace_seconds=2.0,
            wait=wait,
            cleanup_timeout=cleanup_timeout,
        )
    except Exception:
        try:
            session.kill_all()
        except Exception:
            pass

    from slothpy.io.shared_memory import release_all_parent_owned_shared_blocks

    release_all_parent_owned_shared_blocks()


__all__ = ["register_session_lifecycle", "unregister_session_lifecycle"]
