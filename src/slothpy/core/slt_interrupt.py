"""
Signal-driven interruption for :class:`~slothpy.core.slt_session.SltSession`.

When a session registers handlers (default), SIGINT (Ctrl+C) and SIGTERM propagate to
all session jobs: first interrupt requests hard cancellation (SIGTERM to the MPI
process group), a second interrupt within a short window SIGKILLs survivors and
re-raises :exc:`KeyboardInterrupt`.

``SltSession`` uses ``slots=True``, so registrations keep a strong reference keyed
by ``id(session)`` until :func:`unregister_session_interrupt_handlers` runs from
:meth:`~slothpy.core.slt_session.SltSession.shutdown`.
"""

from __future__ import annotations

import signal
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from slothpy.core.slt_session import SltSession

_INTERRUPT_LOCK = threading.Lock()
_SESSION_OPTIONS: dict[int, _SessionInterruptRegistration] = {}
_ORIGINAL_HANDLERS: dict[int, Any] = {}
_HANDLERS_INSTALLED = False
_INTERRUPT_PRESS_COUNT = 0
_LAST_INTERRUPT_MONOTONIC = 0.0

# Second Ctrl+C within this window triggers SIGKILL instead of SIGTERM.
_SECOND_INTERRUPT_WINDOW_SECONDS = 2.0

_SIGNALS: tuple[tuple[int, str], ...] = ((signal.SIGINT, "SIGINT"),)
if hasattr(signal, "SIGTERM"):
    _SIGNALS = _SIGNALS + ((signal.SIGTERM, "SIGTERM"),)


@dataclass(frozen=True, slots=True)
class _SessionInterruptRegistration:
    session: Any
    grace_seconds: float
    raise_keyboard_interrupt: bool


def register_session_interrupt_handlers(
    session: SltSession,
    *,
    grace_seconds: float = 5.0,
    raise_keyboard_interrupt: bool = True,
) -> None:
    """
    Install process-wide SIGINT/SIGTERM handlers for ``session`` (reference-counted).

    Call :func:`unregister_session_interrupt_handlers` from
    :meth:`~slothpy.core.slt_session.SltSession.shutdown`.
    """
    if grace_seconds < 0:
        raise ValueError("grace_seconds must be >= 0.")

    with _INTERRUPT_LOCK:
        _SESSION_OPTIONS[id(session)] = _SessionInterruptRegistration(
            session=session,
            grace_seconds=grace_seconds,
            raise_keyboard_interrupt=raise_keyboard_interrupt,
        )
        _install_handlers_locked()


def unregister_session_interrupt_handlers(session: SltSession) -> None:
    """Remove ``session`` from the interrupt registry and restore default handlers."""
    global _INTERRUPT_PRESS_COUNT

    with _INTERRUPT_LOCK:
        _SESSION_OPTIONS.pop(id(session), None)
        if not _SESSION_OPTIONS:
            _INTERRUPT_PRESS_COUNT = 0
            _restore_handlers_locked()


def interrupt_session(
    session: SltSession,
    *,
    hard: bool = True,
    grace_seconds: float | None = None,
    force_kill: bool = False,
) -> None:
    """
    Cancel or kill all non-finished jobs on ``session``.

    Safe to call from ``except KeyboardInterrupt`` blocks and from signal handlers.
    """
    if force_kill:
        session.kill_all()
        return

    registration = _SESSION_OPTIONS.get(id(session))
    resolved_grace = (
        grace_seconds
        if grace_seconds is not None
        else (registration.grace_seconds if registration is not None else 5.0)
    )
    session.cancel_all(hard=hard, grace_seconds=resolved_grace)


def _install_handlers_locked() -> None:
    global _HANDLERS_INSTALLED

    if _HANDLERS_INSTALLED:
        return

    for sig, _name in _SIGNALS:
        try:
            _ORIGINAL_HANDLERS[sig] = signal.getsignal(sig)
            signal.signal(sig, _handle_signal)
        except (ValueError, OSError):
            # Not on main thread or unsupported platform.
            pass

    _HANDLERS_INSTALLED = True


def _restore_handlers_locked() -> None:
    global _HANDLERS_INSTALLED

    if not _HANDLERS_INSTALLED:
        return

    for sig, _name in _SIGNALS:
        original = _ORIGINAL_HANDLERS.pop(sig, None)
        if original is None:
            continue
        try:
            signal.signal(sig, original)
        except (ValueError, OSError):
            pass

    _HANDLERS_INSTALLED = False


def _active_sessions_locked() -> list[Any]:
    return [registration.session for registration in _SESSION_OPTIONS.values()]


def _handle_signal(signum: int, frame: object | None) -> None:
    del frame

    global _INTERRUPT_PRESS_COUNT, _LAST_INTERRUPT_MONOTONIC

    now = time.monotonic()
    with _INTERRUPT_LOCK:
        sessions = _active_sessions_locked()
        if not sessions:
            _restore_handlers_locked()
            _raise_for_signal(signum)
            return

        if (
            _INTERRUPT_PRESS_COUNT > 0
            and now - _LAST_INTERRUPT_MONOTONIC <= _SECOND_INTERRUPT_WINDOW_SECONDS
        ):
            _INTERRUPT_PRESS_COUNT += 1
            force_kill = True
            grace_seconds = 0.0
        else:
            _INTERRUPT_PRESS_COUNT = 1
            force_kill = False
            grace_seconds = min(
                (
                    registration.grace_seconds
                    for registration in _SESSION_OPTIONS.values()
                ),
                default=5.0,
            )

        _LAST_INTERRUPT_MONOTONIC = now
        raise_keyboard_interrupt = any(
            registration.raise_keyboard_interrupt
            for registration in _SESSION_OPTIONS.values()
        )

    for session in sessions:
        interrupt_session(
            session,
            hard=True,
            grace_seconds=grace_seconds,
            force_kill=force_kill,
        )

    if force_kill:
        with _INTERRUPT_LOCK:
            _restore_handlers_locked()

    if raise_keyboard_interrupt and signum == signal.SIGINT:
        raise KeyboardInterrupt

    if force_kill:
        _raise_for_signal(signum)


def _raise_for_signal(signum: int) -> None:
    if signum == signal.SIGINT:
        raise KeyboardInterrupt
    raise SystemExit(128 + signum)


__all__ = [
    "interrupt_session",
    "register_session_interrupt_handlers",
    "unregister_session_interrupt_handlers",
]
