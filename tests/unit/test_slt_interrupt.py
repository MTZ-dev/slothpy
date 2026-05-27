from __future__ import annotations

from unittest.mock import MagicMock

from slothpy.core.slt_interrupt import (
    interrupt_session,
    register_session_interrupt_handlers,
    unregister_session_interrupt_handlers,
)


def test_interrupt_session_cancel_all_by_default() -> None:
    session = MagicMock()

    interrupt_session(session, hard=True, grace_seconds=3.0)

    session.cancel_all.assert_called_once_with(hard=True, grace_seconds=3.0)
    session.kill_all.assert_not_called()


def test_interrupt_session_force_kill() -> None:
    session = MagicMock()

    interrupt_session(session, force_kill=True)

    session.kill_all.assert_called_once_with()
    session.cancel_all.assert_not_called()


def test_register_unregister_signal_handlers_roundtrip() -> None:
    session = MagicMock()

    register_session_interrupt_handlers(session, grace_seconds=1.0)
    unregister_session_interrupt_handlers(session)


def test_kill_all_invokes_job_kill() -> None:
    from slothpy.core.slt_session import SltSession

    session = SltSession.local(
        cores=2,
        max_running_jobs=1,
        install_signal_handlers=False,
    )

    running_job = MagicMock()
    running_job.done.return_value = False
    finished_job = MagicMock()
    finished_job.done.return_value = True

    with session._condition:  # noqa: SLF001
        session._jobs = {"running": running_job, "finished": finished_job}  # noqa: SLF001

    try:
        session.kill_all()
    finally:
        session.shutdown(wait=False, cancel_running=False)

    running_job.kill.assert_called_once_with()
    finished_job.kill.assert_not_called()
