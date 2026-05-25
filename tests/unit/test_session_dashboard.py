from __future__ import annotations

import asyncio

import pytest

from slothpy.core.slt_session import SltSession


def test_run_dashboard_once() -> None:
    session = SltSession.local(cores=2, max_running_jobs=1)
    try:
        session.run_dashboard(once=True)
    finally:
        session.shutdown(wait=True)


def test_run_dashboard_rejects_non_positive_interval() -> None:
    session = SltSession.local(cores=2, max_running_jobs=1)
    try:
        with pytest.raises(ValueError, match="interval must be > 0"):
            session.run_dashboard(interval=0)
    finally:
        session.shutdown(wait=True)


def test_run_dashboard_async_once() -> None:
    session = SltSession.local(cores=2, max_running_jobs=1)

    async def _run() -> None:
        await session.run_dashboard_async(once=True)

    try:
        asyncio.run(_run())
    finally:
        session.shutdown(wait=True)
