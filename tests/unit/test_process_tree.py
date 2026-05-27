from __future__ import annotations

import os
import signal
import subprocess
import sys
import time

import pytest

from slothpy.core.process_tree import (
    iter_descendant_pids,
    signal_process_tree,
    terminate_process_tree,
)


@pytest.mark.skipif(os.name != "posix", reason="POSIX process-tree helpers only.")
def test_iter_descendant_pids_finds_child() -> None:
    script = (
        "import os, signal, time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "pid = os.fork()\n"
        "if pid == 0:\n"
        "    signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "    time.sleep(30)\n"
        "    raise SystemExit(0)\n"
        "time.sleep(30)\n"
    )
    parent = subprocess.Popen([sys.executable, "-c", script], start_new_session=True)
    try:
        deadline = time.monotonic() + 5.0
        descendants: tuple[int, ...] = ()
        while time.monotonic() < deadline:
            descendants = iter_descendant_pids(parent.pid)
            if descendants:
                break
            time.sleep(0.05)

        assert descendants
        assert all(descendant > 1 for descendant in descendants)
    finally:
        terminate_process_tree(parent.pid, grace_seconds=0.2)
        parent.wait(timeout=5)


@pytest.mark.skipif(os.name != "posix", reason="POSIX process-tree helpers only.")
def test_signal_process_tree_kills_descendants() -> None:
    script = (
        "import os, signal, time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "pid = os.fork()\n"
        "if pid == 0:\n"
        "    time.sleep(30)\n"
        "    raise SystemExit(0)\n"
        "time.sleep(30)\n"
    )
    parent = subprocess.Popen([sys.executable, "-c", script], start_new_session=True)
    try:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if iter_descendant_pids(parent.pid):
                break
            time.sleep(0.05)

        signal_process_tree(parent.pid, signal.SIGKILL)
        assert parent.wait(timeout=5) != 0
    finally:
        if parent.poll() is None:
            os.kill(parent.pid, signal.SIGKILL)
            parent.wait(timeout=5)
