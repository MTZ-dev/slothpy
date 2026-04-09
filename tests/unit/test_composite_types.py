from __future__ import annotations

from slothpy.types.composite import NUM_PROCESSES_ADAPTER, NUM_THREADS_ADAPTER


def test_num_processes_adapter():
    assert NUM_PROCESSES_ADAPTER.validate_python(4) == 4


def test_num_threads_adapter():
    assert NUM_THREADS_ADAPTER.validate_python(8) == 8
