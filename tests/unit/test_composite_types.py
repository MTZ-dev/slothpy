from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

import slothpy.config.settings as settings_mod
import slothpy.types.composite as composite_mod


@pytest.fixture
def fake_settings(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """
    Replace the global settings singleton with a simple fake object.

    This is enough for composite validators because they only read:
    - settings.num_processes
    - settings.num_threads
    """
    fake = SimpleNamespace(num_processes=3, num_threads=6)
    monkeypatch.setattr(settings_mod, "settings", fake)
    return fake


def test_num_processes_adapter_accepts_positive_int() -> None:
    assert composite_mod.NUM_PROCESSES_ADAPTER.validate_python(4) == 4


def test_num_processes_adapter_replaces_zero_with_settings_value(
    fake_settings: SimpleNamespace,
) -> None:
    value = composite_mod.NUM_PROCESSES_ADAPTER.validate_python(0)
    assert value == fake_settings.num_processes


@pytest.mark.parametrize("value", [-1, True, 1.5, "4"])
def test_num_processes_adapter_rejects_invalid_values(value: object) -> None:
    with pytest.raises(ValidationError):
        composite_mod.NUM_PROCESSES_ADAPTER.validate_python(value)


def test_num_threads_adapter_accepts_positive_int(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        composite_mod.os,
        "process_cpu_count",
        lambda: 8,
        raising=False,
    )
    monkeypatch.setattr(
        composite_mod.os,
        "cpu_count",
        lambda: 16,
        raising=False,
    )

    assert composite_mod.NUM_THREADS_ADAPTER.validate_python(4) == 4


def test_num_threads_adapter_replaces_zero_with_settings_value(
    fake_settings: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        composite_mod.os,
        "process_cpu_count",
        lambda: 8,
        raising=False,
    )
    monkeypatch.setattr(
        composite_mod.os,
        "cpu_count",
        lambda: 16,
        raising=False,
    )

    value = composite_mod.NUM_THREADS_ADAPTER.validate_python(0)
    assert value == fake_settings.num_threads


def test_num_threads_adapter_rejects_value_larger_than_process_cpu_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        composite_mod.os,
        "process_cpu_count",
        lambda: 8,
        raising=False,
    )
    monkeypatch.setattr(
        composite_mod.os,
        "cpu_count",
        lambda: 16,
        raising=False,
    )

    with pytest.raises(ValidationError) as exc:
        composite_mod.NUM_THREADS_ADAPTER.validate_python(9)

    assert "exceeds process_cpu_count=8" in str(exc.value)


@pytest.mark.parametrize("value", [-1, True, 1.5, "4"])
def test_num_threads_adapter_rejects_invalid_values(
    value: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        composite_mod.os,
        "process_cpu_count",
        lambda: 8,
        raising=False,
    )
    monkeypatch.setattr(
        composite_mod.os,
        "cpu_count",
        lambda: 16,
        raising=False,
    )

    with pytest.raises(ValidationError):
        composite_mod.NUM_THREADS_ADAPTER.validate_python(value)


def test_num_threads_zero_from_settings_is_still_validated(
    fake_settings: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_settings.num_threads = 10

    monkeypatch.setattr(
        composite_mod.os,
        "process_cpu_count",
        lambda: 8,
        raising=False,
    )
    monkeypatch.setattr(
        composite_mod.os,
        "cpu_count",
        lambda: 16,
        raising=False,
    )

    with pytest.raises(ValidationError) as exc:
        composite_mod.NUM_THREADS_ADAPTER.validate_python(0)

    assert "exceeds process_cpu_count=8" in str(exc.value)


def test_num_threads_uses_cpu_count_when_process_cpu_count_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        composite_mod.os,
        "process_cpu_count",
        lambda: None,
        raising=False,
    )
    monkeypatch.setattr(
        composite_mod.os,
        "cpu_count",
        lambda: 12,
        raising=False,
    )

    assert composite_mod.NUM_THREADS_ADAPTER.validate_python(12) == 12

    with pytest.raises(ValidationError):
        composite_mod.NUM_THREADS_ADAPTER.validate_python(13)


def test_num_threads_falls_back_to_one_when_both_counts_are_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        composite_mod.os,
        "process_cpu_count",
        lambda: None,
        raising=False,
    )
    monkeypatch.setattr(
        composite_mod.os,
        "cpu_count",
        lambda: None,
        raising=False,
    )

    assert composite_mod.NUM_THREADS_ADAPTER.validate_python(1) == 1

    with pytest.raises(ValidationError):
        composite_mod.NUM_THREADS_ADAPTER.validate_python(2)
