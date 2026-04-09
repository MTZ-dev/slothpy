from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

import slothpy.config.settings as settings_mod


@pytest.fixture
def isolated_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """
    Create an isolated settings environment for each test.
    """
    settings_path = tmp_path / "settings.toml"

    monkeypatch.setattr(settings_mod, "USER_SETTINGS_PATH", settings_path)
    monkeypatch.setitem(
        settings_mod.SltSettings.model_config,
        "toml_file",
        settings_path,
    )

    for name in (
        "SLOTHPY_NUM_PROCESSES",
        "SLOTHPY_NUM_THREADS",
    ):
        monkeypatch.delenv(name, raising=False)

    monkeypatch.setattr(
        settings_mod.os,
        "process_cpu_count",
        lambda: 8,
        raising=False,
    )
    monkeypatch.setattr(
        settings_mod.os,
        "cpu_count",
        lambda: 16,
        raising=False,
    )

    monkeypatch.setattr(settings_mod, "settings", settings_mod.SltSettings())
    return settings_mod


def test_default_num_threads_uses_process_cpu_count(
    isolated_settings,
) -> None:
    sm = isolated_settings

    assert sm._default_num_threads() == 8


def test_defaults_are_loaded_into_global_settings(
    isolated_settings,
) -> None:
    sm = isolated_settings

    assert sm.settings.num_processes == 1
    assert sm.settings.num_threads == 8


def test_set_number_processes_updates_global_settings(
    isolated_settings,
) -> None:
    sm = isolated_settings

    sm.set_number_processes(3)

    assert sm.settings.num_processes == 3


def test_set_number_threads_updates_global_settings(
    isolated_settings,
) -> None:
    sm = isolated_settings

    sm.set_number_threads(5)

    assert sm.settings.num_threads == 5


def test_invalid_number_processes_raises_validation_error(
    isolated_settings,
) -> None:
    sm = isolated_settings

    with pytest.raises(ValidationError):
        sm.set_number_processes(0)


def test_invalid_number_threads_raises_validation_error(
    isolated_settings,
) -> None:
    sm = isolated_settings

    with pytest.raises(ValidationError):
        sm.set_number_threads(-1)


def test_direct_assignment_is_blocked_by_frozen_model(
    isolated_settings,
) -> None:
    sm = isolated_settings

    with pytest.raises(ValidationError) as exc:
        sm.settings.num_threads = 12

    assert exc.value.errors()[0]["type"] == "frozen_instance"


def test_save_settings_writes_file(
    isolated_settings,
) -> None:
    sm = isolated_settings

    sm.set_number_processes(2)
    sm.set_number_threads(4)

    path = sm.save_settings()

    assert path.exists()
    assert path.is_file()


def test_save_and_reload_roundtrip(
    isolated_settings,
) -> None:
    sm = isolated_settings

    sm.set_number_processes(2)
    sm.set_number_threads(4)
    sm.save_settings()

    sm.set_number_processes(7)
    sm.set_number_threads(6)

    assert sm.settings.num_processes == 7
    assert sm.settings.num_threads == 6

    reloaded = sm.reload_settings()

    assert reloaded.num_processes == 2
    assert reloaded.num_threads == 4
    assert sm.settings.num_processes == 2
    assert sm.settings.num_threads == 4


def test_set_number_processes_permanent_saves_immediately(
    isolated_settings,
) -> None:
    sm = isolated_settings

    path = sm.USER_SETTINGS_PATH
    assert not path.exists()

    sm.set_number_processes(4, permanent=True)

    assert path.exists()

    sm.reload_settings()
    assert sm.settings.num_processes == 4


def test_set_number_threads_permanent_saves_immediately(
    isolated_settings,
) -> None:
    sm = isolated_settings

    path = sm.USER_SETTINGS_PATH
    assert not path.exists()

    sm.set_number_threads(6, permanent=True)

    assert path.exists()

    sm.reload_settings()
    assert sm.settings.num_threads == 6


def test_reset_settings_restores_defaults(
    isolated_settings,
) -> None:
    sm = isolated_settings

    sm.set_number_processes(5)
    sm.set_number_threads(3)

    assert sm.settings.num_processes == 5
    assert sm.settings.num_threads == 3

    reset = sm.reset_settings()

    assert reset.num_processes == 1
    assert reset.num_threads == 8
    assert sm.settings.num_processes == 1
    assert sm.settings.num_threads == 8


def test_reset_settings_permanent_overwrites_saved_values(
    isolated_settings,
) -> None:
    sm = isolated_settings

    sm.set_number_processes(4, permanent=True)
    sm.set_number_threads(2, permanent=True)

    sm.reset_settings(permanent=True)
    sm.reload_settings()

    assert sm.settings.num_processes == 1
    assert sm.settings.num_threads == 8


def test_default_num_threads_falls_back_to_cpu_count(
    isolated_settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sm = isolated_settings

    monkeypatch.setattr(
        sm.os,
        "process_cpu_count",
        lambda: None,
        raising=False,
    )
    monkeypatch.setattr(
        sm.os,
        "cpu_count",
        lambda: 16,
        raising=False,
    )

    assert sm._default_num_threads() == 16


def test_default_num_threads_falls_back_to_one(
    isolated_settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sm = isolated_settings

    monkeypatch.setattr(
        sm.os,
        "process_cpu_count",
        lambda: None,
        raising=False,
    )
    monkeypatch.setattr(
        sm.os,
        "cpu_count",
        lambda: None,
        raising=False,
    )

    assert sm._default_num_threads() == 1


def test_str_returns_human_readable_representation(
    isolated_settings,
) -> None:
    sm = isolated_settings

    text = str(sm.settings)

    assert text.startswith("SlothPy settings:")
    assert "num_processes = 1" in text
    assert "num_threads   = 8" in text


def test_show_prints_human_readable_representation(
    isolated_settings,
    capsys: pytest.CaptureFixture[str],
) -> None:
    sm = isolated_settings

    sm.settings.show()
    captured = capsys.readouterr()

    assert "SlothPy settings:" in captured.out
    assert "num_processes = 1" in captured.out
    assert "num_threads   = 8" in captured.out
