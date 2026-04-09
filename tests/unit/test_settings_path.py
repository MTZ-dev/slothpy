import importlib
import sys
from pathlib import Path

import pytest

from slothpy.config.paths import USER_SETTINGS_PATH


def test_user_settings_path():
    assert isinstance(USER_SETTINGS_PATH, Path)
    assert USER_SETTINGS_PATH.name == "settings.toml"
    assert USER_SETTINGS_PATH.parts[-3:] == (
        ".config",
        "slothpy",
        "settings.toml",
    )


def test_user_settings_path_deterministic(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    sys.modules.pop("slothpy.config.paths", None)
    paths = importlib.import_module("slothpy.config.paths")

    assert paths.USER_SETTINGS_PATH == (
        tmp_path / ".config" / "slothpy" / "settings.toml"
    )
