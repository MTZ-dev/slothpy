from pathlib import Path
from typing import Final

USER_SETTINGS_PATH: Final[Path] = Path.home() / ".config" / "slothpy" / "settings.toml"

__all__ = ["USER_SETTINGS_PATH"]
