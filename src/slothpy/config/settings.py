from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import tomli_w
from pydantic import Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

from slothpy.config.paths import USER_SETTINGS_PATH
from slothpy.types.primitive import PositiveInt


def _default_num_threads() -> int:
    """
    Return the default thread count.

    Prefer ``os.process_cpu_count()`` when available, then fall back to
    ``os.cpu_count()``, then finally to 1.
    """
    process_cpu_count = getattr(os, "process_cpu_count", None)
    if process_cpu_count is not None:
        value = process_cpu_count()
        if value is not None:
            return value

    value = os.cpu_count()
    return 1 if value is None else value


def _default_settings_data() -> dict[str, int]:
    return {
        "num_processes": 1,
        "num_threads": _default_num_threads(),
    }


class SltSettings(BaseSettings):
    """
    Global SlothPy settings.

    Notes
    -----
    - Instances are frozen to prevent direct user mutation.
    - Use the module-level helper functions to change values.
    - Settings are loaded from, in order:
      1. explicit init kwargs
      2. environment variables
      3. dotenv
      4. TOML file
      5. file secrets
      6. field defaults
    """

    model_config = SettingsConfigDict(
        env_prefix="SLOTHPY_",
        validate_assignment=True,
        extra="forbid",
        frozen=True,
        toml_file=USER_SETTINGS_PATH,
    )

    num_processes: PositiveInt = Field(
        default=1,
        strict=True,
        description="Number of MPI processes used during calculations.",
    )
    num_threads: PositiveInt = Field(
        default_factory=_default_num_threads,
        strict=True,
        description="Number of threads per MPI process.",
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            TomlConfigSettingsSource(settings_cls, deep_merge=True),
            file_secret_settings,
        )

    def show(self) -> None:
        """Print a human-readable representation of the current settings."""
        print(self)

    def __str__(self) -> str:
        data = self.model_dump(mode="python")
        width = max(len(k) for k in data) if data else 0
        lines = ["SlothPy settings:"]
        for key, value in data.items():
            lines.append(f"  {key:<{width}} = {value}")
        return "\n".join(lines)

    def save(self, path: Path | None = None) -> Path:
        """
        Save the current settings to TOML.
        """
        target = path or USER_SETTINGS_PATH
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as fh:
            tomli_w.dump(self.model_dump(mode="json"), fh)
        return target


settings = SltSettings()


def _configure(*, permanent: bool = False, **changes: Any) -> SltSettings:
    """
    Create a new validated global settings object with the requested changes.
    """
    global settings
    data = settings.model_dump(mode="python")
    data.update(changes)
    settings = SltSettings(**data)
    if permanent:
        settings.save()
    return settings


def reload_settings() -> SltSettings:
    """
    Reload settings from configured sources.
    """
    global settings
    settings = SltSettings()
    return settings


def save_settings() -> Path:
    """
    Save the current global settings permanently.
    """
    return settings.save()


def set_number_processes(value: int = 1, permanent: bool = False) -> None:
    """
    Set the number of MPI processes used during calculations.
    """
    _configure(num_processes=value, permanent=permanent)


def set_number_threads(value: int = 1, permanent: bool = False) -> None:
    """
    Set the number of threads used by each MPI process.
    """
    _configure(num_threads=value, permanent=permanent)


def reset_settings(permanent: bool = False) -> SltSettings:
    """
    Reset settings to code defaults.
    """
    global settings
    settings = SltSettings(**_default_settings_data())
    if permanent:
        settings.save()
    return settings
