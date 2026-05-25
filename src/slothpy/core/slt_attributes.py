from __future__ import annotations

from collections.abc import Iterator, MutableMapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
from rich.console import Console
from rich.text import Text

from slothpy.core.slt_common import (
    _attrs_label,
    _get_hdf5_item,
    _open_hdf5_file,
    _rich_to_ansi,
    print_rich_renderable,
)
from slothpy.core.slt_html import attrs_mapping_to_html

# ---------------------------------------------------------------------------
# Attributes
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SltAttributes(MutableMapping[str, Any]):
    """
    Mapping-like wrapper around HDF5 attributes.

    Parameters
    ----------
    file_path
        Path to the underlying ``.slt`` file.
    item_path
        HDF5 path of the item whose attributes are managed.
        Use ``"/"`` for root-file attributes.
    """

    file_path: Path
    item_path: str

    def _target(self, h5: h5py.File) -> h5py.File | h5py.Group | h5py.Dataset:
        if self.item_path == "/":
            return h5

        return _get_hdf5_item(h5, self.item_path)

    def __getitem__(self, key: str) -> Any:
        with h5py.File(self.file_path, "r") as h5:
            return self._target(h5).attrs[key]

    def __setitem__(self, key: str, value: Any) -> None:
        with _open_hdf5_file(self.file_path, "r+") as h5:
            self._target(h5).attrs[key] = value

    def __delitem__(self, key: str) -> None:
        with _open_hdf5_file(self.file_path, "r+") as h5:
            del self._target(h5).attrs[key]

    def __iter__(self) -> Iterator[str]:
        with h5py.File(self.file_path, "r") as h5:
            return iter(list(self._target(h5).attrs.keys()))

    def __len__(self) -> int:
        with h5py.File(self.file_path, "r") as h5:
            return len(self._target(h5).attrs)

    def __contains__(self, key: object) -> bool:
        with h5py.File(self.file_path, "r") as h5:
            return key in self._target(h5).attrs

    def as_dict(self) -> dict[str, Any]:
        """
        Return all attributes as a plain dictionary.
        """
        with h5py.File(self.file_path, "r") as h5:
            return dict(self._target(h5).attrs.items())

    def to_rich(self) -> Text:
        """Rich label for terminal display."""
        return _attrs_label(self.as_dict())

    def print_rich(self, *, console: Console | None = None) -> None:
        """Print the Rich terminal view (not HTML)."""
        print_rich_renderable(self.to_rich(), console=console)

    def show(self, *, console: Console | None = None) -> None:
        """Alias for :meth:`print_rich`."""
        self.print_rich(console=console)

    def _repr_html_(self) -> str:
        subtitle = self.file_path.as_posix()
        if self.item_path != "/":
            subtitle = f"{subtitle} · {self.item_path}"
        return attrs_mapping_to_html(
            self.as_dict(),
            subtitle=subtitle,
        )

    def __repr__(self) -> str:
        return (
            f"SltAttributes(file_path={self.file_path!s}, item_path={self.item_path!r})"
        )

    def __str__(self) -> str:
        return _rich_to_ansi(_attrs_label(self.as_dict()))
