from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from slothpy.core.slt_attributes import SltAttributes
from slothpy.core.slt_common import (
    SltDatasetNode,
    _dataset_label,
    _dataset_node,
    _display_dtype,
    _get_hdf5_item,
    _open_hdf5_file,
    _path_from_key,
    _rich_to_ansi,
    _rich_to_html,
)

# ---------------------------------------------------------------------------
# Raw dataset handle
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SltDataset:
    """
    Lightweight handle to a raw HDF5 dataset inside an ``.slt`` file.
    """

    file_path: Path
    path: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _path_from_key(self.path))

    @property
    def attrs(self) -> SltAttributes:
        """
        HDF5 attributes attached to this dataset.
        """
        return SltAttributes(self.file_path, self.path)

    @property
    def shape(self) -> tuple[int, ...]:
        """
        Dataset shape.
        """
        with h5py.File(self.file_path, "r") as h5:
            item = _get_hdf5_item(h5, self.path)
            if not isinstance(item, h5py.Dataset):
                raise TypeError(f"{self.path!r} is not a dataset.")
            return tuple(int(size) for size in item.shape)

    @property
    def dtype(self) -> str:
        """
        User-friendly dataset dtype.
        """
        with h5py.File(self.file_path, "r") as h5:
            item = _get_hdf5_item(h5, self.path)
            if not isinstance(item, h5py.Dataset):
                raise TypeError(f"{self.path!r} is not a dataset.")
            return _display_dtype(item)

    def to_node(self) -> SltDatasetNode:
        """
        Return this dataset as a serializable node.
        """
        with h5py.File(self.file_path, "r") as h5:
            item = _get_hdf5_item(h5, self.path)
            if not isinstance(item, h5py.Dataset):
                raise TypeError(f"{self.path!r} is not a dataset.")
            return _dataset_node(self.path, item)

    def read(self, selection: Any = ()) -> Any:
        """
        Read the dataset or a slice of it.

        HDF5 string datasets are decoded to Python strings.
        """
        with h5py.File(self.file_path, "r") as h5:
            item = _get_hdf5_item(h5, self.path)
            if not isinstance(item, h5py.Dataset):
                raise TypeError(f"{self.path!r} is not a dataset.")

            if h5py.check_string_dtype(item.dtype) is not None:
                return item.asstr()[selection]

            return item[selection]

    def write(self, selection: Any, value: Any) -> None:
        """
        Write into an existing dataset selection.
        """
        with _open_hdf5_file(self.file_path, "r+") as h5:
            item = _get_hdf5_item(h5, self.path)
            if not isinstance(item, h5py.Dataset):
                raise TypeError(f"{self.path!r} is not a dataset.")

            item[selection] = value

    def __getitem__(self, selection: Any) -> Any:
        return self.read(selection)

    def __setitem__(self, selection: Any, value: Any) -> None:
        self.write(selection, value)

    def to_numpy(self) -> np.ndarray:
        """
        Read the full dataset as a NumPy array.
        """
        return np.asarray(self.read(()))

    def __array__(self, dtype: Any = None, copy: bool | None = None) -> np.ndarray:
        array = np.asarray(self.read(()), dtype=dtype)
        if copy:
            array = array.copy()
        return array

    def show(self) -> None:
        """
        Pretty-print this dataset handle.
        """
        print(self)

    def _repr_html_(self) -> str:
        return _rich_to_html(_dataset_label(self.to_node()))

    def __repr__(self) -> str:
        return f"SltDataset(file_path={self.file_path!s}, path={self.path!r})"

    def __str__(self) -> str:
        return _rich_to_ansi(_dataset_label(self.to_node()))
