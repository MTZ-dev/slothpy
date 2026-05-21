from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import xarray as xr
from rich.text import Text

from slothpy.core.slt_attributes import SltAttributes
from slothpy.core.slt_common import (
    SltGroupNode,
    _create_hdf5_dataset,
    _get_hdf5_item,
    _group_node,
    _group_tree,
    _is_slothpy_group,
    _normalize_hdf5_path,
    _open_hdf5_file,
    _primary_name,
    _rich_to_ansi,
    _rich_to_html,
    _xarray_group_path,
)
from slothpy.core.slt_dataset import SltDataset
from slothpy.core.slt_results import SltResults
from slothpy.logic.predicate import SltPredicate
from slothpy.types.aliases import XarrayChunks

# ---------------------------------------------------------------------------
# Group handle
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SltGroup:
    """
    Lightweight handle to an HDF5 group.

    If the group is marked as a valid SlothPy semantic group, ``to_dataset``,
    ``to_xarray``, and ``__getitem__`` expose lazy xarray objects.

    If the group is a raw HDF5 group, ``__getitem__`` and ``__setitem__`` behave
    like simple HDF5 dataset access/creation helpers.
    """

    file_path: Path
    group_name: str
    chunks: XarrayChunks = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "group_name", _normalize_hdf5_path(self.group_name))

    @property
    def name(self) -> str:
        """
        Leaf name of this group (final ``/``-separated component).
        """
        return self.group_name.rsplit("/", maxsplit=1)[-1]

    @property
    def attrs(self) -> SltAttributes:
        """
        HDF5/netCDF attributes attached to this group.
        """
        return SltAttributes(self.file_path, self.group_name)

    @property
    def exists(self) -> bool:
        """
        Whether this group exists in the file.
        """
        with h5py.File(self.file_path, "r") as h5:
            return self.group_name in h5 and isinstance(h5[self.group_name], h5py.Group)

    @property
    def is_slothpy(self) -> bool:
        """
        Whether this group is marked as a valid SlothPy semantic xarray group.
        """
        return _is_slothpy_group(self.file_path, self.group_name)

    @property
    def type(self) -> str | None:
        """
        SlothPy group type stored in the ``slt_type`` attribute.
        """
        if not self.exists:
            raise KeyError(f"Group {self.group_name!r} does not exist.")

        value = self.attrs.as_dict().get("slt_type")
        return None if value is None else str(value)

    @property
    def primary(self) -> str | None:
        """
        Name of the primary xarray variable declared by this group.
        """
        if not self.exists:
            raise KeyError(f"Group {self.group_name!r} does not exist.")

        value = self.attrs.as_dict().get("slt_primary")
        return None if value is None else str(value)

    def require(self) -> SltGroup:
        """
        Create this raw HDF5 group if necessary and return this handle.
        """
        with _open_hdf5_file(self.file_path, "a") as h5:
            if self.group_name in h5 and not isinstance(
                h5[self.group_name], h5py.Group
            ):
                raise TypeError(
                    f"Cannot create group {self.group_name!r}; a dataset or group "
                    "with this name already exists."
                )
            h5.require_group(self.group_name)

        return self

    def with_chunks(self, chunks: XarrayChunks) -> SltGroup:
        """
        Return a new handle configured to open xarray data with ``chunks``.
        """
        return type(self)(self.file_path, self.group_name, chunks=chunks)

    def to_dataset(
        self,
        *,
        chunks: XarrayChunks = None,
        decode_cf: bool | None = None,
        **kwargs: Any,
    ) -> xr.Dataset:
        """
        Return this SlothPy semantic group as a lazy xarray Dataset.

        Raw HDF5 groups are not interpreted as SlothPy semantic groups.
        """
        if not self.exists:
            raise KeyError(f"Group {self.group_name!r} does not exist.")

        if not self.is_slothpy:
            raise TypeError(
                f"Group {self.group_name!r} is a raw HDF5 group, not a valid "
                "SlothPy semantic xarray group."
            )

        open_kwargs: dict[str, Any] = {
            "engine": "h5netcdf",
            "group": _xarray_group_path(self.group_name),
            "chunks": self.chunks if chunks is None else chunks,
            "phony_dims": "sort",
        }

        if decode_cf is not None:
            open_kwargs["decode_cf"] = decode_cf

        open_kwargs.update(kwargs)

        return xr.open_dataset(self.file_path, **open_kwargs)

    def to_xarray(
        self,
        *,
        chunks: XarrayChunks = None,
        decode_cf: bool | None = None,
        **kwargs: Any,
    ) -> xr.DataArray | xr.Dataset:
        """
        Return the primary xarray object for this SlothPy semantic group.

        If the group declares ``slt_primary``, this returns that DataArray.
        If ``slt_primary`` equals ``"__dataset__"``, this returns the whole Dataset.
        """
        dataset = self.to_dataset(
            chunks=chunks,
            decode_cf=decode_cf,
            **kwargs,
        )
        primary = _primary_name(dataset)

        if primary is None or primary == "__dataset__":
            return dataset

        if primary in dataset.data_vars:
            return dataset[primary]

        if primary in dataset.coords:
            return dataset.coords[primary]

        raise KeyError(
            f"Group {self.group_name!r} declares slt_primary={primary!r}, "
            "but this variable or coordinate is missing."
        )

    def to_slt_results(self) -> SltResults:
        """
        Return this SlothPy semantic group as a SlothPy Results object.
        """
        return SltResults(
            self.to_dataset(),
            slt_type=self.type,
            primary=self.primary,
            attrs=self.attrs.as_dict(),
        )

    def variable(
        self,
        name: str,
        *,
        chunks: XarrayChunks = None,
        decode_cf: bool | None = None,
        **kwargs: Any,
    ) -> xr.DataArray:
        """
        Return an exact data variable or coordinate as an xarray DataArray.

        No aliases are used. The requested name must match the xarray variable
        or coordinate name exactly.
        """
        dataset = self.to_dataset(
            chunks=chunks,
            decode_cf=decode_cf,
            **kwargs,
        )

        if name in dataset.data_vars:
            return dataset[name]

        if name in dataset.coords:
            return dataset.coords[name]

        raise KeyError(
            f"No variable or coordinate {name!r} in group {self.group_name!r}. "
            f"Available data variables: {[str(name) for name in dataset.data_vars]}. "
            f"Available coordinates: {[str(name) for name in dataset.coords]}."
        )

    def create_dataset(
        self,
        key: str,
        data: Any,
        *,
        overwrite: bool = False,
        chunks: bool | tuple[int, ...] | None = True,
        compression: str | None = None,
    ) -> SltDataset:
        """
        Create a raw HDF5 dataset inside this group.

        Existing items are not overwritten unless ``overwrite=True`` is passed.
        SlothPy semantic groups are protected from raw mutation through this API.
        """
        if self.exists and self.is_slothpy:
            raise TypeError(
                f"Group {self.group_name!r} is a SlothPy semantic xarray group. "
                "Delete/recompute it or use h5py explicitly if you really want "
                "to mutate the underlying HDF5 data."
            )

        dataset_name = _normalize_hdf5_path(key)

        if "/" in dataset_name:
            raise ValueError(
                "SltGroup convenience assignment supports only direct child "
                "datasets. Use h5py for deeper custom layouts."
            )

        child_path = f"{self.group_name}/{dataset_name}"

        with _open_hdf5_file(self.file_path, "a") as h5:
            if self.group_name in h5 and not isinstance(
                h5[self.group_name], h5py.Group
            ):
                raise TypeError(
                    f"Parent group {self.group_name!r} exists but is not a group."
                )

            group = h5.require_group(self.group_name)

            if dataset_name in group:
                if not overwrite:
                    raise FileExistsError(
                        f"Item {child_path!r} already exists in {self.file_path!s}. "
                        "Delete it first or pass overwrite=True."
                    )
                del group[dataset_name]

            _create_hdf5_dataset(
                group,
                dataset_name,
                data,
                chunks=chunks,
                compression=compression,
            )

        return SltDataset(self.file_path, child_path)

    def __getitem__(self, key: str) -> xr.DataArray | SltDataset | SltGroup:
        """
        Return a SlothPy xarray variable/coordinate or raw HDF5 child handle.

        For SlothPy semantic groups:
            ``group["spin"]`` returns an xarray DataArray.

        For raw HDF5 groups:
            ``group["dataset"]`` returns an SltDataset.
        """
        if self.exists and self.is_slothpy:
            return self.variable(key)

        child_name = _normalize_hdf5_path(key)

        if "/" in child_name:
            raise ValueError(
                "SltGroup convenience access supports only direct children. "
                "Use h5py for deeper custom layouts."
            )

        child_path = f"{self.group_name}/{child_name}"

        with h5py.File(self.file_path, "r") as h5:
            item = _get_hdf5_item(h5, child_path)

            if isinstance(item, h5py.Dataset):
                return SltDataset(self.file_path, child_path)

            if isinstance(item, h5py.Group):
                return SltGroup(self.file_path, child_path, chunks=self.chunks)

            raise TypeError(f"Unsupported HDF5 object at path {child_path!r}.")

    def __setitem__(self, key: str, value: Any) -> None:
        """
        Create a raw HDF5 dataset inside this group.

        Existing datasets are never overwritten.
        """
        self.create_dataset(key, value, overwrite=False)

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False

        if self.exists and self.is_slothpy:
            dataset = self.to_dataset()
            try:
                return key in dataset.data_vars or key in dataset.coords
            finally:
                dataset.close()

        child_name = _normalize_hdf5_path(key)
        if "/" in child_name:
            return False

        with h5py.File(self.file_path, "r") as h5:
            return f"{self.group_name}/{child_name}" in h5

    def delete(self, key: str) -> None:
        """
        Delete a direct child from this raw HDF5 group.

        SlothPy semantic groups are protected from raw mutation through this API.
        """
        if self.exists and self.is_slothpy:
            raise TypeError(
                f"Group {self.group_name!r} is a SlothPy semantic xarray group. "
                "Delete the whole group from the SltFile level instead."
            )

        child_name = _normalize_hdf5_path(key)

        if "/" in child_name:
            raise ValueError(
                "SltGroup deletion supports only direct children. "
                "Use h5py for deeper custom layouts."
            )

        child_path = f"{self.group_name}/{child_name}"

        with _open_hdf5_file(self.file_path, "r+") as h5:
            if child_path not in h5:
                raise KeyError(
                    f"No item {child_path!r} exists in file {self.file_path!s}."
                )

            del h5[child_path]

    def __delitem__(self, key: str) -> None:
        self.delete(key)

    def variables(self) -> list[str]:
        """
        Return names of xarray data variables in this SlothPy semantic group.
        """
        dataset = self.to_dataset()
        try:
            return [str(name) for name in dataset.data_vars]
        finally:
            dataset.close()

    def coordinates(self) -> list[str]:
        """
        Return names of xarray coordinates in this SlothPy semantic group.
        """
        dataset = self.to_dataset()
        try:
            return [str(name) for name in dataset.coords]
        finally:
            dataset.close()

    def dimensions(self) -> dict[str, int]:
        """
        Return dimension sizes in this SlothPy semantic group.
        """
        dataset = self.to_dataset()
        try:
            return {str(key): int(value) for key, value in dataset.sizes.items()}
        finally:
            dataset.close()

    def keys(self) -> list[str]:
        """
        Return available names.

        For SlothPy semantic groups, this returns data variables followed by
        coordinates. For raw HDF5 groups, this returns direct child names.
        """
        if self.exists and self.is_slothpy:
            dataset = self.to_dataset()
            try:
                return [str(name) for name in dataset.data_vars] + [
                    str(name)
                    for name in dataset.coords
                    if name not in dataset.data_vars
                ]
            finally:
                dataset.close()

        with h5py.File(self.file_path, "r") as h5:
            item = _get_hdf5_item(h5, self.group_name)
            if not isinstance(item, h5py.Group):
                raise TypeError(f"{self.group_name!r} is not a group.")
            return [str(name) for name in item.keys()]

    def items(self) -> dict[str, xr.DataArray | SltDataset | SltGroup]:
        """
        Return available children/variables as handles or xarray DataArrays.
        """
        return {key: self[key] for key in self.keys()}

    def to_dataframe(self, *args: Any, **kwargs: Any) -> Any:
        """
        Convert the primary xarray object to a pandas DataFrame.
        """
        array = self.to_xarray()

        if isinstance(array, xr.Dataset):
            return array.to_dataframe(*args, **kwargs)

        if not args and "name" not in kwargs:
            kwargs["name"] = array.name or self.primary or "value"

        return array.to_dataframe(*args, **kwargs)

    def to_node(self) -> SltGroupNode:
        """
        Return this group as a structured node.
        """
        return _group_node(self.file_path, self.group_name)

    def walk(self) -> SltGroupNode:
        """
        Return this group as a structured node.
        """
        return self.to_node()

    def show(self) -> None:
        """
        Pretty-print this group.
        """
        print(self)

    def _repr_html_(self) -> str:
        if not self.exists:
            text = Text.assemble(
                ("Proxy group", "bold blue"),
                (f" {self.group_name}", "blue"),
                (" in "),
                (str(self.file_path), "green"),
                (" does not exist."),
            )
            return _rich_to_html(text)

        return _rich_to_html(_group_tree(self.to_node()))

    def __repr__(self) -> str:
        return (
            f"SltGroup(file_path={self.file_path!s}, "
            f"group_name={self.group_name!r}, chunks={self.chunks!r})"
        )

    def __str__(self) -> str:
        if not self.exists:
            text = Text.assemble(
                ("Proxy group", "bold blue"),
                (f" {self.group_name}", "blue"),
                (" in "),
                (str(self.file_path), "green"),
                (" does not exist."),
            )
            return _rich_to_ansi(text)

        return _rich_to_ansi(_group_tree(self.to_node()))

    def has_variable(self, variable: str) -> bool:
        return variable in self.to_dataset().data_vars

    def require_rule(self, rule: SltPredicate, function_name: str) -> None:
        if not rule(self):
            raise ValueError(
                f"Group {self.group_name!r} does not satisfy rule {rule.name!r}: {{{rule!r}}}"
                f"for function {function_name!r}."
            )
