from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, ClassVar

import h5py
import xarray as xr
from pydantic import PlainValidator

from slothpy.core.slt_attributes import SltAttributes
from slothpy.core.slt_common import (
    SLOTHPY_FORMAT_VERSION,
    SLOTHPY_STORAGE_MODEL,
    SltFileNode,
    _attrs_mark_slothpy_group,
    _coerce_to_dataset,
    _create_hdf5_dataset,
    _dataset_with_slothpy_attrs,
    _file_node,
    _file_tree,
    _get_hdf5_item,
    _is_slothpy_group,
    _normalize_hdf5_path,
    _normalize_slt_path,
    _open_hdf5_file,
    _open_hdf5_handle,
    _path_from_key,
    _rich_to_ansi,
    _rich_to_html,
    _split_supported_path,
    _write_xarray_to_netcdf_with_retry,
)
from slothpy.core.slt_dataset import SltDataset
from slothpy.core.slt_group import SltGroup
from slothpy.core.slt_results import SltResults
from slothpy.groups.hamiltonian import SltHamiltonianGroup
from slothpy.groups.typed_group import SltTypedGroup
from slothpy.specs.magnetisation import SltMagnetisationGroup
from slothpy.types.aliases import PathLike, XarrayChunks

# ---------------------------------------------------------------------------
# File handle
# ---------------------------------------------------------------------------


class SltFile:
    """
    Main SlothPy file object.

    ``SltFile`` cannot be instantiated directly. Use public creation/opening
    functions, or internal private constructors from SlothPy reader/computation
    functions.

    SlothPy semantic groups are xarray/netCDF-HDF5 groups marked with SlothPy
    metadata. Raw user datasets/groups can also be stored through the simple
    mapping API.
    """

    __slots__ = ("path",)

    __TRUSTED_TOKEN: ClassVar[object] = object()

    def __new__(
        cls,
        path: PathLike,
        *,
        _token: object | None = None,
    ) -> SltFile:
        if _token is not cls.__TRUSTED_TOKEN:
            raise TypeError(
                "The SltFile object should not be instantiated directly. "
                "Use a SlothPy creation/opening function instead."
            )

        return super().__new__(cls)

    def __init__(
        self,
        path: PathLike,
        *,
        _token: object | None = None,
    ) -> None:
        self.path = _normalize_slt_path(path)

    @classmethod
    def _new(cls, path: PathLike) -> SltFile:
        """
        Internal constructor for opening an existing ``.slt`` file.
        """
        file_path = _normalize_slt_path(path)

        if not file_path.exists():
            raise FileNotFoundError(file_path)

        return cls(file_path, _token=cls.__TRUSTED_TOKEN)

    @classmethod
    def _create(cls, path: PathLike, *, overwrite: bool = False) -> SltFile:
        """
        Internal constructor for creating a new ``.slt`` file.
        """
        file_path = _normalize_slt_path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        mode = "w" if overwrite else "x"

        with _open_hdf5_file(file_path, mode) as h5:
            h5.attrs["format"] = "SlothPy"
            h5.attrs["format_version"] = SLOTHPY_FORMAT_VERSION
            h5.attrs["storage_model"] = SLOTHPY_STORAGE_MODEL

        return cls(file_path, _token=cls.__TRUSTED_TOKEN)

    @property
    def exists(self) -> bool:
        """
        Whether the underlying file exists on disk.
        """
        return self.path.exists()

    @property
    def attrs(self) -> SltAttributes:
        """
        Attributes of the root HDF5 file object.
        """
        return SltAttributes(self.path, "/")

    def open_hdf5(self, mode: str = "r") -> h5py.File:
        """
        Open the underlying HDF5 file.

        This is an advanced escape hatch for direct HDF5 access.
        Write-capable modes retry once after targeted xarray cache release if
        HDF5 reports that this file is still open read-only.
        """
        return _open_hdf5_handle(self.path, mode)

    def group(
        self,
        group_name: str,
        *,
        chunks: XarrayChunks = None,
        require_exists: bool = True,
    ) -> SltGroup:
        """
        Return a group handle.
        """
        group = SltGroup(self.path, group_name, chunks=chunks)

        if require_exists and not group.exists:
            raise KeyError(
                f"No group {group_name!r} in {self.path!s}. "
                f"Available groups: {self.groups()}"
            )

        return group

    def create_group(self, group_name: str, **attrs: Any) -> SltGroup:
        """
        Create or require a raw HDF5 group and optionally assign attributes.
        """
        group = self.group(group_name, require_exists=False)
        group.require()

        for key, value in attrs.items():
            group.attrs[key] = value

        return group

    def create_dataset(
        self,
        key: str | tuple[str, str],
        data: Any,
        *,
        overwrite: bool = False,
        chunks: bool | tuple[int, ...] | None = True,
        compression: str | None = None,
    ) -> SltDataset:
        """
        Create a raw root-level or group-level HDF5 dataset.

        Existing items are not overwritten unless ``overwrite=True`` is passed.
        SlothPy semantic groups are protected from raw mutation through this API.
        """
        path = _path_from_key(key)
        group_name, dataset_name = _split_supported_path(path)

        with _open_hdf5_file(self.path, "a") as h5:
            if path in h5:
                if not overwrite:
                    raise FileExistsError(
                        f"Item {path!r} already exists in {self.path!s}. "
                        "Delete it first or pass overwrite=True."
                    )
                del h5[path]

            if group_name is None:
                parent: h5py.File | h5py.Group = h5
            else:
                if group_name in h5 and not isinstance(h5[group_name], h5py.Group):
                    raise TypeError(
                        f"Parent path {group_name!r} exists but is not a group."
                    )

                if group_name in h5 and _attrs_mark_slothpy_group(
                    dict(h5[group_name].attrs.items())
                ):
                    raise TypeError(
                        f"Group {group_name!r} is a SlothPy semantic xarray group. "
                        "Delete/recompute it or use h5py explicitly if you really "
                        "want to mutate the underlying HDF5 data."
                    )

                parent = h5.require_group(group_name)

            _create_hdf5_dataset(
                parent,
                dataset_name,
                data,
                chunks=chunks,
                compression=compression,
            )

        return SltDataset(self.path, path)

    def set_dataset(
        self,
        key: str | tuple[str, str],
        data: Any,
        *,
        overwrite: bool = False,
        chunks: bool | tuple[int, ...] | None = True,
        compression: str | None = None,
    ) -> SltDataset:
        """
        Alias for ``create_dataset``.
        """
        return self.create_dataset(
            key,
            data,
            overwrite=overwrite,
            chunks=chunks,
            compression=compression,
        )

    def _write_slothpy_group(
        self,
        name: str,
        results: SltResults,
        *,
        overwrite: bool = False,
        encoding: dict[str, Any] | None = None,
        invalid_netcdf: bool = True,
    ) -> SltGroup:
        """
        Internal helper for SlothPy computations/readers.

        Write a valid SlothPy semantic xarray group from :class:`SltResults`.
        This is intentionally private: user code should normally not create
        SlothPy semantic groups manually.
        """
        group_name = _normalize_hdf5_path(name)

        if "/" in group_name:
            raise ValueError(
                "SlothPy semantic groups must be root-level groups. "
                "Use h5py for custom nested user layouts."
            )

        dataset_to_write = _coerce_to_dataset(results.dataset)
        dataset_to_write = _dataset_with_slothpy_attrs(
            dataset_to_write,
            primary=results.primary,
            slt_type=results.slt_type,
        )

        with _open_hdf5_file(self.path, "a") as h5:
            if group_name in h5:
                if not overwrite:
                    raise FileExistsError(
                        f"Group {group_name!r} already exists in {self.path!s}. "
                        "Pass overwrite=True to replace it."
                    )
                del h5[group_name]

        kwargs: dict[str, Any] = {
            "group": f"/{group_name}",
            "mode": "a",
            "engine": "h5netcdf",
            "invalid_netcdf": invalid_netcdf,
        }

        if encoding is not None:
            kwargs["encoding"] = encoding

        _write_xarray_to_netcdf_with_retry(dataset_to_write, self.path, **kwargs)

        group = SltGroup(self.path, group_name)
        for key, value in results.attrs.items():
            group.attrs[key] = value
        return group

    def __setitem__(self, key: str | tuple[str, str], value: Any) -> None:
        """
        Create a new raw HDF5 dataset in the file.

        Existing items are never overwritten.
        """
        self.create_dataset(key, value, overwrite=False)

    def __getitem__(
        self, key: str | tuple[str, str]
    ) -> SltGroup | SltDataset | xr.DataArray:
        """
        Return a group, raw dataset, or SlothPy xarray variable.

        Examples
        --------
        ``slt["magnetisation"]`` returns an ``SltGroup``.

        ``slt["magnetisation"]["temperature"]`` returns an xarray DataArray
        if ``magnetisation`` is a SlothPy semantic group.

        ``slt["raw_group"]["dataset"]`` returns an ``SltDataset``.
        """
        path = _path_from_key(key)
        group_name, dataset_name = _split_supported_path(path)

        if group_name is not None:
            group = SltGroup(self.path, group_name)

            if group.exists and group.is_slothpy:
                return group.variable(dataset_name)

            with h5py.File(self.path, "r") as h5:
                item = _get_hdf5_item(h5, path)

                if isinstance(item, h5py.Dataset):
                    return SltDataset(self.path, path)

                if isinstance(item, h5py.Group):
                    return SltGroup(self.path, path)

                raise TypeError(f"Unsupported HDF5 object at path {path!r}.")

        with h5py.File(self.path, "r") as h5:
            if path not in h5:
                return SltGroup(self.path, path)

            item = h5[path]

            if isinstance(item, h5py.Group):
                return SltGroup(self.path, path)

            if isinstance(item, h5py.Dataset):
                return SltDataset(self.path, path)

            raise TypeError(f"Unsupported HDF5 object at path {path!r}.")

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str | tuple):
            return False

        try:
            path = _path_from_key(key)
        except KeyError, ValueError:
            return False

        with h5py.File(self.path, "r") as h5:
            return path in h5

    def delete(self, key: str | tuple[str, str]) -> None:
        """
        Delete a group or dataset from the file.
        """
        path = _path_from_key(key)

        with _open_hdf5_file(self.path, "r+") as h5:
            if path not in h5:
                raise KeyError(f"No item {path!r} exists in file {self.path!s}.")

            del h5[path]

    def __delitem__(self, key: str | tuple[str, str]) -> None:
        self.delete(key)

    def keys(self) -> list[str]:
        """
        Return names of root-level items.
        """
        with h5py.File(self.path, "r") as h5:
            return list(h5.keys())

    def groups(self) -> list[str]:
        """
        Return names of root-level groups.
        """
        with h5py.File(self.path, "r") as h5:
            return [name for name, item in h5.items() if isinstance(item, h5py.Group)]

    def datasets(self) -> list[str]:
        """
        Return names of root-level raw datasets.
        """
        with h5py.File(self.path, "r") as h5:
            return [name for name, item in h5.items() if isinstance(item, h5py.Dataset)]

    def slothpy_groups(self) -> list[str]:
        """
        Return root-level groups marked as valid SlothPy semantic groups.
        """
        return [name for name in self.groups() if _is_slothpy_group(self.path, name)]

    def raw_groups(self) -> list[str]:
        """
        Return root-level groups not marked as SlothPy semantic groups.
        """
        return [
            name for name in self.groups() if not _is_slothpy_group(self.path, name)
        ]

    def items(self) -> dict[str, SltGroup | SltDataset]:
        """
        Return root-level items as SlothPy handles.
        """
        result: dict[str, SltGroup | SltDataset] = {}

        with h5py.File(self.path, "r") as h5:
            for name, item in h5.items():
                if isinstance(item, h5py.Group):
                    result[str(name)] = SltGroup(self.path, str(name))
                elif isinstance(item, h5py.Dataset):
                    result[str(name)] = SltDataset(self.path, str(name))

        return result

    def to_groups(
        self,
        *,
        chunks: XarrayChunks = None,
    ) -> dict[str, xr.Dataset]:
        """
        Open all valid SlothPy semantic groups as xarray Datasets.
        """
        return {
            name: self.group(name, chunks=chunks).to_dataset()
            for name in self.slothpy_groups()
        }

    def to_datatree(
        self,
        *,
        chunks: XarrayChunks = None,
        **kwargs: Any,
    ) -> Any:
        """
        Open the whole file as an xarray DataTree.

        This is an inspection helper and may fail if the file contains custom
        raw HDF5 structures that are not netCDF-compatible.
        """
        if not hasattr(xr, "open_datatree"):
            raise RuntimeError(
                "This xarray installation does not provide open_datatree()."
            )

        return xr.open_datatree(
            self.path,
            engine="h5netcdf",
            chunks=chunks,
            **kwargs,
        )

    def open_groups(
        self,
        *,
        chunks: XarrayChunks = None,
        **kwargs: Any,
    ) -> dict[str, xr.Dataset]:
        """
        Open all netCDF/HDF5 groups with xarray.open_groups.

        This is an inspection helper and may fail if the file contains custom
        raw HDF5 structures that are not netCDF-compatible.
        """
        if not hasattr(xr, "open_groups"):
            raise RuntimeError(
                "This xarray installation does not provide open_groups()."
            )

        return xr.open_groups(
            self.path,
            engine="h5netcdf",
            chunks=chunks,
            **kwargs,
        )

    def to_node(self) -> SltFileNode:
        """
        Return the whole file as a structured node tree.
        """
        return _file_node(self.path)

    def walk(self) -> SltFileNode:
        """
        Return the whole file as a structured node tree.
        """
        return self.to_node()

    def show(self) -> None:
        """
        Pretty-print this file.
        """
        print(self)

    def _repr_html_(self) -> str:
        return _rich_to_html(_file_tree(self.to_node()))

    def __repr__(self) -> str:
        return f"SltFile(path={self.path!s})"

    def __str__(self) -> str:
        return _rich_to_ansi(_file_tree(self.to_node()))

    def typed_group(
        self,
        group_name: str,
        *,
        chunks: XarrayChunks = None,
    ) -> SltTypedGroup:
        """
        Return a typed group view for an existing semantic group in this file.
        """
        group = SltGroup(self.path, group_name, chunks=chunks)

        if not group.exists:
            raise KeyError(
                f"Group {group_name!r} does not exist in file {self.path!s}."
            )

        return group.to_typed_group()

    def hamiltonian(
        self,
        group_name: str,
        *,
        chunks: XarrayChunks = None,
    ) -> SltHamiltonianGroup:

        return SltHamiltonianGroup(self.path, group_name, chunks=chunks)

    def magnetisation(
        self,
        group_name: str,
        *,
        chunks: XarrayChunks = None,
    ) -> SltMagnetisationGroup:
        return SltMagnetisationGroup(self.path, group_name, chunks=chunks)


def _validate_slt_path_or_file(value: object) -> SltFile:
    if isinstance(value, SltFile):
        return value
    if isinstance(value, (str, Path)):
        try:
            return SltFile._new(value)
        except FileNotFoundError:
            return SltFile._create(value)
    raise TypeError(
        f"slt_path_or_file must be a path or SltFile, got {type(value).__name__}."
    )


SltPathOrFile = Annotated[
    SltFile | PathLike, PlainValidator(_validate_slt_path_or_file)
]
