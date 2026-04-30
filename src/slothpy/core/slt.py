from __future__ import annotations

from collections.abc import Iterator, MutableMapping
from dataclasses import dataclass
from io import StringIO
from pathlib import Path, PurePosixPath
from typing import Any, ClassVar, Literal

import h5py
import numpy as np
import xarray as xr
from rich.console import Console
from rich.text import Text
from rich.tree import Tree

type PathLike = str | Path
type NodeKind = Literal["file", "group", "dataset", "coordinate", "data_variable"]
type XarrayChunks = int | str | dict[str, Any] | None


SLOTHPY_FORMAT_VERSION = "0.4"
SLOTHPY_STORAGE_MODEL = "xarray-netcdf4-hdf5"


# ---------------------------------------------------------------------------
# Rich rendering helpers
# ---------------------------------------------------------------------------


def _rich_to_ansi(renderable: Any) -> str:
    """
    Render a Rich renderable to an ANSI-colored string.

    This is used by ``__str__`` and therefore by ``print(obj)``.
    """
    stream = StringIO()
    console = Console(
        file=stream,
        force_terminal=True,
        color_system="auto",
        width=120,
    )
    console.print(renderable)
    return stream.getvalue().rstrip()


def _rich_to_html(renderable: Any) -> str:
    """
    Render a Rich renderable to an HTML fragment.

    This is used by ``_repr_html_`` for Jupyter/marimo bare-cell display.
    """
    stream = StringIO()
    console = Console(
        file=stream,
        record=True,
        force_terminal=True,
        color_system="truecolor",
        width=120,
    )
    console.print(renderable)

    return console.export_html(
        inline_styles=True,
        code_format=(
            "<pre style='white-space: pre; "
            "font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "
            "Liberation Mono, monospace; margin: 0;'>{code}</pre>"
        ),
    ).rstrip()


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def _normalize_slt_path(path: PathLike) -> Path:
    """
    Normalize a filesystem path to an ``.slt`` file path.

    If the suffix is missing or different, ``.slt`` is used.
    """
    path = Path(path).expanduser()

    if path.suffix != ".slt":
        path = path.with_suffix(".slt")

    return path


def _normalize_hdf5_path(path: str) -> str:
    """
    Normalize an HDF5 path to a relative POSIX-style path.
    """
    normalized = str(PurePosixPath(path.strip("/")))

    if normalized in {"", "."}:
        raise ValueError("HDF5 path cannot be empty.")

    return normalized


def _path_from_key(key: str | tuple[str, str]) -> str:
    """
    Convert SlothPy indexing keys into normalized HDF5 paths.

    Examples
    --------
    ``"energies"`` becomes ``"energies"``.

    ``"orca_triplets/energies"`` becomes ``"orca_triplets/energies"``.

    ``("orca_triplets", "energies")`` becomes ``"orca_triplets/energies"``.
    """
    if isinstance(key, tuple):
        if len(key) != 2:
            raise KeyError("Tuple indexing expects exactly (group, dataset).")

        group, dataset = key
        group = _normalize_hdf5_path(group)
        dataset = _normalize_hdf5_path(dataset)

        if "/" in group or "/" in dataset:
            raise ValueError("Tuple indexing supports only (root_group, dataset).")

        return f"{group}/{dataset}"

    path = _normalize_hdf5_path(key)
    parts = path.split("/")

    if len(parts) > 2:
        raise ValueError(
            f"HDF5 path {path!r} is too deep for SlothPy's convenience API. "
            "Use h5py directly for deeper custom layouts."
        )

    return path


def _split_supported_path(path: str) -> tuple[str | None, str]:
    """
    Split a supported SlothPy HDF5 path.

    Returns
    -------
    tuple[str | None, str]
        ``(None, name)`` for root-level objects,
        ``(group, dataset)`` for datasets inside root-level groups.
    """
    path = _path_from_key(path)
    parts = path.split("/")

    if len(parts) == 1:
        return None, parts[0]

    return parts[0], parts[1]


def _xarray_group_path(path: str) -> str:
    """
    Return an xarray/netCDF group path with leading slash.
    """
    return f"/{_normalize_hdf5_path(path)}"


# ---------------------------------------------------------------------------
# HDF5 dataset helpers
# ---------------------------------------------------------------------------


def _string_dtype() -> Any:
    """
    Return SlothPy's default HDF5 string dtype.
    """
    return h5py.string_dtype(encoding="utf-8")


def _prepare_dataset_data(value: Any) -> tuple[Any, Any | None]:
    """
    Prepare data for HDF5 dataset creation.

    Strings and sequences of strings are stored as UTF-8 HDF5 strings.
    Other values are converted through ``numpy.asarray``.
    """
    if isinstance(value, str):
        return value, _string_dtype()

    if (
        isinstance(value, list | tuple)
        and value
        and all(isinstance(item, str) for item in value)
    ):
        return np.asarray(value, dtype=object), _string_dtype()

    array = np.asarray(value)

    if array.dtype.kind in {"U", "O"}:
        flat = array.ravel()
        if flat.size > 0 and all(isinstance(item, str) for item in flat):
            return array.astype(object), _string_dtype()

    return array, None


def _is_scalar_dataset_data(data: Any) -> bool:
    """
    Return True if the dataset data represents an HDF5 scalar dataset.

    Scalar datasets cannot be chunked or compressed.
    """
    if isinstance(data, str):
        return True

    if isinstance(data, np.ndarray):
        return data.shape == ()

    return np.asarray(data).shape == ()


def _display_dtype(dataset: h5py.Dataset) -> str:
    """
    Return a user-friendly dtype representation.
    """
    if h5py.check_string_dtype(dataset.dtype) is not None:
        return "str"

    return str(dataset.dtype)


def _create_hdf5_dataset(
    parent: h5py.File | h5py.Group,
    name: str,
    data: Any,
    dtype: Any | None,
    *,
    chunks: bool | tuple[int, ...] | None,
    compression: str | None,
) -> h5py.Dataset:
    """
    Create an HDF5 dataset, avoiding chunk/compression options for scalar data.
    """
    kwargs: dict[str, Any] = {"data": data}

    if dtype is not None:
        kwargs["dtype"] = dtype

    if not _is_scalar_dataset_data(data):
        if chunks is not None:
            kwargs["chunks"] = chunks
        if compression is not None:
            kwargs["compression"] = compression

    return parent.create_dataset(name, **kwargs)


def _get_hdf5_item(h5: h5py.File, path: str) -> h5py.File | h5py.Group | h5py.Dataset:
    """
    Return an HDF5 item or raise a clear KeyError.
    """
    if path == "/":
        return h5

    path = _normalize_hdf5_path(path)

    if path not in h5:
        raise KeyError(f"No item {path!r} exists in file {h5.filename!r}.")

    item = h5[path]

    if isinstance(item, h5py.Group | h5py.Dataset):
        return item

    raise TypeError(f"Unsupported HDF5 object at path {path!r}.")


# ---------------------------------------------------------------------------
# SlothPy/xarray helpers
# ---------------------------------------------------------------------------


def _release_xarray_file_handles() -> None:
    """
    Release xarray backend file handles before opening the same ``.slt`` file
    for HDF5 writing.

    This is mainly needed in notebooks, where lazy xarray objects or displayed
    outputs can keep read-only backend handles alive.

    Notes
    -----
    This clears xarray's global file cache. Lazy xarray objects can usually
    reopen files when accessed again, but do not call this while a Dask graph is
    actively reading from the same file.
    """
    try:
        from xarray.backends.file_manager import FILE_CACHE
    except Exception:
        return

    FILE_CACHE.clear()


def _truthy_attr(value: Any) -> bool:
    """
    Interpret a stored HDF5/netCDF attribute as a boolean.
    """
    if isinstance(value, bytes):
        value = value.decode("utf-8")

    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}

    if isinstance(value, int | np.integer):
        return bool(value)

    return False


def _attrs_mark_slothpy_group(attrs: dict[str, Any]) -> bool:
    """
    Return True if attributes mark a group as a SlothPy semantic xarray group.
    """
    return _truthy_attr(attrs.get("slt_valid", False))


def _coerce_to_dataset(
    value: xr.Dataset | xr.DataArray,
    *,
    dataarray_name: str = "data",
) -> xr.Dataset:
    """
    Coerce an xarray object to ``xr.Dataset``.
    """
    if isinstance(value, xr.Dataset):
        return value

    if isinstance(value, xr.DataArray):
        name = value.name or dataarray_name
        return value.to_dataset(name=name)

    raise TypeError(
        "SlothPy semantic groups can only be written from "
        "xarray.Dataset or xarray.DataArray objects."
    )


def _primary_name(dataset: xr.Dataset) -> str | None:
    """
    Return the primary variable declared by a SlothPy xarray group.
    """
    primary = dataset.attrs.get("slt_primary")

    if primary is None:
        return None

    if isinstance(primary, bytes):
        primary = primary.decode("utf-8")

    primary = str(primary)

    if primary == "":
        return None

    return primary


def _dataset_with_slothpy_attrs(
    dataset: xr.Dataset,
    *,
    primary: str | None,
    slt_type: str | None,
) -> xr.Dataset:
    """
    Return a shallow copy of a dataset with SlothPy semantic metadata attached.
    """
    result = dataset.copy(deep=False)
    attrs = dict(result.attrs)

    attrs["slt_valid"] = "true"
    attrs["slt_version"] = SLOTHPY_FORMAT_VERSION
    attrs["slt_storage_model"] = SLOTHPY_STORAGE_MODEL

    if slt_type is not None:
        attrs["slt_type"] = slt_type

    if primary is not None:
        attrs["slt_primary"] = primary
    elif "slt_primary" not in attrs:
        if len(result.data_vars) == 1:
            attrs["slt_primary"] = next(iter(result.data_vars))
        else:
            attrs["slt_primary"] = "__dataset__"

    declared_primary = str(attrs["slt_primary"])
    if declared_primary != "__dataset__":
        if (
            declared_primary not in result.data_vars
            and declared_primary not in result.coords
        ):
            raise KeyError(
                f"Declared primary variable {declared_primary!r} is not present "
                "as a data variable or coordinate."
            )

    result.attrs = attrs
    return result


def _is_slothpy_group(file_path: Path, path: str) -> bool:
    """
    Return True if a group is marked as a SlothPy semantic group.
    """
    path = _normalize_hdf5_path(path)

    with h5py.File(file_path, "r") as h5:
        if path not in h5 or not isinstance(h5[path], h5py.Group):
            return False

        return _attrs_mark_slothpy_group(dict(h5[path].attrs.items()))


# ---------------------------------------------------------------------------
# Node model
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SltDatasetNode:
    """
    Serializable description of one raw HDF5 dataset node.
    """

    path: str
    name: str
    attrs: dict[str, Any]
    shape: tuple[int, ...]
    dtype: str
    kind: Literal["dataset"] = "dataset"


@dataclass(frozen=True, slots=True)
class SltVariableNode:
    """
    Serializable description of one xarray coordinate or data variable.
    """

    name: str
    kind: Literal["coordinate", "data_variable"]
    dims: tuple[str, ...]
    shape: tuple[int, ...]
    dtype: str
    attrs: dict[str, Any]
    primary: bool = False


@dataclass(frozen=True, slots=True)
class SltGroupNode:
    """
    Serializable description of one HDF5 group.

    If ``is_slothpy`` is True, the group is shown as an xarray-backed semantic
    SlothPy group. Otherwise, it is shown as a raw HDF5 group.
    """

    path: str
    name: str
    attrs: dict[str, Any]
    is_slothpy: bool
    dimensions: dict[str, int]
    coordinates: tuple[SltVariableNode, ...]
    data_variables: tuple[SltVariableNode, ...]
    primary: str | None
    raw_datasets: tuple[SltDatasetNode, ...]
    child_groups: tuple[str, ...]
    readable: bool = True
    error: str | None = None
    kind: Literal["group"] = "group"


@dataclass(frozen=True, slots=True)
class SltFileNode:
    """
    Serializable description of the whole ``.slt`` file tree.
    """

    path: Path
    attrs: dict[str, Any]
    groups: tuple[SltGroupNode, ...]
    datasets: tuple[SltDatasetNode, ...]
    kind: Literal["file"] = "file"


SltNodeInfo = SltFileNode | SltGroupNode | SltDatasetNode | SltVariableNode


def _dataset_node(path: str, dataset: h5py.Dataset) -> SltDatasetNode:
    """
    Build a dataset node from an open HDF5 dataset.
    """
    return SltDatasetNode(
        path=path,
        name=path.rsplit("/", maxsplit=1)[-1],
        attrs=dict(dataset.attrs.items()),
        shape=tuple(int(size) for size in dataset.shape),
        dtype=_display_dtype(dataset),
    )


def _variable_node(
    name: str,
    data_array: xr.DataArray,
    *,
    kind: Literal["coordinate", "data_variable"],
    primary: str | None,
) -> SltVariableNode:
    """
    Build a serializable variable node from an xarray DataArray.
    """
    return SltVariableNode(
        name=name,
        kind=kind,
        dims=tuple(str(dim) for dim in data_array.dims),
        shape=tuple(int(size) for size in data_array.shape),
        dtype=str(data_array.dtype),
        attrs=dict(data_array.attrs),
        primary=name == primary,
    )


def _raw_group_children(
    group_path: str,
    group: h5py.Group,
) -> tuple[tuple[SltDatasetNode, ...], tuple[str, ...]]:
    """
    Return direct raw datasets and child group names of an HDF5 group.
    """
    datasets: list[SltDatasetNode] = []
    child_groups: list[str] = []

    for child_name, child in group.items():
        child_path = f"{group_path}/{child_name}"

        if isinstance(child, h5py.Dataset):
            datasets.append(_dataset_node(child_path, child))
        elif isinstance(child, h5py.Group):
            child_groups.append(child_name)

    return tuple(datasets), tuple(child_groups)


def _group_node(file_path: Path, path: str) -> SltGroupNode:
    """
    Build a structured node for one HDF5 group.
    """
    path = _normalize_hdf5_path(path)
    name = path.rsplit("/", maxsplit=1)[-1]

    with h5py.File(file_path, "r") as h5:
        item = _get_hdf5_item(h5, path)

        if not isinstance(item, h5py.Group):
            raise TypeError(f"{path!r} is not a group.")

        attrs = dict(item.attrs.items())
        is_slothpy = _attrs_mark_slothpy_group(attrs)
        raw_datasets, child_groups = _raw_group_children(path, item)

    if not is_slothpy:
        return SltGroupNode(
            path=path,
            name=name,
            attrs=attrs,
            is_slothpy=False,
            dimensions={},
            coordinates=(),
            data_variables=(),
            primary=None,
            raw_datasets=raw_datasets,
            child_groups=child_groups,
        )

    try:
        dataset = xr.open_dataset(
            file_path,
            group=f"/{path}",
            engine="h5netcdf",
            chunks=None,
            phony_dims="sort",
        )
    except Exception as exc:
        return SltGroupNode(
            path=path,
            name=name,
            attrs=attrs,
            is_slothpy=True,
            dimensions={},
            coordinates=(),
            data_variables=(),
            primary=None,
            raw_datasets=(),
            child_groups=(),
            readable=False,
            error=str(exc),
        )

    try:
        primary = _primary_name(dataset)
        dimensions = {str(key): int(value) for key, value in dataset.sizes.items()}

        coordinates = tuple(
            _variable_node(
                coord_name,
                dataset.coords[coord_name],
                kind="coordinate",
                primary=primary,
            )
            for coord_name in dataset.coords
        )

        data_variables = tuple(
            _variable_node(
                var_name,
                dataset[var_name],
                kind="data_variable",
                primary=primary,
            )
            for var_name in dataset.data_vars
        )

        return SltGroupNode(
            path=path,
            name=name,
            attrs=dict(dataset.attrs),
            is_slothpy=True,
            dimensions=dimensions,
            coordinates=coordinates,
            data_variables=data_variables,
            primary=primary,
            raw_datasets=(),
            child_groups=(),
        )
    finally:
        dataset.close()


def _file_node(path: Path) -> SltFileNode:
    """
    Build a structured node tree for an ``.slt`` file.
    """
    groups: list[SltGroupNode] = []
    datasets: list[SltDatasetNode] = []

    with h5py.File(path, "r") as h5:
        attrs = dict(h5.attrs.items())

        for name, item in h5.items():
            if isinstance(item, h5py.Group):
                groups.append(_group_node(path, name))
            elif isinstance(item, h5py.Dataset):
                datasets.append(_dataset_node(name, item))

    return SltFileNode(
        path=path,
        attrs=attrs,
        groups=tuple(groups),
        datasets=tuple(datasets),
    )


# ---------------------------------------------------------------------------
# Rich tree labels
# ---------------------------------------------------------------------------


def _file_label(node: SltFileNode) -> Text:
    """
    Build the Rich label for an SltFile node.
    """
    text = Text.assemble(
        ("SltFile", "bold red"),
        (": ", "default"),
        (str(node.path), "green"),
    )

    version = node.attrs.get("format_version")
    if version is not None:
        text.append(f" [version={version!r}]", style="yellow")

    return text


def _group_label(node: SltGroupNode) -> Text:
    """
    Build the Rich label for a group node.
    """
    text = Text.assemble(
        ("Group", "bold blue"),
        (f": {node.name}", "blue"),
    )

    if not node.readable:
        text.append(" [unreadable]", style="bold red")
        return text

    if node.is_slothpy:
        slt_type = node.attrs.get("slt_type")
        if slt_type is not None:
            text.append(f" [Type={slt_type!r}]", style="yellow")

        if node.primary is not None:
            text.append(f" [Primary={node.primary!r}]", style="yellow")
    else:
        text.append(" [raw HDF5]", style="bright_black")

    return text


def _dimension_label(name: str, size: int) -> Text:
    """
    Build the Rich label for a dimension.
    """
    return Text.assemble(
        (name, "cyan"),
        (": ", "default"),
        (str(size), "green"),
    )


def _dataset_label(node: SltDatasetNode) -> Text:
    """
    Build the Rich label for a raw HDF5 dataset.
    """
    text = Text.assemble(
        ("Dataset", "bold magenta"),
        (f": {node.name}", "magenta"),
        (f" shape={node.shape}", "cyan"),
        (f" dtype={node.dtype}", "cyan"),
    )

    for key, value in node.attrs.items():
        text.append(f" | {key}: {value!r}", style="yellow")

    return text


def _variable_label(node: SltVariableNode) -> Text:
    """
    Build the Rich label for a coordinate or data variable.
    """
    if node.kind == "coordinate":
        title = "Coordinate"
        style = "bold cyan"
        name_style = "cyan"
    else:
        title = "Data variable"
        style = "bold magenta"
        name_style = "magenta"

    dims = ", ".join(node.dims)
    shape = ", ".join(str(size) for size in node.shape)

    text = Text.assemble(
        (title, style),
        (": ", "default"),
        (node.name, name_style),
        (f"({dims})", "bright_black"),
        (f" shape=({shape})", "green"),
        (f" dtype={node.dtype}", "green"),
    )

    if node.primary:
        text.append(" ← primary", style="bold yellow")

    unit = node.attrs.get("unit")
    if unit is not None:
        text.append(f" [{unit}]", style="yellow")

    long_name = node.attrs.get("long_name")
    if long_name is not None:
        text.append(f" | {long_name}", style="yellow")

    return text


def _attrs_label(attrs: dict[str, Any]) -> Text:
    """
    Build a Rich label for an attributes mapping.
    """
    text = Text("{")
    for index, (key, value) in enumerate(attrs.items()):
        if index:
            text.append(", ")
        text.append(str(key), style="yellow")
        text.append(": ")
        text.append(repr(value), style="green")
    text.append("}")
    return text


def _add_group_to_tree(parent: Tree, node: SltGroupNode) -> None:
    """
    Add a group node to a Rich tree.
    """
    group_tree = parent.add(_group_label(node))

    if not node.readable:
        group_tree.add(Text(node.error or "Unknown error", style="bold red"))
        return

    if not node.is_slothpy:
        if node.raw_datasets:
            datasets_tree = group_tree.add(Text("Datasets", style="bold magenta"))
            for dataset in node.raw_datasets:
                datasets_tree.add(_dataset_label(dataset))

        if node.child_groups:
            child_groups_tree = group_tree.add(Text("Child groups", style="bold blue"))
            for child_group in node.child_groups:
                child_groups_tree.add(Text(child_group, style="blue"))

        if not node.raw_datasets and not node.child_groups:
            group_tree.add(Text("(empty)", style="bright_black"))

        return

    dimensions_tree = group_tree.add(Text("Dimensions", style="bold green"))
    if node.dimensions:
        for name, size in node.dimensions.items():
            dimensions_tree.add(_dimension_label(name, size))
    else:
        dimensions_tree.add(Text("(none)", style="bright_black"))

    coords_tree = group_tree.add(Text("Coordinates", style="bold cyan"))
    if node.coordinates:
        for coord in node.coordinates:
            coords_tree.add(_variable_label(coord))
    else:
        coords_tree.add(Text("(none)", style="bright_black"))

    vars_tree = group_tree.add(Text("Data variables", style="bold magenta"))
    if node.data_variables:
        for variable in node.data_variables:
            vars_tree.add(_variable_label(variable))
    else:
        vars_tree.add(Text("(none)", style="bright_black"))


def _file_tree(node: SltFileNode) -> Tree:
    """
    Build a Rich tree for an SltFile node.
    """
    tree = Tree(_file_label(node))

    if node.datasets:
        root_datasets = tree.add(Text("Root datasets", style="bold magenta"))
        for dataset in node.datasets:
            root_datasets.add(_dataset_label(dataset))

    if node.groups:
        for group in node.groups:
            _add_group_to_tree(tree, group)

    if not node.datasets and not node.groups:
        tree.add(Text("(empty)", style="bright_black"))

    return tree


def _group_tree(node: SltGroupNode) -> Tree:
    """
    Build a Rich tree for an SltGroup node.
    """
    tree = Tree(_group_label(node))

    if not node.readable:
        tree.add(Text(node.error or "Unknown error", style="bold red"))
        return tree

    if not node.is_slothpy:
        if node.raw_datasets:
            datasets_tree = tree.add(Text("Datasets", style="bold magenta"))
            for dataset in node.raw_datasets:
                datasets_tree.add(_dataset_label(dataset))

        if node.child_groups:
            child_groups_tree = tree.add(Text("Child groups", style="bold blue"))
            for child_group in node.child_groups:
                child_groups_tree.add(Text(child_group, style="blue"))

        if not node.raw_datasets and not node.child_groups:
            tree.add(Text("(empty)", style="bright_black"))

        return tree

    dimensions_tree = tree.add(Text("Dimensions", style="bold green"))
    if node.dimensions:
        for name, size in node.dimensions.items():
            dimensions_tree.add(_dimension_label(name, size))
    else:
        dimensions_tree.add(Text("(none)", style="bright_black"))

    coords_tree = tree.add(Text("Coordinates", style="bold cyan"))
    if node.coordinates:
        for coord in node.coordinates:
            coords_tree.add(_variable_label(coord))
    else:
        coords_tree.add(Text("(none)", style="bright_black"))

    vars_tree = tree.add(Text("Data variables", style="bold magenta"))
    if node.data_variables:
        for variable in node.data_variables:
            vars_tree.add(_variable_label(variable))
    else:
        vars_tree.add(Text("(none)", style="bright_black"))

    return tree


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

        return _get_hdf5_item(h5, _normalize_hdf5_path(self.item_path))

    def __getitem__(self, key: str) -> Any:
        with h5py.File(self.file_path, "r") as h5:
            return self._target(h5).attrs[key]

    def __setitem__(self, key: str, value: Any) -> None:
        _release_xarray_file_handles()
        with h5py.File(self.file_path, "r+") as h5:
            self._target(h5).attrs[key] = value

    def __delitem__(self, key: str) -> None:
        _release_xarray_file_handles()
        with h5py.File(self.file_path, "r+") as h5:
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

    def show(self) -> None:
        """
        Pretty-print attributes.
        """
        print(self)

    def _repr_html_(self) -> str:
        return _rich_to_html(_attrs_label(self.as_dict()))

    def __repr__(self) -> str:
        return (
            f"SltAttributes(file_path={self.file_path!s}, item_path={self.item_path!r})"
        )

    def __str__(self) -> str:
        return _rich_to_ansi(_attrs_label(self.as_dict()))


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
        _release_xarray_file_handles()
        with h5py.File(self.file_path, "r+") as h5:
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
    path: str
    chunks: XarrayChunks = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _normalize_hdf5_path(self.path))

    @property
    def name(self) -> str:
        """
        Final path component of this group.
        """
        return self.path.rsplit("/", maxsplit=1)[-1]

    @property
    def attrs(self) -> SltAttributes:
        """
        HDF5/netCDF attributes attached to this group.
        """
        return SltAttributes(self.file_path, self.path)

    @property
    def exists(self) -> bool:
        """
        Whether this group exists in the file.
        """
        with h5py.File(self.file_path, "r") as h5:
            return self.path in h5 and isinstance(h5[self.path], h5py.Group)

    @property
    def is_slothpy(self) -> bool:
        """
        Whether this group is marked as a valid SlothPy semantic xarray group.
        """
        return _is_slothpy_group(self.file_path, self.path)

    @property
    def type(self) -> str | None:
        """
        SlothPy group type stored in the ``slt_type`` attribute.
        """
        if not self.exists:
            raise KeyError(f"Group {self.path!r} does not exist.")

        value = self.attrs.as_dict().get("slt_type")
        return None if value is None else str(value)

    @property
    def primary(self) -> str | None:
        """
        Name of the primary xarray variable declared by this group.
        """
        if not self.exists:
            raise KeyError(f"Group {self.path!r} does not exist.")

        value = self.attrs.as_dict().get("slt_primary")
        return None if value is None else str(value)

    def require(self) -> SltGroup:
        """
        Create this raw HDF5 group if necessary and return this handle.
        """
        _release_xarray_file_handles()
        with h5py.File(self.file_path, "a") as h5:
            if self.path in h5 and not isinstance(h5[self.path], h5py.Group):
                raise TypeError(
                    f"Cannot create group {self.path!r}; a dataset with this "
                    "name already exists."
                )
            h5.require_group(self.path)

        return self

    def with_chunks(self, chunks: XarrayChunks) -> SltGroup:
        """
        Return a new handle configured to open xarray data with ``chunks``.
        """
        return type(self)(self.file_path, self.path, chunks=chunks)

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
            raise KeyError(f"Group {self.path!r} does not exist.")

        if not self.is_slothpy:
            raise TypeError(
                f"Group {self.path!r} is a raw HDF5 group, not a valid "
                "SlothPy semantic xarray group."
            )

        open_kwargs: dict[str, Any] = {
            "engine": "h5netcdf",
            "group": _xarray_group_path(self.path),
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
            f"Group {self.path!r} declares slt_primary={primary!r}, "
            "but this variable or coordinate is missing."
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
            f"No variable or coordinate {name!r} in group {self.path!r}. "
            f"Available data variables: {list(dataset.data_vars)}. "
            f"Available coordinates: {list(dataset.coords)}."
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
                f"Group {self.path!r} is a SlothPy semantic xarray group. "
                "Delete/recompute it or use h5py explicitly if you really want "
                "to mutate the underlying HDF5 data."
            )

        dataset_name = _normalize_hdf5_path(key)

        if "/" in dataset_name:
            raise ValueError(
                "SltGroup convenience assignment supports only direct child "
                "datasets. Use h5py for deeper custom layouts."
            )

        child_path = f"{self.path}/{dataset_name}"
        prepared, dtype = _prepare_dataset_data(data)

        _release_xarray_file_handles()
        with h5py.File(self.file_path, "a") as h5:
            if self.path in h5 and not isinstance(h5[self.path], h5py.Group):
                raise TypeError(f"Parent path {self.path!r} exists but is not a group.")

            group = h5.require_group(self.path)

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
                prepared,
                dtype,
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

        child_path = f"{self.path}/{child_name}"

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
            return f"{self.path}/{child_name}" in h5

    def delete(self, key: str) -> None:
        """
        Delete a direct child from this raw HDF5 group.

        SlothPy semantic groups are protected from raw mutation through this API.
        """
        if self.exists and self.is_slothpy:
            raise TypeError(
                f"Group {self.path!r} is a SlothPy semantic xarray group. "
                "Delete the whole group from the SltFile level instead."
            )

        child_name = _normalize_hdf5_path(key)

        if "/" in child_name:
            raise ValueError(
                "SltGroup deletion supports only direct children. "
                "Use h5py for deeper custom layouts."
            )

        child_path = f"{self.path}/{child_name}"

        _release_xarray_file_handles()
        with h5py.File(self.file_path, "r+") as h5:
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
            return list(dataset.data_vars)
        finally:
            dataset.close()

    def coordinates(self) -> list[str]:
        """
        Return names of xarray coordinates in this SlothPy semantic group.
        """
        dataset = self.to_dataset()
        try:
            return list(dataset.coords)
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
                return list(dataset.data_vars) + [
                    name for name in dataset.coords if name not in dataset.data_vars
                ]
            finally:
                dataset.close()

        with h5py.File(self.file_path, "r") as h5:
            item = _get_hdf5_item(h5, self.path)
            if not isinstance(item, h5py.Group):
                raise TypeError(f"{self.path!r} is not a group.")
            return list(item.keys())

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

        name = array.name or self.primary or "value"
        return array.to_dataframe(*args, **kwargs, name=name)

    def to_node(self) -> SltGroupNode:
        """
        Return this group as a structured node.
        """
        return _group_node(self.file_path, self.path)

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
                (f" {self.path!r}", "blue"),
                (" in "),
                (str(self.file_path), "green"),
                (" does not exist."),
            )
            return _rich_to_html(text)

        return _rich_to_html(_group_tree(self.to_node()))

    def __repr__(self) -> str:
        return (
            f"SltGroup(file_path={self.file_path!s}, "
            f"path={self.path!r}, chunks={self.chunks!r})"
        )

    def __str__(self) -> str:
        if not self.exists:
            text = Text.assemble(
                ("Proxy group", "bold blue"),
                (f" {self.path!r}", "blue"),
                (" in "),
                (str(self.file_path), "green"),
                (" does not exist."),
            )
            return _rich_to_ansi(text)

        return _rich_to_ansi(_group_tree(self.to_node()))


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

        _release_xarray_file_handles()
        with h5py.File(file_path, mode) as h5:
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
        """
        if any(flag in mode for flag in ("+", "a", "w", "x")):
            _release_xarray_file_handles()
        return h5py.File(self.path, mode)

    def group(
        self,
        path: str,
        *,
        chunks: XarrayChunks = None,
        require_exists: bool = True,
    ) -> SltGroup:
        """
        Return a group handle.
        """
        group = SltGroup(self.path, path, chunks=chunks)

        if require_exists and not group.exists:
            raise KeyError(
                f"No group {path!r} in {self.path!s}. Available groups: {self.groups()}"
            )

        return group

    def create_group(self, path: str, **attrs: Any) -> SltGroup:
        """
        Create or require a raw HDF5 group and optionally assign attributes.
        """
        group = self.group(path, require_exists=False)
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
        prepared, dtype = _prepare_dataset_data(data)

        _release_xarray_file_handles()
        with h5py.File(self.path, "a") as h5:
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
                prepared,
                dtype,
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
        dataset: xr.Dataset | xr.DataArray,
        *,
        overwrite: bool = False,
        primary: str | None = None,
        slt_type: str | None = None,
        encoding: dict[str, Any] | None = None,
        invalid_netcdf: bool = True,
    ) -> SltGroup:
        """
        Internal helper for SlothPy computations/readers.

        Write a valid SlothPy semantic xarray group. This is intentionally
        private: user code should normally not create SlothPy semantic groups
        manually.
        """
        group_name = _normalize_hdf5_path(name)

        if "/" in group_name:
            raise ValueError(
                "SlothPy semantic groups must be root-level groups. "
                "Use h5py for custom nested user layouts."
            )

        dataset_to_write = _coerce_to_dataset(dataset)
        dataset_to_write = _dataset_with_slothpy_attrs(
            dataset_to_write,
            primary=primary,
            slt_type=slt_type,
        )

        _release_xarray_file_handles()
        with h5py.File(self.path, "a") as h5:
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

        _release_xarray_file_handles()
        dataset_to_write.to_netcdf(self.path, **kwargs)

        return SltGroup(self.path, group_name)

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

        _release_xarray_file_handles()
        with h5py.File(self.path, "r+") as h5:
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
        return {key: self[key] for key in self.keys()}  # type: ignore[dict-item]

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


# ---------------------------------------------------------------------------
# Public factory functions
# ---------------------------------------------------------------------------


def create_slt_file(path: PathLike, *, overwrite: bool = False) -> SltFile:
    """
    Create a new xarray-backed ``.slt`` file and return an ``SltFile`` handle.
    """
    return SltFile._create(path, overwrite=overwrite)


def open_slt_file(path: PathLike) -> SltFile:
    """
    Open an existing ``.slt`` file and return an ``SltFile`` handle.
    """
    return SltFile._new(path)


def slt_file(path: PathLike) -> SltFile:
    """
    Backward-compatible alias for opening an existing ``.slt`` file.
    """
    return open_slt_file(path)


__all__ = [
    "SltAttributes",
    "SltDataset",
    "SltDatasetNode",
    "SltFile",
    "SltFileNode",
    "SltGroup",
    "SltGroupNode",
    "SltNodeInfo",
    "SltVariableNode",
    "create_slt_file",
    "open_slt_file",
    "slt_file",
]
