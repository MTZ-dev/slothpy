from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from io import StringIO
from pathlib import Path, PurePosixPath
from typing import Any, Literal

import h5py
import numpy as np
import xarray as xr
from rich.console import Console
from rich.text import Text
from rich.tree import Tree

from slothpy import __version__
from slothpy.types.aliases import PathLike

SLOTHPY_FORMAT_VERSION = __version__
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
    console = Console(
        file=StringIO(),
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

    ``"triplets/energies"`` becomes ``"triplets/energies"``.

    ``("triplets", "energies")`` becomes ``"triplets/energies"``.
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
        (None, name) for root-level objects,
        (group, dataset) for datasets inside root-level groups.
    """
    parts = path.split("/")

    if len(parts) == 1:
        return None, parts[0]

    if len(parts) == 2:
        return parts[0], parts[1]

    raise ValueError(f"HDF5 path {path!r} is too deep for SlothPy's convenience API.")


def _xarray_group_path(path: str) -> str:
    """
    Return an xarray/netCDF group path with leading slash.
    """
    return f"/{path}"


def _resolve_file_path(path: PathLike) -> Path:
    """
    Return an absolute, expanded path.

    The path does not need to exist. strict=False is intentionally used
    because new .slt files may be opened in creation modes.
    """
    return Path(path).expanduser().resolve(strict=False)


# ---------------------------------------------------------------------------
# HDF5 dataset helpers
# ---------------------------------------------------------------------------


def _coerce_hdf5_dataset_data(value: Any) -> Any:
    """
    Coerce data for HDF5 dataset creation.

    Most values are passed directly to h5py, which already handles Python
    scalars, lists, tuples, bytes, strings, and NumPy arrays.

    The only special case is an existing NumPy fixed-width Unicode array,
    dtype kind ``"U"``, because h5py cannot write NumPy Unicode arrays
    directly. With NumPy >= 2.0 and h5py >= 3.14, we convert it to NumPy's
    native variable-width string dtype.
    """
    if isinstance(value, np.ndarray) and value.dtype.kind == "U":
        return value.astype("T")

    return value


def _is_scalar_dataset_data(data: Any) -> bool:
    """
    Return True if the dataset data represents an HDF5 scalar dataset.

    Scalar datasets cannot be chunked or compressed.

    This function intentionally avoids converting large Python sequences with
    ``np.asarray`` just to inspect their shape.
    """
    if isinstance(data, np.ndarray):
        return data.shape == ()

    if isinstance(data, np.generic):
        return True

    if isinstance(data, str | bytes):
        return True

    if isinstance(data, list | tuple | range):
        return False

    if np.isscalar(data):
        return True

    try:
        return np.asarray(data).shape == ()
    except Exception:
        # Let h5py raise the real error during dataset creation.
        return False


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
    *,
    chunks: bool | tuple[int, ...] | None,
    compression: str | None,
) -> h5py.Dataset:
    """
    Create an HDF5 dataset, avoiding chunk/compression options for scalar data.
    """
    data = _coerce_hdf5_dataset_data(data)

    kwargs: dict[str, Any] = {"data": data}

    if not _is_scalar_dataset_data(data):
        if chunks is not None:
            kwargs["chunks"] = chunks
        if compression is not None:
            kwargs["compression"] = compression

    return parent.create_dataset(name, **kwargs)


def _get_hdf5_item(h5: h5py.File, path: str) -> h5py.File | h5py.Group | h5py.Dataset:
    """
    Return an HDF5 item or raise a clear ``KeyError``.
    """
    if path == "/":
        return h5

    path = _normalize_hdf5_path(path)

    if path not in h5:
        raise KeyError(f"No item {path!r} exists in file {h5.filename!s}.")

    item = h5[path]

    if isinstance(item, h5py.Group | h5py.Dataset):
        return item

    raise TypeError(f"Unsupported HDF5 object at path {path!r}.")


# ---------------------------------------------------------------------------
# xarray file-cache and HDF5 opening helpers
# ---------------------------------------------------------------------------


def _as_resolved_path(value: Any) -> Path | None:
    """
    Try to interpret a value as an absolute filesystem path.

    Returns ``None`` if the value is not path-like.
    """
    if isinstance(value, Path):
        try:
            return value.expanduser().resolve(strict=False)
        except OSError:
            return None

    if isinstance(value, str):
        try:
            return Path(value).expanduser().resolve(strict=False)
        except OSError:
            return None

    return None


def _iter_nested_values(value: Any) -> Iterator[Any]:
    """
    Recursively yield values from common Python containers.

    xarray's file-cache keys may contain nested tuples/dicts with the filename
    somewhere inside the opener arguments.
    """
    yield value

    if isinstance(value, dict):
        for key, item in value.items():
            yield from _iter_nested_values(key)
            yield from _iter_nested_values(item)
        return

    if isinstance(value, tuple | list | set | frozenset):
        for item in value:
            yield from _iter_nested_values(item)


def _cached_file_matches_path(
    *,
    cache_key: Any,
    cached_file: Any,
    target_path: Path,
) -> bool:
    """
    Return True if an xarray FILE_CACHE entry appears to belong to target_path.

    The function checks both the cache key and common filename-like attributes
    on cached backend file objects.
    """
    target_path = _resolve_file_path(target_path)

    for value in _iter_nested_values(cache_key):
        candidate = _as_resolved_path(value)
        if candidate == target_path:
            return True

    filename_attrs = (
        "filename",
        "filepath",
        "path",
        "_filename",
        "_filepath",
        "_path",
    )

    for attr in filename_attrs:
        try:
            value = getattr(cached_file, attr)
        except AttributeError:
            continue
        except Exception:
            continue

        if callable(value):
            try:
                value = value()
            except Exception:
                continue

        candidate = _as_resolved_path(value)
        if candidate == target_path:
            return True

    return False


def _close_cached_file(cached_file: Any) -> None:
    """
    Best-effort close for an xarray cached backend file object.
    """
    close = getattr(cached_file, "close", None)
    if callable(close):
        close()


def _release_xarray_file_handles(file_path: PathLike | None = None) -> None:
    """
    Release xarray backend file handles.

    Parameters
    ----------
    file_path
        If provided, release only cached xarray file handles that appear to
        belong to this file. If ``None``, release all cached file handles.

    Notes
    -----
    This is mainly needed in notebooks, where lazy xarray objects or displayed
    outputs can keep read-only backend handles alive.

    The path-targeted mode is safer than clearing xarray's whole global file
    cache. Still, do not release handles while a Dask computation is actively
    reading from the same file.
    """
    try:
        from xarray.backends.file_manager import FILE_CACHE
    except Exception:
        return

    if file_path is None:
        FILE_CACHE.clear()
        return

    target_path = _resolve_file_path(file_path)

    try:
        cache_items = list(FILE_CACHE.items())
    except Exception:
        FILE_CACHE.clear()
        return

    for cache_key, cached_file in cache_items:
        if not _cached_file_matches_path(
            cache_key=cache_key,
            cached_file=cached_file,
            target_path=target_path,
        ):
            continue

        try:
            del FILE_CACHE[cache_key]
        except Exception:
            pass

        _close_cached_file(cached_file)


def _hdf5_mode_requests_write(mode: str) -> bool:
    """
    Return True if an HDF5 open mode can write, create, or truncate.
    """
    return any(flag in mode for flag in ("+", "a", "w", "x"))


def _is_xarray_read_only_conflict(exc: OSError) -> bool:
    """
    Return True if an HDF5 open error likely comes from a cached xarray reader.
    """
    message = str(exc).lower()
    return (
        "already open for read-only" in message
        or "file is already open for read-only" in message
    )


def _open_hdf5_handle(file_path: PathLike, mode: str) -> h5py.File:
    """
    Open an HDF5 file.

    For write-capable modes, retry once after releasing xarray cached handles
    for the same file if HDF5 reports a read-only open-handle conflict.
    """
    path = _resolve_file_path(file_path)

    try:
        return h5py.File(path, mode)
    except OSError as exc:
        if not _hdf5_mode_requests_write(mode):
            raise

        if not _is_xarray_read_only_conflict(exc):
            raise

    _release_xarray_file_handles(path)
    return h5py.File(path, mode)


@contextmanager
def _open_hdf5_file(file_path: PathLike, mode: str) -> Iterator[h5py.File]:
    """
    Context-manager wrapper around ``_open_hdf5_handle``.
    """
    h5 = _open_hdf5_handle(file_path, mode)
    try:
        yield h5
    finally:
        h5.close()


def _write_xarray_to_netcdf_with_retry(
    dataset: xr.Dataset,
    file_path: PathLike,
    **kwargs: Any,
) -> None:
    """
    Write an xarray Dataset to netCDF/HDF5.

    If writing fails because xarray still has a read-only cached handle for the
    same file, release handles for that file and retry once.
    """
    path = _resolve_file_path(file_path)

    try:
        dataset.to_netcdf(path, **kwargs)
        return
    except OSError as exc:
        if not _is_xarray_read_only_conflict(exc):
            raise

    _release_xarray_file_handles(path)
    dataset.to_netcdf(path, **kwargs)


# ---------------------------------------------------------------------------
# SlothPy/xarray helpers
# ---------------------------------------------------------------------------


def _attrs_mark_slothpy_group(attrs: dict[str, Any]) -> bool:
    value = attrs.get("slt_valid")

    if isinstance(value, bytes):
        value = value.decode("utf-8")

    return isinstance(value, str) and value.strip().lower() == "true"


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
                str(coord_name),
                dataset.coords[coord_name],
                kind="coordinate",
                primary=primary,
            )
            for coord_name in dataset.coords
        )

        data_variables = tuple(
            _variable_node(
                str(var_name),
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
        text.append(f" [version={version}]", style="yellow")

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
            text.append(f" [Type={slt_type}]", style="yellow")

        if node.primary is not None:
            text.append(f" [Primary={node.primary}]", style="yellow")
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


def _add_group_attrs_branch(group_tree: Tree, node: SltGroupNode) -> None:
    """
    Append an ``Attributes`` subtree (HDF5 / xarray group metadata).
    """
    attrs_tree = group_tree.add(Text("Attributes", style="bold yellow"))
    if node.attrs:
        attrs_tree.add(_attrs_label(node.attrs))
    else:
        attrs_tree.add(Text("(none)", style="bright_black"))


def _add_group_to_tree(parent: Tree, node: SltGroupNode) -> None:
    """
    Add a group node to a Rich tree.
    """
    group_tree = parent.add(_group_label(node))
    _add_group_attrs_branch(group_tree, node)

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
    _add_group_attrs_branch(tree, node)

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
