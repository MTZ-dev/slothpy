from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, MutableMapping
from dataclasses import asdict, dataclass, field, replace
from inspect import signature
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from numpy.typing import DTypeLike

from slothpy.types.aliases import ArrayOrder, PathLike

_SHARED_MEMORY_SUPPORTS_TRACK = "track" in signature(SharedMemory).parameters
_MANIFEST_VERSION = 1


def _open_shared_memory(
    *,
    name: str | None = None,
    create: bool = False,
    size: int = 0,
    track: bool = True,
) -> SharedMemory:
    kwargs: dict[str, Any] = {
        "name": name,
        "create": create,
        "size": size,
    }

    if _SHARED_MEMORY_SUPPORTS_TRACK:
        kwargs["track"] = track

    return SharedMemory(**kwargs)


def _dtype_to_string(dtype: DTypeLike) -> str:
    return np.dtype(dtype).str


def _array_nbytes(shape: tuple[int, ...], dtype: DTypeLike) -> int:
    size = int(np.prod(shape, dtype=np.int64)) if shape else 1
    return size * np.dtype(dtype).itemsize


def _shared_memory_size(nbytes: int) -> int:
    # multiprocessing.shared_memory.SharedMemory rejects size=0.
    return max(int(nbytes), 1)


def _normalise_shape(shape: tuple[int, ...] | list[int]) -> tuple[int, ...]:
    return tuple(int(size) for size in shape)


def _normalise_hdf5_selection(
    selection: Any,
    *,
    ndim: int,
) -> tuple[Any, ...]:
    if selection is None:
        return tuple(slice(None) for _ in range(ndim))

    if not isinstance(selection, tuple):
        selection = (selection,)

    result: list[Any] = []
    ellipsis_seen = False

    for item in selection:
        if item is Ellipsis:
            if ellipsis_seen:
                raise ValueError("Only one ellipsis is allowed in an HDF5 selection.")
            ellipsis_seen = True
            missing = ndim - (len(selection) - 1)
            result.extend(slice(None) for _ in range(missing))
            continue

        result.append(item)

    while len(result) < ndim:
        result.append(slice(None))

    if len(result) != ndim:
        raise ValueError(
            f"HDF5 selection has {len(result)} entries, but dataset has {ndim} dimensions."
        )

    return tuple(result)


def _slice_length(item: slice, dim_size: int) -> int:
    start, stop, step = item.indices(dim_size)
    return len(range(start, stop, step))


def _selection_shape(
    dataset_shape: tuple[int, ...],
    selection: Any,
) -> tuple[int, ...]:
    normalised = _normalise_hdf5_selection(selection, ndim=len(dataset_shape))

    result: list[int] = []

    for item, dim_size in zip(normalised, dataset_shape, strict=True):
        if isinstance(item, slice):
            result.append(_slice_length(item, dim_size))
            continue

        if isinstance(item, int | np.integer):
            index = int(item)
            if index < -dim_size or index >= dim_size:
                raise IndexError(
                    f"HDF5 integer selection index {index} is out of bounds "
                    f"for dimension of size {dim_size}."
                )
            continue

        raise TypeError(
            "Only integers, slices, and ellipsis are supported for direct HDF5 "
            f"shared-memory reads. Got selection item {item!r}."
        )

    return tuple(result)


@dataclass(frozen=True, slots=True)
class SharedArraySpec:
    """
    Serializable description of a NumPy array stored in shared memory.

    This object is safe to pass through JSON, command-line arguments, temporary
    manifest files, MPI messages, or environment variables.
    """

    name: str
    shape: tuple[int, ...]
    dtype: str
    order: ArrayOrder
    nbytes: int
    readonly: bool = False

    @classmethod
    def from_array_metadata(
        cls,
        *,
        name: str,
        shape: tuple[int, ...],
        dtype: DTypeLike,
        order: ArrayOrder = "C",
        readonly: bool = False,
    ) -> SharedArraySpec:
        dtype_string = _dtype_to_string(dtype)
        shape = _normalise_shape(shape)
        return cls(
            name=name,
            shape=shape,
            dtype=dtype_string,
            order=order,
            nbytes=_array_nbytes(shape, dtype_string),
            readonly=readonly,
        )

    def to_json_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["shape"] = list(self.shape)
        return data

    @classmethod
    def from_json_dict(cls, data: Mapping[str, Any]) -> SharedArraySpec:
        return cls(
            name=str(data["name"]),
            shape=tuple(int(size) for size in data["shape"]),
            dtype=str(data["dtype"]),
            order=data.get("order", "C"),
            nbytes=int(data["nbytes"]),
            readonly=bool(data.get("readonly", False)),
        )


@dataclass(slots=True)
class SharedNumpyArray:
    """
    NumPy ndarray backed by multiprocessing shared memory.

    Parent process:
        creates arrays and owns unlinking.

    MPI worker rank 0:
        attaches to arrays using SharedArraySpec and only closes them.
    """

    spec: SharedArraySpec
    _shm: SharedMemory
    _array: np.ndarray
    _owns_memory: bool = False
    _closed: bool = False
    _unlinked: bool = False

    @property
    def array(self) -> np.ndarray:
        if self._closed:
            raise RuntimeError(f"Shared memory block {self.spec.name!r} is closed.")
        return self._array

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def shape(self) -> tuple[int, ...]:
        return self.spec.shape

    @property
    def dtype(self) -> np.dtype[Any]:
        return np.dtype(self.spec.dtype)

    @property
    def nbytes(self) -> int:
        return self.spec.nbytes

    @classmethod
    def empty(
        cls,
        shape: tuple[int, ...] | list[int],
        *,
        dtype: DTypeLike,
        order: ArrayOrder = "C",
        readonly: bool = False,
        name: str | None = None,
        track: bool = True,
    ) -> SharedNumpyArray:
        shape = _normalise_shape(shape)
        dtype_obj = np.dtype(dtype)
        nbytes = _array_nbytes(shape, dtype_obj)

        shm = _open_shared_memory(
            name=name,
            create=True,
            size=_shared_memory_size(nbytes),
            track=track,
        )

        spec = SharedArraySpec.from_array_metadata(
            name=shm.name,
            shape=shape,
            dtype=dtype_obj,
            order=order,
            readonly=readonly,
        )

        array = np.ndarray(
            shape,
            dtype=dtype_obj,
            buffer=shm.buf,
            order=order,
        )
        array.setflags(write=not readonly)

        return cls(
            spec=spec,
            _shm=shm,
            _array=array,
            _owns_memory=True,
        )

    @classmethod
    def from_array(
        cls,
        array: np.ndarray,
        *,
        dtype: DTypeLike | None = None,
        order: ArrayOrder = "C",
        readonly: bool = True,
        name: str | None = None,
        track: bool = True,
    ) -> SharedNumpyArray:
        dtype_obj = np.dtype(array.dtype if dtype is None else dtype)

        shared = cls.empty(
            array.shape,
            dtype=dtype_obj,
            order=order,
            readonly=False,
            name=name,
            track=track,
        )

        shared.array[...] = np.asarray(array, dtype=dtype_obj, order=order)

        if readonly:
            shared.set_readonly(True)

        return shared

    @classmethod
    def from_hdf5_dataset(
        cls,
        file_path: PathLike,
        dataset_path: str,
        *,
        dtype: DTypeLike | None = None,
        source_sel: Any = None,
        shape: tuple[int, ...] | None = None,
        order: ArrayOrder = "C",
        readonly: bool = True,
        name: str | None = None,
        track: bool = True,
    ) -> SharedNumpyArray:
        """
        Create a shared array and read an HDF5 dataset directly into it.

        This avoids creating a full intermediate NumPy array in the parent process.
        Internally it uses h5py.Dataset.read_direct(shared_array).

        For selections, simple integers/slices/ellipsis are supported. More
        advanced h5py selections should pass explicit ``shape``.
        """
        if order != "C":
            raise ValueError(
                "Direct HDF5 read into shared memory currently requires order='C'. "
                "Use order='C' here and convert later only if a downstream LAPACK "
                "routine truly needs Fortran layout."
            )

        with h5py.File(file_path, "r") as h5:
            dataset = h5[dataset_path]
            if not isinstance(dataset, h5py.Dataset):
                raise TypeError(f"HDF5 path {dataset_path!r} is not a dataset.")

            dtype_obj = np.dtype(dataset.dtype if dtype is None else dtype)

            if shape is None:
                if source_sel is None:
                    shape = tuple(int(size) for size in dataset.shape)
                else:
                    shape = _selection_shape(
                        tuple(int(size) for size in dataset.shape),
                        source_sel,
                    )

            shared = cls.empty(
                shape,
                dtype=dtype_obj,
                order="C",
                readonly=False,
                name=name,
                track=track,
            )

            if source_sel is None:
                dataset.read_direct(shared.array)
            else:
                normalised_sel = _normalise_hdf5_selection(
                    source_sel,
                    ndim=dataset.ndim,
                )
                dataset.read_direct(shared.array, source_sel=normalised_sel)

        if readonly:
            shared.set_readonly(True)

        return shared

    @classmethod
    def attach(
        cls,
        spec: SharedArraySpec,
        *,
        readonly: bool | None = None,
        track: bool = False,
    ) -> SharedNumpyArray:
        """
        Attach to an existing shared-memory array.

        For MPI rank 0 launched outside multiprocessing, ``track=False`` prevents
        the worker process resource tracker from unlinking memory owned by the
        parent process.
        """
        shm = _open_shared_memory(
            name=spec.name,
            create=False,
            size=0,
            track=track,
        )

        if shm.size < spec.nbytes:
            shm.close()
            raise ValueError(
                f"Shared memory block {spec.name!r} has size {shm.size}, "
                f"but spec requires at least {spec.nbytes} bytes."
            )

        array = np.ndarray(
            spec.shape,
            dtype=np.dtype(spec.dtype),
            buffer=shm.buf,
            order=spec.order,
        )

        effective_readonly = spec.readonly if readonly is None else readonly
        array.setflags(write=not effective_readonly)

        if effective_readonly != spec.readonly:
            spec = replace(spec, readonly=effective_readonly)

        return cls(
            spec=spec,
            _shm=shm,
            _array=array,
            _owns_memory=False,
        )

    def set_readonly(self, readonly: bool = True) -> None:
        self.array.setflags(write=not readonly)
        self.spec = replace(self.spec, readonly=readonly)

    def copy(self) -> np.ndarray:
        return np.array(self.array, copy=True, order="C")

    def close(self) -> None:
        if self._closed:
            return

        self._closed = True

        # Drop ndarray view before closing the mmap.
        self._array = np.ndarray((0,), dtype=np.uint8)

        self._shm.close()

    def unlink(self) -> None:
        if self._unlinked:
            return

        if not self._owns_memory:
            raise RuntimeError(
                f"Shared memory block {self.spec.name!r} is not owned by this process."
            )

        self._unlinked = True
        self._shm.unlink()

    def release(self) -> None:
        """
        Close and, if owned, unlink this shared-memory block.
        """
        if self._owns_memory and not self._unlinked:
            self.unlink()
        self.close()

    def __enter__(self) -> SharedNumpyArray:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.release() if self._owns_memory else self.close()


@dataclass(frozen=True, slots=True)
class Hdf5SharedArrayRequest:
    """
    Request describing one HDF5 dataset that should be staged into shared memory.
    """

    dataset_path: str
    dtype: str | None = None
    source_sel: Any = None
    shape: tuple[int, ...] | None = None
    readonly: bool = True


@dataclass(slots=True)
class SharedArrayBundle(MutableMapping[str, SharedNumpyArray]):
    """
    Named collection of shared arrays.

    This is used as the object passed around by parent-side computation setup.
    Rank 0 reconstructs an attached bundle from the JSON manifest.
    """

    arrays: dict[str, SharedNumpyArray] = field(default_factory=dict)
    _owns_memory: bool = True

    def __getitem__(self, key: str) -> SharedNumpyArray:
        return self.arrays[key]

    def __setitem__(self, key: str, value: SharedNumpyArray) -> None:
        self.arrays[key] = value

    def __delitem__(self, key: str) -> None:
        array = self.arrays.pop(key)
        array.release() if array._owns_memory else array.close()

    def __iter__(self) -> Iterator[str]:
        return iter(self.arrays)

    def __len__(self) -> int:
        return len(self.arrays)

    def add(self, key: str, array: SharedNumpyArray) -> SharedNumpyArray:
        if key in self.arrays:
            raise KeyError(f"Shared array key {key!r} already exists.")
        self.arrays[key] = array
        return array

    def add_empty(
        self,
        key: str,
        shape: tuple[int, ...] | list[int],
        *,
        dtype: DTypeLike,
        order: ArrayOrder = "C",
        readonly: bool = False,
        track: bool = True,
    ) -> SharedNumpyArray:
        return self.add(
            key,
            SharedNumpyArray.empty(
                shape,
                dtype=dtype,
                order=order,
                readonly=readonly,
                track=track,
            ),
        )

    def add_array(
        self,
        key: str,
        array: np.ndarray,
        *,
        dtype: DTypeLike | None = None,
        order: ArrayOrder = "C",
        readonly: bool = True,
        track: bool = True,
    ) -> SharedNumpyArray:
        return self.add(
            key,
            SharedNumpyArray.from_array(
                array,
                dtype=dtype,
                order=order,
                readonly=readonly,
                track=track,
            ),
        )

    def add_hdf5_dataset(
        self,
        key: str,
        file_path: PathLike,
        dataset_path: str,
        *,
        dtype: DTypeLike | None = None,
        source_sel: Any = None,
        shape: tuple[int, ...] | None = None,
        readonly: bool = True,
        track: bool = True,
    ) -> SharedNumpyArray:
        return self.add(
            key,
            SharedNumpyArray.from_hdf5_dataset(
                file_path,
                dataset_path,
                dtype=dtype,
                source_sel=source_sel,
                shape=shape,
                readonly=readonly,
                track=track,
            ),
        )

    def specs(self) -> dict[str, SharedArraySpec]:
        return {key: array.spec for key, array in self.arrays.items()}

    def to_manifest_dict(self) -> dict[str, Any]:
        return {
            "version": _MANIFEST_VERSION,
            "arrays": {key: spec.to_json_dict() for key, spec in self.specs().items()},
        }

    @classmethod
    def from_manifest_dict(
        cls,
        manifest: Mapping[str, Any],
        *,
        track: bool = False,
    ) -> SharedArrayBundle:
        version = int(manifest.get("version", -1))
        if version != _MANIFEST_VERSION:
            raise ValueError(
                f"Unsupported shared-memory manifest version {version}; "
                f"expected {_MANIFEST_VERSION}."
            )

        bundle = cls(_owns_memory=False)

        arrays = manifest.get("arrays")
        if not isinstance(arrays, Mapping):
            raise TypeError("Shared-memory manifest field 'arrays' must be a mapping.")

        for key, raw_spec in arrays.items():
            spec = SharedArraySpec.from_json_dict(raw_spec)
            bundle.add(str(key), SharedNumpyArray.attach(spec, track=track))

        return bundle

    def write_manifest(self, path: PathLike) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        temporary_path = path.with_suffix(path.suffix + ".tmp")
        temporary_path.write_text(
            json.dumps(self.to_manifest_dict(), indent=2),
            encoding="utf-8",
        )
        temporary_path.replace(path)

        return path

    @classmethod
    def attach_from_manifest(
        cls,
        path: PathLike,
        *,
        track: bool = False,
    ) -> SharedArrayBundle:
        manifest = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_manifest_dict(manifest, track=track)

    @classmethod
    def stage_hdf5_datasets(
        cls,
        file_path: PathLike,
        requests: Mapping[str, str | Hdf5SharedArrayRequest],
        *,
        track: bool = True,
    ) -> SharedArrayBundle:
        """
        Create a bundle by reading several HDF5 datasets directly into shared memory.

        Example
        -------
        bundle = SharedArrayBundle.stage_hdf5_datasets(
            "hamiltonian.slt",
            {
                "energies": "hamiltonian/states_energies",
                "spin": Hdf5SharedArrayRequest("hamiltonian/spin_matrices"),
            },
        )
        """
        bundle = cls()

        for key, request in requests.items():
            if isinstance(request, str):
                request = Hdf5SharedArrayRequest(dataset_path=request)

            bundle.add_hdf5_dataset(
                key,
                file_path,
                request.dataset_path,
                dtype=request.dtype,
                source_sel=request.source_sel,
                shape=request.shape,
                readonly=request.readonly,
                track=track,
            )

        return bundle

    def close(self) -> None:
        for array in self.arrays.values():
            array.close()

    def unlink(self) -> None:
        for array in self.arrays.values():
            if array._owns_memory:
                array.unlink()

    def release(self) -> None:
        for array in self.arrays.values():
            array.release() if array._owns_memory else array.close()

    def copies(self) -> dict[str, np.ndarray]:
        return {key: array.copy() for key, array in self.arrays.items()}

    def __enter__(self) -> SharedArrayBundle:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.release()
