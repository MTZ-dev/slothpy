from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest
import xarray as xr

import slothpy.core.slt as slt_mod
from slothpy.core.slt import (
    SltDataset,
    SltFile,
    SltGroup,
    create_slt_file,
    open_slt_file,
    slt_file,
)

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def slt_path(tmp_path: Path) -> Path:
    return tmp_path / "demo.slt"


@pytest.fixture
def slt(slt_path: Path) -> SltFile:
    return create_slt_file(slt_path, overwrite=True)


@pytest.fixture
def semantic_dataset() -> xr.Dataset:
    fields = np.linspace(0.0, 7.0, 8)
    temperatures = np.array([2.0, 5.0, 10.0])
    orientations = np.arange(4)

    data = np.arange(
        fields.size * temperatures.size * orientations.size,
        dtype=np.float64,
    ).reshape(fields.size, temperatures.size, orientations.size)

    return xr.Dataset(
        data_vars={
            "magnetisation": (
                ("field", "temperature", "orientation"),
                data,
                {"unit": "mu_B", "long_name": "magnetisation"},
            ),
            "orientation_weight": (
                ("orientation",),
                np.ones(orientations.size) / orientations.size,
                {"long_name": "powder weight"},
            ),
        },
        coords={
            "field": ("field", fields, {"unit": "T"}),
            "temperature": ("temperature", temperatures, {"unit": "K"}),
            "orientation": orientations,
        },
        attrs={"title": "demo semantic dataset"},
    )


@pytest.fixture
def slt_with_semantic_group(
    slt: SltFile,
    semantic_dataset: xr.Dataset,
) -> SltFile:
    slt._write_slothpy_group(
        "magnetisation_001",
        semantic_dataset,
        slt_type="MAGNETISATION",
        primary="magnetisation",
        overwrite=True,
    )
    return slt


def _safe_close(obj: Any) -> None:
    close = getattr(obj, "close", None)
    if callable(close):
        close()


# ---------------------------------------------------------------------------
# Path and low-level helper tests
# ---------------------------------------------------------------------------


def test_normalize_slt_path_adds_suffix(tmp_path: Path) -> None:
    assert slt_mod._normalize_slt_path(tmp_path / "demo").name == "demo.slt"


def test_normalize_slt_path_replaces_suffix(tmp_path: Path) -> None:
    assert slt_mod._normalize_slt_path(tmp_path / "demo.h5").name == "demo.slt"


def test_normalize_hdf5_path_strips_slashes() -> None:
    assert slt_mod._normalize_hdf5_path("/group/dataset/") == "group/dataset"


@pytest.mark.parametrize("path", ["", "/", ".", "///"])
def test_normalize_hdf5_path_rejects_empty_paths(path: str) -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        slt_mod._normalize_hdf5_path(path)


def test_path_from_tuple_key() -> None:
    assert slt_mod._path_from_key(("group", "dataset")) == "group/dataset"


@pytest.mark.parametrize(
    "key",
    [
        ("group",),
        ("group", "dataset", "extra"),
    ],
)
def test_path_from_tuple_key_rejects_wrong_tuple_length(key: tuple[str, ...]) -> None:
    with pytest.raises(KeyError):
        slt_mod._path_from_key(key)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "key",
    [
        ("group/subgroup", "dataset"),
        ("group", "subgroup/dataset"),
    ],
)
def test_path_from_tuple_key_rejects_nested_tuple_parts(
    key: tuple[str, str],
) -> None:
    with pytest.raises(ValueError):
        slt_mod._path_from_key(key)


def test_path_from_string_key() -> None:
    assert slt_mod._path_from_key("group/dataset") == "group/dataset"


def test_path_from_key_rejects_too_deep_path() -> None:
    with pytest.raises(ValueError, match="too deep"):
        slt_mod._path_from_key("a/b/c")


def test_split_supported_path_root_dataset() -> None:
    assert slt_mod._split_supported_path("dataset") == (None, "dataset")


def test_split_supported_path_group_dataset() -> None:
    assert slt_mod._split_supported_path("group/dataset") == ("group", "dataset")


def test_xarray_group_path_adds_leading_slash() -> None:
    assert slt_mod._xarray_group_path("group") == "/group"


def test_string_dtype_is_utf8() -> None:
    dtype = slt_mod._string_dtype()
    assert h5py.check_string_dtype(dtype).encoding == "utf-8"


def test_prepare_dataset_data_scalar_string() -> None:
    data, dtype = slt_mod._prepare_dataset_data("abc")
    assert data == "abc"
    assert h5py.check_string_dtype(dtype).encoding == "utf-8"


def test_prepare_dataset_data_string_list() -> None:
    data, dtype = slt_mod._prepare_dataset_data(["a", "b"])
    assert data.tolist() == ["a", "b"]
    assert h5py.check_string_dtype(dtype).encoding == "utf-8"


def test_prepare_dataset_data_string_numpy_array() -> None:
    data, dtype = slt_mod._prepare_dataset_data(np.array(["a", "b"]))
    assert data.tolist() == ["a", "b"]
    assert h5py.check_string_dtype(dtype).encoding == "utf-8"


def test_prepare_dataset_data_numeric_array() -> None:
    data, dtype = slt_mod._prepare_dataset_data([1, 2, 3])
    assert data.tolist() == [1, 2, 3]
    assert dtype is None


@pytest.mark.parametrize("value", ["abc", np.array(1), 1])
def test_is_scalar_dataset_data(value: Any) -> None:
    assert slt_mod._is_scalar_dataset_data(value)


@pytest.mark.parametrize("value", [[1, 2], np.array([1, 2])])
def test_is_scalar_dataset_data_false(value: Any) -> None:
    assert not slt_mod._is_scalar_dataset_data(value)


def test_get_hdf5_item_root_and_missing(slt: SltFile) -> None:
    with h5py.File(slt.path, "r") as h5:
        assert slt_mod._get_hdf5_item(h5, "/") is h5
        with pytest.raises(KeyError):
            slt_mod._get_hdf5_item(h5, "missing")


def test_get_hdf5_item_rejects_unsupported_object(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    file_path = tmp_path / "demo.slt"
    create_slt_file(file_path, overwrite=True)

    class FakeFile(dict):
        filename = "fake.slt"

    fake = FakeFile()
    fake["bad"] = object()

    with pytest.raises(TypeError, match="Unsupported HDF5 object"):
        slt_mod._get_hdf5_item(fake, "bad")  # type: ignore[arg-type]


def test_display_dtype_for_string_and_numeric(slt: SltFile) -> None:
    slt["strings"] = ["a", "b"]
    slt["numbers"] = [1, 2]

    with h5py.File(slt.path, "r") as h5:
        assert slt_mod._display_dtype(h5["strings"]) == "str"
        assert slt_mod._display_dtype(h5["numbers"]) == "int64"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (True, True),
        (False, False),
        ("true", True),
        ("yes", True),
        ("1", True),
        ("false", False),
        (1, True),
        (0, False),
        (np.int64(1), True),
        (b"true", True),
        (object(), False),
    ],
)
def test_truthy_attr(value: Any, expected: bool) -> None:
    assert slt_mod._truthy_attr(value) is expected


def test_attrs_mark_slothpy_group() -> None:
    assert slt_mod._attrs_mark_slothpy_group({"slt_valid": "true"})
    assert not slt_mod._attrs_mark_slothpy_group({})


def test_coerce_to_dataset_from_dataset_and_dataarray(
    semantic_dataset: xr.Dataset,
) -> None:
    assert slt_mod._coerce_to_dataset(semantic_dataset) is semantic_dataset

    data_array = semantic_dataset["magnetisation"]
    coerced = slt_mod._coerce_to_dataset(data_array)
    assert "magnetisation" in coerced.data_vars


def test_coerce_to_dataset_names_anonymous_dataarray() -> None:
    data_array = xr.DataArray(np.array([1, 2, 3]), dims=("x",))
    coerced = slt_mod._coerce_to_dataset(data_array, dataarray_name="value")
    assert "value" in coerced.data_vars


def test_coerce_to_dataset_rejects_non_xarray() -> None:
    with pytest.raises(TypeError):
        slt_mod._coerce_to_dataset([1, 2, 3])  # type: ignore[arg-type]


def test_primary_name_variants(semantic_dataset: xr.Dataset) -> None:
    semantic_dataset.attrs["slt_primary"] = b"magnetisation"
    assert slt_mod._primary_name(semantic_dataset) == "magnetisation"

    semantic_dataset.attrs["slt_primary"] = ""
    assert slt_mod._primary_name(semantic_dataset) is None

    del semantic_dataset.attrs["slt_primary"]
    assert slt_mod._primary_name(semantic_dataset) is None


def test_dataset_with_slothpy_attrs_single_variable_default_primary() -> None:
    ds = xr.Dataset({"x": ("i", [1, 2, 3])})
    result = slt_mod._dataset_with_slothpy_attrs(
        ds,
        primary=None,
        slt_type="TEST",
    )
    assert result.attrs["slt_valid"] == "true"
    assert result.attrs["slt_type"] == "TEST"
    assert result.attrs["slt_primary"] == "x"


def test_dataset_with_slothpy_attrs_multi_variable_dataset_primary() -> None:
    ds = xr.Dataset({"x": ("i", [1]), "y": ("i", [2])})
    result = slt_mod._dataset_with_slothpy_attrs(
        ds,
        primary=None,
        slt_type=None,
    )
    assert result.attrs["slt_primary"] == "__dataset__"


def test_dataset_with_slothpy_attrs_preserves_existing_primary() -> None:
    ds = xr.Dataset({"x": ("i", [1])}, attrs={"slt_primary": "__dataset__"})
    result = slt_mod._dataset_with_slothpy_attrs(
        ds,
        primary=None,
        slt_type=None,
    )
    assert result.attrs["slt_primary"] == "__dataset__"


def test_dataset_with_slothpy_attrs_rejects_missing_primary() -> None:
    ds = xr.Dataset({"x": ("i", [1])})
    with pytest.raises(KeyError):
        slt_mod._dataset_with_slothpy_attrs(
            ds,
            primary="missing",
            slt_type=None,
        )


def test_is_slothpy_group_false_for_missing_and_raw_group(slt: SltFile) -> None:
    slt.create_group("raw")
    assert not slt_mod._is_slothpy_group(slt.path, "missing")
    assert not slt_mod._is_slothpy_group(slt.path, "raw")


def test_release_xarray_file_handles_normal_and_import_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    slt_mod._release_xarray_file_handles()

    monkeypatch.setitem(sys.modules, "xarray.backends.file_manager", None)
    slt_mod._release_xarray_file_handles()


def test_rich_render_helpers() -> None:
    text = slt_mod.Text("hello", style="bold red")
    ansi = slt_mod._rich_to_ansi(text)
    html = slt_mod._rich_to_html(text)

    assert "hello" in ansi
    assert "hello" in html
    assert "<pre" in html


# ---------------------------------------------------------------------------
# File creation/opening tests
# ---------------------------------------------------------------------------


def test_sltfile_direct_instantiation_is_forbidden(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="should not be instantiated directly"):
        SltFile(tmp_path / "demo.slt")  # type: ignore[call-arg]


def test_create_slt_file_writes_root_metadata(slt_path: Path) -> None:
    slt = create_slt_file(slt_path, overwrite=True)

    assert slt.exists
    assert slt.attrs["format"] == "SlothPy"
    assert slt.attrs["format_version"] == slt_mod.SLOTHPY_FORMAT_VERSION
    assert slt.attrs["storage_model"] == slt_mod.SLOTHPY_STORAGE_MODEL


def test_create_slt_file_without_overwrite_rejects_existing_file(
    slt_path: Path,
) -> None:
    create_slt_file(slt_path, overwrite=True)

    with pytest.raises(FileExistsError):
        create_slt_file(slt_path, overwrite=False)


def test_open_slt_file_and_alias(slt_path: Path) -> None:
    created = create_slt_file(slt_path, overwrite=True)
    opened = open_slt_file(slt_path)
    opened_alias = slt_file(slt_path)

    assert opened.path == created.path
    assert opened_alias.path == created.path


def test_open_slt_file_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        open_slt_file(tmp_path / "missing.slt")


def test_create_slt_file_accepts_path_without_suffix(tmp_path: Path) -> None:
    slt = create_slt_file(tmp_path / "demo", overwrite=True)
    assert slt.path.name == "demo.slt"
    assert slt.path.exists()


# ---------------------------------------------------------------------------
# Attributes tests
# ---------------------------------------------------------------------------


def test_root_attributes_mapping_protocol(slt: SltFile) -> None:
    slt.attrs["author"] = "tester"
    slt.attrs["version"] = 1

    assert slt.attrs["author"] == "tester"
    assert "author" in slt.attrs
    assert len(slt.attrs) >= 2
    assert "author" in list(iter(slt.attrs))
    assert slt.attrs.as_dict()["version"] == 1

    del slt.attrs["author"]
    assert "author" not in slt.attrs


def test_attributes_repr_str_html(slt: SltFile) -> None:
    slt.attrs["a"] = "b"

    assert "SltAttributes" in repr(slt.attrs)
    assert "a" in str(slt.attrs)
    assert "a" in slt.attrs._repr_html_()


def test_attributes_show(slt: SltFile, capsys: pytest.CaptureFixture[str]) -> None:
    slt.attrs["a"] = "b"
    slt.attrs.show()
    captured = capsys.readouterr()
    assert "a" in captured.out


# ---------------------------------------------------------------------------
# Raw dataset tests
# ---------------------------------------------------------------------------


def test_create_root_dataset_and_read(slt: SltFile) -> None:
    dataset = slt.create_dataset("numbers", [1, 2, 3])

    assert isinstance(dataset, SltDataset)
    assert dataset.shape == (3,)
    assert dataset.dtype == "int64"
    assert dataset.read().tolist() == [1, 2, 3]
    assert dataset[:2].tolist() == [1, 2]
    assert dataset.to_numpy().tolist() == [1, 2, 3]
    assert np.asarray(dataset).tolist() == [1, 2, 3]
    assert np.asarray(dataset, dtype=np.float64).dtype == np.float64
    assert np.asarray(dataset, copy=True).tolist() == [1, 2, 3]


def test_create_scalar_dataset_ignores_chunking_and_compression(slt: SltFile) -> None:
    dataset = slt.create_dataset(
        "scalar",
        1,
        chunks=True,
        compression="gzip",
    )
    assert dataset.read() == 1


def test_create_string_datasets_and_read(slt: SltFile) -> None:
    slt["scalar_string"] = "abc"
    slt["string_array"] = ["a", "b", "c"]

    assert slt["scalar_string"].read() == "abc"  # type: ignore[union-attr]
    assert slt["string_array"].read().tolist() == ["a", "b", "c"]  # type: ignore[union-attr]


def test_dataset_write(slt: SltFile) -> None:
    dataset = slt.create_dataset("numbers", [1, 2, 3])
    dataset[1] = 20

    assert dataset.read().tolist() == [1, 20, 3]


def test_dataset_to_node_repr_str_html_show(
    slt: SltFile,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dataset = slt.create_dataset("numbers", [1, 2, 3])
    dataset.attrs["unit"] = "K"

    node = dataset.to_node()
    assert node.name == "numbers"
    assert node.path == "numbers"
    assert node.attrs["unit"] == "K"

    assert "SltDataset" in repr(dataset)
    assert "Dataset" in str(dataset)
    assert "Dataset" in dataset._repr_html_()

    dataset.show()
    captured = capsys.readouterr()
    assert "numbers" in captured.out


def test_dataset_to_node_rejects_non_dataset(slt: SltFile) -> None:
    slt.create_group("raw")

    dataset_handle = SltDataset(slt.path, "raw")
    with pytest.raises(TypeError):
        dataset_handle.to_node()


def test_dataset_shape_dtype_reject_non_dataset(slt: SltFile) -> None:
    slt.create_group("raw")
    dataset_handle = SltDataset(slt.path, "raw")

    with pytest.raises(TypeError):
        _ = dataset_handle.shape

    with pytest.raises(TypeError):
        _ = dataset_handle.dtype


def test_dataset_read_write_reject_non_dataset(slt: SltFile) -> None:
    slt.create_group("raw")
    dataset_handle = SltDataset(slt.path, "raw")

    with pytest.raises(TypeError):
        dataset_handle.read()

    with pytest.raises(TypeError):
        dataset_handle.write((), 1)


# ---------------------------------------------------------------------------
# Raw group tests
# ---------------------------------------------------------------------------


def test_missing_getitem_returns_proxy_group(slt: SltFile) -> None:
    group = slt["new_group"]
    assert isinstance(group, SltGroup)
    assert not group.exists
    assert "Proxy group" in str(group)
    assert "Proxy group" in group._repr_html_()


def test_create_group_and_attributes(slt: SltFile) -> None:
    group = slt.create_group("raw", kind="scratch")

    assert isinstance(group, SltGroup)
    assert group.exists
    assert group.attrs["kind"] == "scratch"
    assert not group.is_slothpy
    assert group.type is None
    assert group.primary is None


def test_group_require_rejects_existing_dataset(slt: SltFile) -> None:
    slt["raw"] = [1, 2, 3]
    group = SltGroup(slt.path, "raw")

    with pytest.raises(TypeError, match="dataset with this name"):
        group.require()


def test_group_create_dataset_and_access(slt: SltFile) -> None:
    group = slt.create_group("scratch")
    dataset = group.create_dataset("values", [1, 2, 3])

    assert isinstance(dataset, SltDataset)
    assert group["values"].read().tolist() == [1, 2, 3]  # type: ignore[union-attr]
    assert "values" in group
    assert group.keys() == ["values"]
    assert isinstance(group.items()["values"], SltDataset)


def test_group_setitem_creates_dataset(slt: SltFile) -> None:
    group = slt.create_group("scratch")
    group["values"] = [1, 2, 3]

    assert group["values"].read().tolist() == [1, 2, 3]  # type: ignore[union-attr]


def test_group_create_dataset_rejects_nested_key(slt: SltFile) -> None:
    group = slt.create_group("scratch")

    with pytest.raises(ValueError, match="direct child"):
        group.create_dataset("a/b", [1])


def test_group_create_dataset_rejects_duplicate_without_overwrite(slt: SltFile) -> None:
    group = slt.create_group("scratch")
    group.create_dataset("values", [1])

    with pytest.raises(FileExistsError):
        group.create_dataset("values", [2])


def test_group_create_dataset_overwrite(slt: SltFile) -> None:
    group = slt.create_group("scratch")
    group.create_dataset("values", [1])
    group.create_dataset("values", [2], overwrite=True)

    assert group["values"].read().tolist() == [2]  # type: ignore[union-attr]


def test_group_getitem_rejects_nested_key(slt: SltFile) -> None:
    group = slt.create_group("scratch")

    with pytest.raises(ValueError):
        _ = group["a/b"]


def test_group_getitem_can_return_child_group(slt: SltFile) -> None:
    with h5py.File(slt.path, "a") as h5:
        h5.require_group("scratch/child")

    child = slt["scratch"]["child"]
    assert isinstance(child, SltGroup)
    assert child.path == "scratch/child"


def test_group_contains_rejects_non_string_and_nested(slt: SltFile) -> None:
    group = slt.create_group("scratch")

    assert 1 not in group
    assert "a/b" not in group


def test_group_delete(slt: SltFile) -> None:
    group = slt.create_group("scratch")
    group["values"] = [1]

    del group["values"]
    assert "values" not in group


def test_group_delete_rejects_missing_and_nested(slt: SltFile) -> None:
    group = slt.create_group("scratch")

    with pytest.raises(KeyError):
        group.delete("missing")

    with pytest.raises(ValueError):
        group.delete("a/b")


def test_group_keys_rejects_non_group(slt: SltFile) -> None:
    slt["raw"] = [1]
    group_handle = SltGroup(slt.path, "raw")

    with pytest.raises(TypeError):
        group_handle.keys()


def test_group_to_node_raw_with_child_group(slt: SltFile) -> None:
    with h5py.File(slt.path, "a") as h5:
        raw = h5.require_group("raw")
        raw.create_dataset("values", data=[1, 2])
        raw.require_group("child")

    node = slt["raw"].to_node()
    assert node.name == "raw"
    assert not node.is_slothpy
    assert node.raw_datasets[0].name == "values"
    assert node.child_groups == ("child",)


def test_group_repr_str_html_show(
    slt: SltFile,
    capsys: pytest.CaptureFixture[str],
) -> None:
    group = slt.create_group("scratch")
    group["values"] = [1]

    assert "SltGroup" in repr(group)
    assert "Group" in str(group)
    assert "Group" in group._repr_html_()

    group.show()
    captured = capsys.readouterr()
    assert "scratch" in captured.out


# ---------------------------------------------------------------------------
# Semantic xarray group tests
# ---------------------------------------------------------------------------


def test_write_slothpy_group_and_read_xarray(
    slt_with_semantic_group: SltFile,
) -> None:
    group = slt_with_semantic_group["magnetisation_001"]

    assert isinstance(group, SltGroup)
    assert group.exists
    assert group.is_slothpy
    assert group.type == "MAGNETISATION"
    assert group.primary == "magnetisation"

    ds = group.to_dataset()
    da = group.to_xarray()
    try:
        assert isinstance(ds, xr.Dataset)
        assert isinstance(da, xr.DataArray)
        assert da.name == "magnetisation"
        assert ds.attrs["slt_valid"] == "true"
    finally:
        _safe_close(da)
        ds.close()


def test_semantic_group_variable_access(
    slt_with_semantic_group: SltFile,
) -> None:
    group = slt_with_semantic_group["magnetisation_001"]

    magnetisation = group["magnetisation"]
    temperature = group["temperature"]
    weight = group["orientation_weight"]
    tuple_access = slt_with_semantic_group["magnetisation_001/temperature"]

    try:
        assert isinstance(magnetisation, xr.DataArray)
        assert isinstance(temperature, xr.DataArray)
        assert isinstance(weight, xr.DataArray)
        assert isinstance(tuple_access, xr.DataArray)
        assert temperature.attrs["unit"] == "K"
    finally:
        _safe_close(magnetisation)
        _safe_close(temperature)
        _safe_close(weight)
        _safe_close(tuple_access)


def test_semantic_group_variable_missing_raises(
    slt_with_semantic_group: SltFile,
) -> None:
    group = slt_with_semantic_group["magnetisation_001"]

    with pytest.raises(KeyError, match="No variable or coordinate"):
        group.variable("missing")


def test_semantic_group_to_xarray_returns_dataset_for_dataset_primary(
    slt: SltFile,
) -> None:
    ds = xr.Dataset({"x": ("i", [1]), "y": ("i", [2])})
    group = slt._write_slothpy_group(
        "multi",
        ds,
        primary="__dataset__",
        overwrite=True,
    )

    opened = group.to_xarray()
    try:
        assert isinstance(opened, xr.Dataset)
        assert set(opened.data_vars) == {"x", "y"}
    finally:
        opened.close()


def test_semantic_group_to_xarray_missing_declared_primary_raises(
    slt: SltFile,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ds = xr.Dataset({"x": ("i", [1])})
    group = slt._write_slothpy_group("bad_primary", ds, primary="x")

    original_to_dataset = slt_mod.SltGroup.to_dataset

    def fake_to_dataset(self: SltGroup, *args: Any, **kwargs: Any) -> xr.Dataset:
        opened = original_to_dataset(self, *args, **kwargs)
        if self.path == "bad_primary":
            opened.attrs["slt_primary"] = "missing"
        return opened

    monkeypatch.setattr(slt_mod.SltGroup, "to_dataset", fake_to_dataset)

    with pytest.raises(KeyError, match="declares slt_primary"):
        group.to_xarray()


def test_semantic_group_metadata_methods(
    slt_with_semantic_group: SltFile,
) -> None:
    group = slt_with_semantic_group["magnetisation_001"]

    assert set(group.variables()) == {
        "magnetisation",
        "orientation_weight",
    }
    assert set(group.coordinates()) == {
        "field",
        "temperature",
        "orientation",
    }
    assert group.dimensions() == {
        "field": 8,
        "temperature": 3,
        "orientation": 4,
    }

    assert "magnetisation" in group
    assert "temperature" in group
    assert "missing" not in group

    assert set(group.keys()) == {
        "magnetisation",
        "orientation_weight",
        "field",
        "temperature",
        "orientation",
    }

    items = group.items()
    assert set(items) == set(group.keys())

    for value in items.values():
        close = getattr(value, "close", None)
        if callable(close):
            close()


def test_semantic_group_with_chunks(
    slt_with_semantic_group: SltFile,
) -> None:
    group = slt_with_semantic_group["magnetisation_001"].with_chunks({"field": 4})

    assert group.chunks == {"field": 4}

    try:
        da = group.to_xarray()
    except ImportError:
        pytest.skip("Dask is not installed.")
    else:
        try:
            assert getattr(da, "chunks", None) is not None
        finally:
            _safe_close(da)


def test_semantic_group_to_dataframe(
    slt_with_semantic_group: SltFile,
) -> None:
    group = slt_with_semantic_group["magnetisation_001"]
    frame = group.to_dataframe()

    assert "magnetisation" in frame.columns
    assert len(frame) == 8 * 3 * 4


def test_semantic_group_to_dataframe_dataset_primary(slt: SltFile) -> None:
    ds = xr.Dataset({"x": ("i", [1]), "y": ("i", [2])})
    group = slt._write_slothpy_group(
        "multi",
        ds,
        primary="__dataset__",
        overwrite=True,
    )
    frame = group.to_dataframe()

    assert set(frame.columns) == {"x", "y"}


def test_semantic_group_raw_mutation_is_protected(
    slt_with_semantic_group: SltFile,
) -> None:
    group = slt_with_semantic_group["magnetisation_001"]

    with pytest.raises(TypeError, match="semantic xarray group"):
        group.create_dataset("new", [1])

    with pytest.raises(TypeError, match="semantic xarray group"):
        group.delete("magnetisation")


def test_semantic_group_to_dataset_rejects_raw_group(slt: SltFile) -> None:
    group = slt.create_group("raw")

    with pytest.raises(TypeError, match="raw HDF5 group"):
        group.to_dataset()


def test_semantic_group_to_dataset_rejects_missing_group(slt: SltFile) -> None:
    group = SltGroup(slt.path, "missing")

    with pytest.raises(KeyError):
        group.to_dataset()


def test_write_slothpy_group_rejects_nested_name(slt: SltFile) -> None:
    with pytest.raises(ValueError, match="root-level"):
        slt._write_slothpy_group(
            "a/b",
            xr.Dataset({"x": ("i", [1])}),
        )


def test_write_slothpy_group_rejects_duplicate_without_overwrite(
    slt: SltFile,
) -> None:
    ds = xr.Dataset({"x": ("i", [1])})
    slt._write_slothpy_group("group", ds)

    with pytest.raises(FileExistsError):
        slt._write_slothpy_group("group", ds)


def test_write_slothpy_group_overwrite(slt: SltFile) -> None:
    ds1 = xr.Dataset({"x": ("i", [1])})
    ds2 = xr.Dataset({"x": ("i", [2])})

    slt._write_slothpy_group("group", ds1)
    group = slt._write_slothpy_group("group", ds2, overwrite=True)

    da = group.to_xarray()
    try:
        assert da.values.tolist() == [2]
    finally:
        _safe_close(da)


def test_write_slothpy_group_rejects_non_xarray(slt: SltFile) -> None:
    with pytest.raises(TypeError):
        slt._write_slothpy_group("group", [1, 2, 3])  # type: ignore[arg-type]


def test_semantic_group_to_node(
    slt_with_semantic_group: SltFile,
) -> None:
    node = slt_with_semantic_group["magnetisation_001"].to_node()

    assert node.is_slothpy
    assert node.name == "magnetisation_001"
    assert node.primary == "magnetisation"
    assert "field" in node.dimensions
    assert {coord.name for coord in node.coordinates} == {
        "field",
        "temperature",
        "orientation",
    }
    assert {var.name for var in node.data_variables} == {
        "magnetisation",
        "orientation_weight",
    }


def test_group_node_unreadable_semantic_group(
    slt: SltFile,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with h5py.File(slt.path, "a") as h5:
        bad = h5.require_group("bad")
        bad.attrs["slt_valid"] = "true"

    def raise_open_dataset(*args: Any, **kwargs: Any) -> xr.Dataset:
        raise RuntimeError("boom")

    monkeypatch.setattr(slt_mod.xr, "open_dataset", raise_open_dataset)

    node = slt_mod._group_node(slt.path, "bad")
    assert node.is_slothpy
    assert not node.readable
    assert node.error == "boom"


# ---------------------------------------------------------------------------
# File API tests
# ---------------------------------------------------------------------------


def test_file_group_require_exists(slt: SltFile) -> None:
    with pytest.raises(KeyError, match="No group"):
        slt.group("missing")

    group = slt.group("missing", require_exists=False)
    assert isinstance(group, SltGroup)
    assert not group.exists


def test_file_create_dataset_path_styles(slt: SltFile) -> None:
    slt["root"] = [1, 2]
    slt["group/values"] = [3, 4]
    slt[("group", "more_values")] = [5, 6]

    assert slt["root"].read().tolist() == [1, 2]  # type: ignore[union-attr]
    assert slt["group/values"].read().tolist() == [3, 4]  # type: ignore[union-attr]
    assert slt[("group", "more_values")].read().tolist() == [5, 6]  # type: ignore[union-attr]


def test_file_create_dataset_duplicate_and_overwrite(slt: SltFile) -> None:
    slt.create_dataset("root", [1])

    with pytest.raises(FileExistsError):
        slt.create_dataset("root", [2])

    dataset = slt.create_dataset("root", [2], overwrite=True)
    assert dataset.read().tolist() == [2]


def test_file_set_dataset_alias(slt: SltFile) -> None:
    dataset = slt.set_dataset("root", [1])
    assert dataset.read().tolist() == [1]


def test_file_create_dataset_rejects_parent_dataset(slt: SltFile) -> None:
    slt["parent"] = [1]

    with pytest.raises(TypeError, match="not a group"):
        slt["parent/child"] = [2]


def test_file_create_dataset_rejects_semantic_parent(
    slt_with_semantic_group: SltFile,
) -> None:
    with pytest.raises(TypeError, match="semantic xarray group"):
        slt_with_semantic_group["magnetisation_001/new"] = [1]


def test_file_getitem_group_dataset_and_proxy(slt: SltFile) -> None:
    slt["root"] = [1]
    slt["group/values"] = [2]

    assert isinstance(slt["root"], SltDataset)
    assert isinstance(slt["group"], SltGroup)
    assert isinstance(slt["group/values"], SltDataset)
    assert isinstance(slt["missing"], SltGroup)
    assert not slt["missing"].exists  # type: ignore[union-attr]


def test_file_contains(slt: SltFile) -> None:
    slt["root"] = [1]

    assert "root" in slt
    assert "missing" not in slt
    assert ("bad",) not in slt
    assert ("a", "b", "c") not in slt
    assert "a/b/c" not in slt
    assert 1 not in slt


def test_file_delete(slt: SltFile) -> None:
    slt["root"] = [1]
    assert "root" in slt

    del slt["root"]
    assert "root" not in slt


def test_file_delete_missing_raises(slt: SltFile) -> None:
    with pytest.raises(KeyError):
        slt.delete("missing")


def test_file_keys_groups_datasets_items(
    slt_with_semantic_group: SltFile,
) -> None:
    slt_with_semantic_group["notes"] = ["a"]
    slt_with_semantic_group["scratch/values"] = [1]

    assert set(slt_with_semantic_group.keys()) == {
        "magnetisation_001",
        "notes",
        "scratch",
    }
    assert set(slt_with_semantic_group.groups()) == {
        "magnetisation_001",
        "scratch",
    }
    assert slt_with_semantic_group.datasets() == ["notes"]
    assert slt_with_semantic_group.slothpy_groups() == ["magnetisation_001"]
    assert slt_with_semantic_group.raw_groups() == ["scratch"]

    items = slt_with_semantic_group.items()
    assert set(items) == {"magnetisation_001", "notes", "scratch"}


def test_file_to_groups(slt_with_semantic_group: SltFile) -> None:
    groups = slt_with_semantic_group.to_groups()

    try:
        assert set(groups) == {"magnetisation_001"}
        assert isinstance(groups["magnetisation_001"], xr.Dataset)
    finally:
        for ds in groups.values():
            ds.close()


def test_file_to_node_and_rendering(
    slt_with_semantic_group: SltFile,
    capsys: pytest.CaptureFixture[str],
) -> None:
    slt_with_semantic_group["notes"] = ["a"]
    slt_with_semantic_group["scratch/values"] = [1]

    node = slt_with_semantic_group.to_node()
    assert node.path == slt_with_semantic_group.path
    assert {group.name for group in node.groups} == {
        "magnetisation_001",
        "scratch",
    }
    assert {dataset.name for dataset in node.datasets} == {"notes"}

    assert slt_with_semantic_group.walk() == node
    assert "SltFile" in repr(slt_with_semantic_group)
    assert "SltFile" in str(slt_with_semantic_group)
    assert "SltFile" in slt_with_semantic_group._repr_html_()

    slt_with_semantic_group.show()
    captured = capsys.readouterr()
    assert "SltFile" in captured.out


def test_file_node_empty_file(slt: SltFile) -> None:
    assert "(empty)" in str(slt)


def test_open_hdf5_read_and_write_modes(slt: SltFile) -> None:
    with slt.open_hdf5("r") as h5:
        assert h5.attrs["format"] == "SlothPy"

    with slt.open_hdf5("a") as h5:
        h5.attrs["edited"] = True

    assert bool(slt.attrs["edited"])


def test_to_datatree_unavailable(
    slt: SltFile,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delattr(slt_mod.xr, "open_datatree", raising=False)

    with pytest.raises(RuntimeError, match="open_datatree"):
        slt.to_datatree()


def test_to_datatree_available(
    slt: SltFile,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = object()

    def fake_open_datatree(*args: Any, **kwargs: Any) -> object:
        assert args[0] == slt.path
        assert kwargs["engine"] == "h5netcdf"
        return sentinel

    monkeypatch.setattr(slt_mod.xr, "open_datatree", fake_open_datatree, raising=False)

    assert slt.to_datatree(chunks=None) is sentinel


def test_open_groups_unavailable(
    slt: SltFile,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delattr(slt_mod.xr, "open_groups", raising=False)

    with pytest.raises(RuntimeError, match="open_groups"):
        slt.open_groups()


def test_open_groups_available(
    slt: SltFile,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel: dict[str, xr.Dataset] = {}

    def fake_open_groups(*args: Any, **kwargs: Any) -> dict[str, xr.Dataset]:
        assert args[0] == slt.path
        assert kwargs["engine"] == "h5netcdf"
        return sentinel

    monkeypatch.setattr(slt_mod.xr, "open_groups", fake_open_groups, raising=False)

    assert slt.open_groups(chunks=None) is sentinel


# ---------------------------------------------------------------------------
# Cache-release regression test
# ---------------------------------------------------------------------------


def test_raw_attribute_write_after_lazy_xarray_open_does_not_fail(
    slt_with_semantic_group: SltFile,
) -> None:
    slt_with_semantic_group["scratch/values"] = [1, 2, 3]

    lazy_array = slt_with_semantic_group["magnetisation_001"].to_xarray()

    try:
        slt_with_semantic_group["scratch"]["values"].attrs["unit"] = "arb. u."
        assert slt_with_semantic_group["scratch"]["values"].attrs["unit"] == "arb. u."
    finally:
        _safe_close(lazy_array)


def test_raw_dataset_write_after_lazy_xarray_open_does_not_fail(
    slt_with_semantic_group: SltFile,
) -> None:
    lazy_array = slt_with_semantic_group["magnetisation_001"].to_xarray()

    try:
        slt_with_semantic_group["scratch/values"] = [1, 2, 3]
        assert slt_with_semantic_group["scratch"]["values"].read().tolist() == [1, 2, 3]  # type: ignore[union-attr]
    finally:
        _safe_close(lazy_array)


# ---------------------------------------------------------------------------
# Additional targeted coverage tests
# ---------------------------------------------------------------------------


def test_create_dataset_with_compression_for_non_scalar(slt: SltFile) -> None:
    dataset = slt.create_dataset(
        "compressed",
        np.arange(10),
        compression="gzip",
    )

    assert dataset.read().tolist() == list(range(10))

    with h5py.File(slt.path, "r") as h5:
        assert h5["compressed"].compression == "gzip"


def test_group_node_rejects_dataset_path(slt: SltFile) -> None:
    slt["numbers"] = [1, 2, 3]

    with pytest.raises(TypeError, match="not a group"):
        slt_mod._group_node(slt.path, "numbers")


def test_rich_label_and_trees_for_unreadable_group() -> None:
    node = slt_mod.SltGroupNode(
        path="broken",
        name="broken",
        attrs={},
        is_slothpy=True,
        dimensions={},
        coordinates=(),
        data_variables=(),
        primary=None,
        raw_datasets=(),
        child_groups=(),
        readable=False,
        error="boom",
    )

    assert "unreadable" in slt_mod._rich_to_ansi(slt_mod._group_label(node))
    assert "boom" in slt_mod._rich_to_ansi(slt_mod._group_tree(node))

    file_node = slt_mod.SltFileNode(
        path=Path("demo.slt"),
        attrs={},
        groups=(node,),
        datasets=(),
    )
    assert "boom" in slt_mod._rich_to_ansi(slt_mod._file_tree(file_node))


def test_rich_trees_for_empty_raw_group_and_child_group(slt: SltFile) -> None:
    slt.create_group("empty")

    with h5py.File(slt.path, "a") as h5:
        h5.require_group("raw_with_child/child")

    file_text = str(slt)
    empty_text = str(slt["empty"])
    child_text = str(slt["raw_with_child"])

    assert "(empty)" in file_text
    assert "(empty)" in empty_text
    assert "Child groups" in file_text
    assert "Child groups" in child_text
    assert "child" in child_text


def test_rich_trees_for_semantic_empty_sections() -> None:
    node = slt_mod.SltGroupNode(
        path="empty_semantic",
        name="empty_semantic",
        attrs={"slt_type": "EMPTY"},
        is_slothpy=True,
        dimensions={},
        coordinates=(),
        data_variables=(),
        primary=None,
        raw_datasets=(),
        child_groups=(),
        readable=True,
        error=None,
    )

    group_text = slt_mod._rich_to_ansi(slt_mod._group_tree(node))
    file_text = slt_mod._rich_to_ansi(
        slt_mod._file_tree(
            slt_mod.SltFileNode(
                path=Path("demo.slt"),
                attrs={},
                groups=(node,),
                datasets=(),
            )
        )
    )

    assert group_text.count("(none)") == 3
    assert file_text.count("(none)") == 3


def test_group_name_property_and_missing_type_primary(slt: SltFile) -> None:
    group = slt.create_group("scratch")
    assert group.name == "scratch"

    missing = SltGroup(slt.path, "missing")

    with pytest.raises(KeyError):
        _ = missing.type

    with pytest.raises(KeyError):
        _ = missing.primary


def test_to_dataset_decode_cf_argument(
    slt_with_semantic_group: SltFile,
) -> None:
    dataset = slt_with_semantic_group["magnetisation_001"].to_dataset(
        decode_cf=False,
    )

    try:
        assert isinstance(dataset, xr.Dataset)
    finally:
        dataset.close()


def test_to_xarray_can_return_primary_coordinate(slt: SltFile) -> None:
    ds = xr.Dataset(
        data_vars={"values": ("temperature", [1.0, 2.0, 3.0])},
        coords={"temperature": ("temperature", [2.0, 5.0, 10.0])},
    )
    group = slt._write_slothpy_group(
        "coord_primary",
        ds,
        primary="temperature",
        overwrite=True,
    )

    result = group.to_xarray()

    try:
        assert isinstance(result, xr.DataArray)
        assert result.name == "temperature"
        assert result.values.tolist() == [2.0, 5.0, 10.0]
    finally:
        result.close()


def test_group_create_dataset_rejects_parent_path_that_is_dataset(
    slt: SltFile,
) -> None:
    slt["parent"] = [1, 2, 3]
    group = SltGroup(slt.path, "parent")

    with pytest.raises(TypeError, match="not a group"):
        group.create_dataset("child", [1])


def test_group_getitem_unsupported_item_branch(
    slt: SltFile,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    slt.create_group("raw")

    def fake_get_hdf5_item(h5: h5py.File, path: str) -> object:
        return object()

    monkeypatch.setattr(slt_mod, "_get_hdf5_item", fake_get_hdf5_item)

    with pytest.raises(TypeError, match="Unsupported HDF5 object"):
        _ = slt["raw"]["anything"]


def test_semantic_group_walk(slt_with_semantic_group: SltFile) -> None:
    group = slt_with_semantic_group["magnetisation_001"]

    assert group.walk() == group.to_node()


def test_write_slothpy_group_with_encoding(slt: SltFile) -> None:
    ds = xr.Dataset({"x": ("i", np.array([1.0, 2.0], dtype=np.float64))})
    group = slt._write_slothpy_group(
        "encoded",
        ds,
        primary="x",
        encoding={"x": {"dtype": "float32"}},
        overwrite=True,
    )

    result = group.to_xarray()

    try:
        assert result.dtype == np.float32
    finally:
        result.close()


def test_file_getitem_group_path_unsupported_item_branch(
    slt: SltFile,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    slt.create_group("raw")

    def fake_get_hdf5_item(h5: h5py.File, path: str) -> object:
        return object()

    monkeypatch.setattr(slt_mod, "_get_hdf5_item", fake_get_hdf5_item)

    with pytest.raises(TypeError, match="Unsupported HDF5 object"):
        _ = slt["raw/anything"]


def test_file_getitem_root_unsupported_hdf5_object(slt: SltFile) -> None:
    with h5py.File(slt.path, "a") as h5:
        h5["named_dtype"] = np.dtype("int32")

    with pytest.raises(TypeError, match="Unsupported HDF5 object"):
        _ = slt["named_dtype"]


def test_semantic_group_str_covers_dimension_coordinate_variable_tree_branches(
    slt_with_semantic_group: SltFile,
) -> None:
    group = slt_with_semantic_group["magnetisation_001"]

    text = str(group)

    assert "Dimensions" in text
    assert "field" in text
    assert "temperature" in text
    assert "orientation" in text
    assert "Coordinates" in text
    assert "Data variables" in text
    assert "magnetisation" in text
    assert "orientation_weight" in text


def test_file_getitem_group_path_can_return_nested_group(
    slt: SltFile,
) -> None:
    with h5py.File(slt.path, "a") as h5:
        h5.require_group("raw/child")

    child = slt["raw/child"]

    assert isinstance(child, SltGroup)
    assert child.path == "raw/child"
