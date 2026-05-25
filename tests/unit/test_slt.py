from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest
import xarray as xr
from rich.text import Text

import slothpy.core.slt_common as slt_mod
import slothpy.core.slt_file as slt_file_mod
import slothpy.core.slt_group as slt_group_mod
from slothpy.core.slt import create_slt_file, open_slt_file, slt_file
from slothpy.core.slt_dataset import SltDataset
from slothpy.core.slt_file import SltFile
from slothpy.core.slt_group import SltGroup
from slothpy.core.slt_results import SltResults

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
        SltResults(
            dataset=semantic_dataset,
            slt_type="MAGNETISATION",
            primary="magnetisation",
        ),
        overwrite=True,
    )
    return slt


def _safe_close(obj: Any) -> None:
    close = getattr(obj, "close", None)
    if callable(close):
        close()


def _expect_group(value: object) -> SltGroup:
    assert isinstance(value, SltGroup)
    return value


def _expect_dataset(value: object) -> SltDataset:
    assert isinstance(value, SltDataset)
    return value


def _expect_dataarray(value: object) -> xr.DataArray:
    assert isinstance(value, xr.DataArray)
    return value


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


def test__split_supported_path_to_deep_path() -> None:
    with pytest.raises(ValueError):
        slt_mod._split_supported_path("group_1/group_2/dataset")


def test_xarray_group_path_adds_leading_slash() -> None:
    assert slt_mod._xarray_group_path("group") == "/group"


def test_coerce_hdf5_dataset_data_leaves_scalar_string_unchanged() -> None:
    value = "abc"

    result = slt_mod._coerce_hdf5_dataset_data(value)

    assert result is value


def test_coerce_hdf5_dataset_data_leaves_string_list_unchanged() -> None:
    value = ["a", "b"]

    result = slt_mod._coerce_hdf5_dataset_data(value)

    assert result is value


def test_coerce_hdf5_dataset_data_leaves_numeric_list_unchanged() -> None:
    value = [1, 2, 3]

    result = slt_mod._coerce_hdf5_dataset_data(value)

    assert result is value


def test_coerce_hdf5_dataset_data_leaves_numeric_numpy_array_unchanged() -> None:
    value = np.array([1, 2, 3])

    result = slt_mod._coerce_hdf5_dataset_data(value)

    assert result is value


def test_coerce_hdf5_dataset_data_converts_numpy_unicode_array() -> None:
    value = np.array(["a", "b"])

    result = slt_mod._coerce_hdf5_dataset_data(value)

    assert isinstance(result, np.ndarray)
    assert result.tolist() == ["a", "b"]
    assert h5py.check_string_dtype(result.dtype).encoding == "utf-8"


def test_create_dataset_from_numpy_unicode_array_and_read(slt: SltFile) -> None:
    dataset = slt.create_dataset("unicode_strings", np.array(["a", "b"]))

    assert dataset.read().tolist() == ["a", "b"]
    assert dataset.dtype == "str"


@pytest.mark.parametrize(
    "value",
    ["abc", np.array(1), 1, np.array(1), np.float64(1.0), np.int64(1), np.bool_(True)],
)
def test_is_scalar_dataset_data(value: Any) -> None:
    assert slt_mod._is_scalar_dataset_data(value)


@pytest.mark.parametrize("value", [[1, 2], np.array([1, 2])])
def test_is_scalar_dataset_data_false(value: Any) -> None:
    assert not slt_mod._is_scalar_dataset_data(value)


def test_is_scalar_dataset_data_returns_false_when_array_conversion_fails() -> None:
    class BadArrayLike:
        def __array__(self) -> np.ndarray:
            raise RuntimeError("boom")

    assert not slt_mod._is_scalar_dataset_data(BadArrayLike())


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
        ("true", True),
        (" TRUE ", True),
        (b"true", True),
        (b" TRUE ", True),
        ("false", False),
        (b"false", False),
        ("1", False),
        (b"1", False),
        (True, False),
        (1, False),
        (None, False),
    ],
)
def test_attrs_mark_slothpy_group(value: Any, expected: bool) -> None:
    assert slt_mod._attrs_mark_slothpy_group({"slt_valid": value}) is expected


def test_attrs_mark_slothpy_group_empty() -> None:
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
    text = Text("hello", style="bold red")
    ansi = slt_mod._rich_to_ansi(text)
    html = slt_mod._rich_to_html(text)

    assert "hello" in ansi
    assert "hello" in html
    assert "<pre" in html


def test_structure_html_uses_shared_styles(slt: SltFile) -> None:
    from slothpy.core.slt_html import structure_css

    slt.attrs["tag"] = "demo"
    html = slt.attrs._repr_html_()

    assert "slt-structure" in html
    assert "slt-card" in html
    assert "slt-table" in html
    assert "tag" in html
    assert structure_css() in html


def test_file_html_includes_groups_and_datasets(slt_with_semantic_group: SltFile) -> None:
    html = slt_with_semantic_group._repr_html_()

    assert "SltFile" in html
    assert "slt-structure" in html
    assert "Groups" in html
    assert "magnetisation_001" in html


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


def test_validate_slt_path_or_file_rejects_invalid_type(slt: SltFile) -> None:
    with pytest.raises(TypeError, match="slt_path_or_file must be a path or SltFile"):
        slt_file_mod._validate_slt_path_or_file(123)

    assert slt_file_mod._validate_slt_path_or_file(slt) is slt


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

    scalar_string = _expect_dataset(slt["scalar_string"])
    string_array = _expect_dataset(slt["string_array"])

    assert scalar_string.read() == "abc"
    assert string_array.read().tolist() == ["a", "b", "c"]


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
    group = _expect_group(slt["new_group"])
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

    with pytest.raises(
        TypeError, match="dataset or group with this name already exists"
    ):
        group.require()


def test_group_create_dataset_and_access(slt: SltFile) -> None:
    group = slt.create_group("scratch")
    dataset = group.create_dataset("values", [1, 2, 3])

    values = _expect_dataset(group["values"])

    assert isinstance(dataset, SltDataset)
    assert values.read().tolist() == [1, 2, 3]
    assert "values" in group
    assert group.keys() == ["values"]
    assert isinstance(group.items()["values"], SltDataset)


def test_group_setitem_creates_dataset(slt: SltFile) -> None:
    group = slt.create_group("scratch")
    group["values"] = [1, 2, 3]

    values = _expect_dataset(group["values"])

    assert values.read().tolist() == [1, 2, 3]


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

    values = _expect_dataset(group["values"])

    assert values.read().tolist() == [2]


def test_group_getitem_rejects_nested_key(slt: SltFile) -> None:
    group = slt.create_group("scratch")

    with pytest.raises(ValueError):
        _ = group["a/b"]


def test_group_getitem_can_return_child_group(slt: SltFile) -> None:
    with h5py.File(slt.path, "a") as h5:
        h5.require_group("scratch/child")

    scratch = _expect_group(slt["scratch"])
    child = _expect_group(scratch["child"])

    assert child.group_name == "scratch/child"


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

    group = _expect_group(slt["raw"])
    node = group.to_node()

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
    group = _expect_group(slt_with_semantic_group["magnetisation_001"])

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
    group = _expect_group(slt_with_semantic_group["magnetisation_001"])

    magnetisation = _expect_dataarray(group["magnetisation"])
    temperature = _expect_dataarray(group["temperature"])
    weight = _expect_dataarray(group["orientation_weight"])
    tuple_access = _expect_dataarray(
        slt_with_semantic_group["magnetisation_001/temperature"]
    )

    try:
        assert magnetisation.name == "magnetisation"
        assert temperature.name == "temperature"
        assert weight.name == "orientation_weight"
        assert tuple_access.name == "temperature"
        assert temperature.attrs["unit"] == "K"
    finally:
        _safe_close(magnetisation)
        _safe_close(temperature)
        _safe_close(weight)
        _safe_close(tuple_access)


def test_semantic_group_variable_missing_raises(
    slt_with_semantic_group: SltFile,
) -> None:
    group = _expect_group(slt_with_semantic_group["magnetisation_001"])

    with pytest.raises(KeyError, match="No variable or coordinate"):
        group.variable("missing")


def test_semantic_group_to_xarray_returns_dataset_for_dataset_primary(
    slt: SltFile,
) -> None:
    ds = xr.Dataset({"x": ("i", [1]), "y": ("i", [2])})
    group = slt._write_slothpy_group(
        "multi",
        SltResults(dataset=ds, primary="__dataset__"),
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
    group = slt._write_slothpy_group("bad_primary", SltResults(dataset=ds, primary="x"))

    original_to_dataset = SltGroup.to_dataset

    def fake_to_dataset(self: SltGroup, *args: Any, **kwargs: Any) -> xr.Dataset:
        opened = original_to_dataset(self, *args, **kwargs)
        if self.group_name == "bad_primary":
            opened.attrs["slt_primary"] = "missing"
        return opened

    monkeypatch.setattr(SltGroup, "to_dataset", fake_to_dataset)

    with pytest.raises(KeyError, match="declares slt_primary"):
        group.to_xarray()


def test_semantic_group_metadata_methods(
    slt_with_semantic_group: SltFile,
) -> None:
    group = _expect_group(slt_with_semantic_group["magnetisation_001"])

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
    group = _expect_group(slt_with_semantic_group["magnetisation_001"]).with_chunks(
        {"field": 4}
    )

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
    group = _expect_group(slt_with_semantic_group["magnetisation_001"])
    frame = group.to_dataframe()

    assert "magnetisation" in frame.columns
    assert len(frame) == 8 * 3 * 4


def test_semantic_group_to_dataframe_dataset_primary(slt: SltFile) -> None:
    ds = xr.Dataset({"x": ("i", [1]), "y": ("i", [2])})
    group = slt._write_slothpy_group(
        "multi",
        SltResults(dataset=ds, primary="__dataset__"),
        overwrite=True,
    )
    frame = group.to_dataframe()

    assert set(frame.columns) == {"x", "y"}


def test_semantic_group_raw_mutation_is_protected(
    slt_with_semantic_group: SltFile,
) -> None:
    group = _expect_group(slt_with_semantic_group["magnetisation_001"])

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
            SltResults(dataset=xr.Dataset({"x": ("i", [1])}), primary="x"),
        )


def test_write_slothpy_group_rejects_duplicate_without_overwrite(
    slt: SltFile,
) -> None:
    ds = xr.Dataset({"x": ("i", [1])})
    slt._write_slothpy_group("group", SltResults(dataset=ds))

    with pytest.raises(FileExistsError):
        slt._write_slothpy_group("group", SltResults(dataset=ds))


def test_write_slothpy_group_overwrite(slt: SltFile) -> None:
    ds1 = xr.Dataset({"x": ("i", [1])})
    ds2 = xr.Dataset({"x": ("i", [2])})

    slt._write_slothpy_group("group", SltResults(dataset=ds1))
    group = slt._write_slothpy_group("group", SltResults(dataset=ds2), overwrite=True)

    da = _expect_dataarray(group.to_xarray())
    try:
        assert da.to_numpy().tolist() == [2]
    finally:
        _safe_close(da)


def test_write_slothpy_group_rejects_non_xarray(slt: SltFile) -> None:
    with pytest.raises(TypeError):
        slt._write_slothpy_group(
            "group",
            SltResults(dataset=[1, 2, 3]),  # type: ignore[arg-type]
        )


def test_semantic_group_to_node(
    slt_with_semantic_group: SltFile,
) -> None:
    group = _expect_group(slt_with_semantic_group["magnetisation_001"])
    node = group.to_node()

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

    root = _expect_dataset(slt["root"])
    values = _expect_dataset(slt["group/values"])
    more_values = _expect_dataset(slt[("group", "more_values")])

    assert root.read().tolist() == [1, 2]
    assert values.read().tolist() == [3, 4]
    assert more_values.read().tolist() == [5, 6]


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

    root = slt["root"]
    group = slt["group"]
    values = slt["group/values"]
    missing = _expect_group(slt["missing"])

    assert isinstance(root, SltDataset)
    assert isinstance(group, SltGroup)
    assert isinstance(values, SltDataset)
    assert not missing.exists


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
    monkeypatch.delattr(slt_file_mod.xr, "open_datatree", raising=False)

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

    monkeypatch.setattr(
        slt_file_mod.xr, "open_datatree", fake_open_datatree, raising=False
    )

    assert slt.to_datatree(chunks=None) is sentinel


def test_open_groups_unavailable(
    slt: SltFile,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delattr(slt_file_mod.xr, "open_groups", raising=False)

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

    monkeypatch.setattr(slt_file_mod.xr, "open_groups", fake_open_groups, raising=False)

    assert slt.open_groups(chunks=None) is sentinel


# ---------------------------------------------------------------------------
# Cache-release regression test
# ---------------------------------------------------------------------------


def test_raw_attribute_write_after_lazy_xarray_open_does_not_fail(
    slt_with_semantic_group: SltFile,
) -> None:
    slt_with_semantic_group["scratch/values"] = [1, 2, 3]

    semantic_group = _expect_group(slt_with_semantic_group["magnetisation_001"])
    lazy_array = semantic_group.to_xarray()

    try:
        scratch = _expect_group(slt_with_semantic_group["scratch"])
        values = _expect_dataset(scratch["values"])

        values.attrs["unit"] = "arb. u."
        assert values.attrs["unit"] == "arb. u."
    finally:
        _safe_close(lazy_array)


def test_raw_dataset_write_after_lazy_xarray_open_does_not_fail(
    slt_with_semantic_group: SltFile,
) -> None:
    semantic_group = _expect_group(slt_with_semantic_group["magnetisation_001"])
    lazy_array = semantic_group.to_xarray()

    try:
        slt_with_semantic_group["scratch/values"] = [1, 2, 3]

        scratch = _expect_group(slt_with_semantic_group["scratch"])
        values = _expect_dataset(scratch["values"])

        assert values.read().tolist() == [1, 2, 3]
    finally:
        _safe_close(lazy_array)


# ---------------------------------------------------------------------------
# Path-targeted xarray cache release and retry-opening tests
# ---------------------------------------------------------------------------


def test_resolve_file_path_does_not_require_existing_file(tmp_path: Path) -> None:
    path = tmp_path / "missing.slt"

    resolved = slt_mod._resolve_file_path(path)

    assert resolved.is_absolute()
    assert resolved.name == "missing.slt"


def test_as_resolved_path_accepts_path_and_string(tmp_path: Path) -> None:
    path = tmp_path / "demo.slt"

    assert slt_mod._as_resolved_path(path) == path.resolve(strict=False)
    assert slt_mod._as_resolved_path(str(path)) == path.resolve(strict=False)
    assert slt_mod._as_resolved_path(object()) is None


def test_iter_nested_values_recurses_through_containers(tmp_path: Path) -> None:
    path = tmp_path / "demo.slt"
    nested = {
        "outer": [
            ("inner", path),
            {"other": 1},
        ]
    }

    values = list(slt_mod._iter_nested_values(nested))

    assert nested in values
    assert path in values
    assert "outer" in values
    assert "inner" in values
    assert "other" in values
    assert 1 in values


def test_cached_file_matches_path_from_cache_key(tmp_path: Path) -> None:
    target = tmp_path / "demo.slt"
    other = tmp_path / "other.slt"

    assert slt_mod._cached_file_matches_path(
        cache_key=("open", (str(target),), {}),
        cached_file=object(),
        target_path=target,
    )
    assert not slt_mod._cached_file_matches_path(
        cache_key=("open", (str(other),), {}),
        cached_file=object(),
        target_path=target,
    )


def test_cached_file_matches_path_from_cached_file_attribute(tmp_path: Path) -> None:
    target = tmp_path / "demo.slt"

    class CachedFile:
        filename = str(target)

    assert slt_mod._cached_file_matches_path(
        cache_key=("no-path-here",),
        cached_file=CachedFile(),
        target_path=target,
    )


def test_cached_file_matches_path_from_callable_attribute(tmp_path: Path) -> None:
    target = tmp_path / "demo.slt"

    class CachedFile:
        def filepath(self) -> str:
            return str(target)

    assert slt_mod._cached_file_matches_path(
        cache_key=("no-path-here",),
        cached_file=CachedFile(),
        target_path=target,
    )


def test_cached_file_matches_path_ignores_attribute_errors(tmp_path: Path) -> None:
    target = tmp_path / "demo.slt"

    class CachedFile:
        @property
        def filename(self) -> str:
            raise RuntimeError("boom")

    assert not slt_mod._cached_file_matches_path(
        cache_key=("no-path-here",),
        cached_file=CachedFile(),
        target_path=target,
    )


def test_cached_file_matches_path_ignores_callable_errors(tmp_path: Path) -> None:
    target = tmp_path / "demo.slt"

    class CachedFile:
        def filename(self) -> str:
            raise RuntimeError("boom")

    assert not slt_mod._cached_file_matches_path(
        cache_key=("no-path-here",),
        cached_file=CachedFile(),
        target_path=target,
    )


def test_close_cached_file_calls_close() -> None:
    class CachedFile:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    cached_file = CachedFile()

    slt_mod._close_cached_file(cached_file)

    assert cached_file.closed


def test_close_cached_file_without_close_method() -> None:
    slt_mod._close_cached_file(object())


def test_release_xarray_file_handles_only_releases_matching_file(
    tmp_path: Path,
) -> None:
    from xarray.backends import file_manager

    target = tmp_path / "target.slt"
    other = tmp_path / "other.slt"

    class CachedFile:
        def __init__(self, filename: Path) -> None:
            self.filename = str(filename)
            self.closed = False

        def close(self) -> None:
            self.closed = True

    target_cached = CachedFile(target)
    other_cached = CachedFile(other)

    old_cache = file_manager.FILE_CACHE
    fake_cache = {
        ("target-key", str(target)): target_cached,
        ("other-key", str(other)): other_cached,
    }

    try:
        file_manager.FILE_CACHE = fake_cache  # type: ignore[assignment]
        slt_mod._release_xarray_file_handles(target)

        assert ("target-key", str(target)) not in fake_cache
        assert ("other-key", str(other)) in fake_cache
        assert target_cached.closed
        assert not other_cached.closed
    finally:
        file_manager.FILE_CACHE = old_cache  # type: ignore[assignment]


def test_release_xarray_file_handles_without_path_clears_all(
    tmp_path: Path,
) -> None:
    from xarray.backends import file_manager

    old_cache = file_manager.FILE_CACHE
    fake_cache = {"a": object(), "b": object()}

    try:
        file_manager.FILE_CACHE = fake_cache  # type: ignore[assignment]
        slt_mod._release_xarray_file_handles()

        assert fake_cache == {}
    finally:
        file_manager.FILE_CACHE = old_cache  # type: ignore[assignment]


def test_release_xarray_file_handles_falls_back_to_clear_when_items_fails(
    tmp_path: Path,
) -> None:
    from xarray.backends import file_manager

    class BadCache:
        def __init__(self) -> None:
            self.cleared = False

        def items(self) -> None:
            raise RuntimeError("boom")

        def clear(self) -> None:
            self.cleared = True

    old_cache = file_manager.FILE_CACHE
    bad_cache = BadCache()

    try:
        file_manager.FILE_CACHE = bad_cache  # type: ignore[assignment]
        slt_mod._release_xarray_file_handles(tmp_path / "demo.slt")

        assert bad_cache.cleared
    finally:
        file_manager.FILE_CACHE = old_cache  # type: ignore[assignment]


def test_release_xarray_file_handles_ignores_delete_errors(
    tmp_path: Path,
) -> None:
    from xarray.backends import file_manager

    target = tmp_path / "target.slt"

    class Cache:
        def __init__(self) -> None:
            self.cached_file = CachedFile()
            self.deleted = False

        def items(self):
            return [((str(target),), self.cached_file)]

        def __delitem__(self, key) -> None:
            self.deleted = True
            raise RuntimeError("cannot delete")

    class CachedFile:
        filename = str(target)

        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    old_cache = file_manager.FILE_CACHE
    cache = Cache()

    try:
        file_manager.FILE_CACHE = cache  # type: ignore[assignment]
        slt_mod._release_xarray_file_handles(target)

        assert cache.deleted
        assert cache.cached_file.closed
    finally:
        file_manager.FILE_CACHE = old_cache  # type: ignore[assignment]


def test_hdf5_mode_requests_write() -> None:
    assert not slt_mod._hdf5_mode_requests_write("r")
    assert slt_mod._hdf5_mode_requests_write("r+")
    assert slt_mod._hdf5_mode_requests_write("a")
    assert slt_mod._hdf5_mode_requests_write("w")
    assert slt_mod._hdf5_mode_requests_write("x")


def test_is_xarray_read_only_conflict() -> None:
    assert slt_mod._is_xarray_read_only_conflict(
        OSError("file is already open for read-only")
    )
    assert slt_mod._is_xarray_read_only_conflict(
        OSError("unable to open file: already open for read-only")
    )
    assert not slt_mod._is_xarray_read_only_conflict(OSError("different error"))


def test_open_hdf5_handle_read_mode_does_not_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def fake_file(path: Path, mode: str) -> h5py.File:
        calls.append((path, mode))
        raise OSError("file is already open for read-only")

    monkeypatch.setattr(slt_mod.h5py, "File", fake_file)

    with pytest.raises(OSError):
        slt_mod._open_hdf5_handle(tmp_path / "demo.slt", "r")

    assert len(calls) == 1


def test_open_hdf5_handle_write_mode_does_not_retry_unrelated_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def fake_file(path: Path, mode: str) -> h5py.File:
        calls.append((path, mode))
        raise OSError("permission denied")

    monkeypatch.setattr(slt_mod.h5py, "File", fake_file)

    with pytest.raises(OSError, match="permission denied"):
        slt_mod._open_hdf5_handle(tmp_path / "demo.slt", "a")

    assert len(calls) == 1


def test_open_hdf5_handle_write_mode_releases_target_and_retries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "demo.slt"
    opened = object()
    calls = []
    released = []

    def fake_file(path: Path, mode: str) -> object:
        calls.append((path, mode))
        if len(calls) == 1:
            raise OSError("file is already open for read-only")
        return opened

    def fake_release(path: Path) -> None:
        released.append(path)

    monkeypatch.setattr(slt_mod.h5py, "File", fake_file)
    monkeypatch.setattr(slt_mod, "_release_xarray_file_handles", fake_release)

    result = slt_mod._open_hdf5_handle(target, "a")

    assert result is opened
    assert len(calls) == 2
    assert released == [target.resolve(strict=False)]


def test_open_hdf5_file_closes_handle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeH5:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    fake_h5 = FakeH5()

    monkeypatch.setattr(
        slt_mod,
        "_open_hdf5_handle",
        lambda file_path, mode: fake_h5,
    )

    with slt_mod._open_hdf5_file(tmp_path / "demo.slt", "a") as h5:
        assert h5 is fake_h5
        assert not fake_h5.closed

    assert fake_h5.closed


def test_write_xarray_to_netcdf_with_retry_releases_target_and_retries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "demo.slt"
    ds = xr.Dataset({"x": ("i", [1, 2, 3])})

    calls = []
    released = []

    def fake_to_netcdf(self: xr.Dataset, path: Path, **kwargs: object) -> None:
        calls.append((self, path, kwargs))
        if len(calls) == 1:
            raise OSError("file is already open for read-only")

    def fake_release(path: Path) -> None:
        released.append(path)

    monkeypatch.setattr(xr.Dataset, "to_netcdf", fake_to_netcdf)
    monkeypatch.setattr(slt_mod, "_release_xarray_file_handles", fake_release)

    slt_mod._write_xarray_to_netcdf_with_retry(
        ds,
        target,
        group="/group",
        mode="a",
        engine="h5netcdf",
    )

    assert len(calls) == 2
    assert calls[0][0] is ds
    assert calls[1][0] is ds
    assert released == [target.resolve(strict=False)]
    assert calls[0][2]["group"] == "/group"


def test_create_different_slt_file_does_not_release_unrelated_xarray_handle(
    tmp_path: Path,
) -> None:
    from xarray.backends import file_manager

    first = create_slt_file(tmp_path / "first.slt", overwrite=True)
    first._write_slothpy_group(
        "data",
        SltResults(dataset=xr.Dataset({"x": ("i", [1, 2, 3])}), primary="x"),
    )

    lazy = _expect_group(first["data"]).to_xarray()

    class CachedFile:
        def __init__(self, filename: Path) -> None:
            self.filename = str(filename)
            self.closed = False

        def close(self) -> None:
            self.closed = True

    cached = CachedFile(first.path)

    old_cache = file_manager.FILE_CACHE
    fake_cache = {("first", str(first.path)): cached}

    try:
        file_manager.FILE_CACHE = fake_cache  # type: ignore[assignment]

        second = create_slt_file(tmp_path / "second.slt", overwrite=True)

        assert second.path.name == "second.slt"
        assert ("first", str(first.path)) in fake_cache
        assert not cached.closed
    finally:
        _safe_close(lazy)
        file_manager.FILE_CACHE = old_cache  # type: ignore[assignment]


def test_mutating_same_slt_file_after_lazy_xarray_open_retries_and_succeeds(
    tmp_path: Path,
) -> None:
    slt = create_slt_file(tmp_path / "demo.slt", overwrite=True)
    slt._write_slothpy_group(
        "data",
        SltResults(dataset=xr.Dataset({"x": ("i", [1, 2, 3])}), primary="x"),
    )

    lazy = _expect_group(slt["data"]).to_xarray()

    try:
        slt["scratch/values"] = [4, 5, 6]

        scratch = _expect_group(slt["scratch"])
        values = _expect_dataset(scratch["values"])

        values.attrs["unit"] = "arb. u."

        assert values.read().tolist() == [4, 5, 6]
        assert values.attrs["unit"] == "arb. u."
    finally:
        _safe_close(lazy)


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
    group = _expect_group(slt_with_semantic_group["magnetisation_001"])
    dataset = group.to_dataset(decode_cf=False)

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
        SltResults(dataset=ds, primary="temperature"),
        overwrite=True,
    )

    result = _expect_dataarray(group.to_xarray())

    try:
        assert result.name == "temperature"
        assert result.to_numpy().tolist() == [2.0, 5.0, 10.0]
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

    monkeypatch.setattr(slt_group_mod, "_get_hdf5_item", fake_get_hdf5_item)

    raw = _expect_group(slt["raw"])

    with pytest.raises(TypeError, match="Unsupported HDF5 object"):
        _ = raw["anything"]


def test_semantic_group_walk(slt_with_semantic_group: SltFile) -> None:
    group = _expect_group(slt_with_semantic_group["magnetisation_001"])

    assert group.walk() == group.to_node()


def test_write_slothpy_group_with_encoding(slt: SltFile) -> None:
    ds = xr.Dataset({"x": ("i", np.array([1.0, 2.0], dtype=np.float64))})
    group = slt._write_slothpy_group(
        "encoded",
        SltResults(dataset=ds, primary="x"),
        encoding={"x": {"dtype": "float32"}},
        overwrite=True,
    )

    result = _expect_dataarray(group.to_xarray())

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

    monkeypatch.setattr(slt_file_mod, "_get_hdf5_item", fake_get_hdf5_item)

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
    group = _expect_group(slt_with_semantic_group["magnetisation_001"])

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
    assert child.group_name == "raw/child"


def test_as_resolved_path_returns_none_when_path_expanduser_fails() -> None:
    class BadPath(type(Path())):  # type: ignore[misc]
        def expanduser(self):
            raise OSError("boom")

    assert slt_mod._as_resolved_path(BadPath("demo.slt")) is None


def test_as_resolved_path_returns_none_when_string_path_resolution_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BadPath:
        def __init__(self, value: str) -> None:
            self.value = value

        def expanduser(self):
            raise OSError("boom")

    monkeypatch.setattr(slt_mod, "Path", BadPath)

    assert slt_mod._as_resolved_path("demo.slt") is None


def test_write_xarray_to_netcdf_with_retry_success_first_try(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "demo.slt"
    ds = xr.Dataset({"x": ("i", [1, 2, 3])})

    calls = []

    def fake_to_netcdf(self: xr.Dataset, path: Path, **kwargs: object) -> None:
        calls.append((self, path, kwargs))

    monkeypatch.setattr(xr.Dataset, "to_netcdf", fake_to_netcdf)

    slt_mod._write_xarray_to_netcdf_with_retry(
        ds,
        target,
        group="/group",
        mode="a",
        engine="h5netcdf",
    )

    assert len(calls) == 1
    assert calls[0][0] is ds
    assert calls[0][1] == target.resolve(strict=False)
    assert calls[0][2]["group"] == "/group"


def test_write_xarray_to_netcdf_with_retry_checks_unrelated_oserror_branch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "demo.slt"
    ds = xr.Dataset({"x": ("i", [1, 2, 3])})

    calls = []

    def fake_to_netcdf(self: xr.Dataset, path: Path, **kwargs: object) -> None:
        calls.append((self, path, kwargs))
        raise OSError("unrelated HDF5 failure")

    conflict_checks = []

    def fake_is_conflict(exc: OSError) -> bool:
        conflict_checks.append(str(exc))
        return False

    monkeypatch.setattr(xr.Dataset, "to_netcdf", fake_to_netcdf)
    monkeypatch.setattr(
        slt_mod,
        "_is_xarray_read_only_conflict",
        fake_is_conflict,
    )

    with pytest.raises(OSError, match="unrelated HDF5 failure"):
        slt_mod._write_xarray_to_netcdf_with_retry(
            ds,
            target,
            group="/group",
            mode="a",
            engine="h5netcdf",
        )

    assert len(calls) == 1
    assert conflict_checks == ["unrelated HDF5 failure"]
