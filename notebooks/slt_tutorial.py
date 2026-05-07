import marimo

__generated_with = "0.23.3"
app = marimo.App(width="wide")


@app.cell
def _():
    from pathlib import Path
    import base64

    import marimo as mo
    import numpy as np
    import xarray as xr

    from slothpy.core.slt import create_slt_file, open_slt_file
    from slothpy.orca_hamiltonian_reader import hamiltonian_from_orca

    return (
        Path,
        create_slt_file,
        hamiltonian_from_orca,
        mo,
        np,
        open_slt_file,
        xr,
    )


@app.cell
def _(hamiltonian_from_orca):
    slt = hamiltonian_from_orca(
        "src/slothpy/Pr_minimal.out",
        "demo.slt",
        "orca_hamiltonian",
        electric_dipole_momenta=True,
        ci_basis=True,
        overwrite=True,
    )
    slt["orca_hamiltonian"]["ci_coefficients_mult_1"]
    return (slt,)


@app.cell
def _(np, slt):
    from numpy.dtypes import StringDType

    a = [[1,2,2], "kkkokokokok", "eoriej erogi"]
    b = np.asarray(a, dtype=object)
    print(b.dtype)
    c = ["Dy", "Tb", "Yb"]
    d = 1
    import h5py

    with h5py.File("demo.slt", "a") as f:
        # del f["newe"]
        print(f["newe"].value)
    # del slt["new"]
    # slt["new"] = a
    # slt["new"][:]
    type(slt["newe"])
    return


@app.cell
def _(Path):
    if "__file__" in globals():
        NOTEBOOK_DIR = Path(__file__).resolve().parent
    else:
        NOTEBOOK_DIR = Path.cwd()

    DATA_DIR = NOTEBOOK_DIR / "tutorial_data"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    SLT_PATH = DATA_DIR / "demo_tutorial.slt"
    return (SLT_PATH,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ![SlothPy logo](public/slothpy_logo.ico)

    # SlothPy `.slt` tutorial

    A practical introduction to SlothPy storage, HDF5 integration, and xarray-backed scientific workflows.

    ---

    ## What this notebook covers

    1. What an `.slt` file is
    2. Creating and opening SlothPy files
    3. Writing raw user datasets and groups
    4. Reading data and working with attributes
    5. Writing SlothPy semantic xarray groups
    6. Reading data lazily with xarray
    7. Selections, reductions, and dataframe conversion
    8. Using the low-level HDF5 escape hatch
    9. Important notes about lazy file handles and mutation
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Conceptual overview

    SlothPy uses the **HDF5 file format** as the physical storage layer and provides a convenient Python API on top.

    This gives you:

    - **HDF5 compatibility** for robust hierarchical storage
    - **xarray integration** for labeled, multidimensional scientific data
    - **simple raw storage** for user-defined scratch data, notes, arrays, etc.
    - **structured SlothPy semantic groups** for domain-specific computed results

    In practice, an `.slt` file can contain two kinds of content:

    ### A. Raw user content

    Examples:

    - root-level datasets
    - raw HDF5 groups
    - raw datasets inside those groups

    These are convenient for scratch work and custom additions.

    ### B. SlothPy semantic groups

    These are special HDF5 groups that:

    - are written by SlothPy internals,
    - carry metadata identifying them as valid SlothPy groups,
    - can be opened directly as **xarray datasets/data arrays**.

    This architecture lets SlothPy integrate naturally into the scientific Python ecosystem while still keeping the file format open and inspectable.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Imports used in this notebook

    We will use:

    - `slothpy.core.slt` for the SlothPy file API
    - `numpy` for numerical arrays
    - `xarray` for labeled multidimensional data
    - `h5py` for low-level HDF5 inspection
    """)
    return


@app.cell
def _(SLT_PATH, create_slt_file):
    slt = create_slt_file(SLT_PATH, overwrite=True)
    slt
    return (slt,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Creating a new `.slt` file

    A new SlothPy file is created with:

    ```python
    slt = create_slt_file("demo", overwrite=True)
    ```

    This creates a fresh HDF5-backed `.slt` file and writes basic file-level metadata such as the SlothPy format version and storage model.
    """)
    return


@app.cell
def _(slt):
    file_attrs = slt.attrs.as_dict()
    file_attrs
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Writing raw user data

    The simplest way to use an `.slt` file is like a mapping.

    You can write:

    - a **root dataset**:

    ```python
    slt["notes"] = ["a", "b", "c"]
    ```

    - a dataset inside a group using a slash path:

    ```python
    slt["scratch/values"] = [1, 2, 3]
    ```

    - or with chained group access:

    ```python
    slt["scratch"]["more_values"] = [4, 5, 6]
    ```

    If a target already exists, this API raises an error rather than overwriting silently.
    """)
    return


@app.cell
def _(slt):
    slt["notes"] = ["a", "b", "c"]
    slt["scratch/values"] = [1, 2, 3]
    slt["scratch"]["more_values"] = [4, 5, 6]

    slt
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Adding and reading attributes

    Every file, group, and dataset has an `.attrs` mapping.

    Examples:

    ```python
    slt.attrs["author"] = "..."
    slt["scratch"].attrs["kind"] = "workspace"
    slt["scratch"]["more_values"].attrs["unit"] = "arb. u."
    ```

    These are just HDF5 attributes under the hood.
    """)
    return


@app.cell
def _(slt):
    slt.attrs["author"] = "SlothPy tutorial"
    slt.attrs["purpose"] = "Demonstration of .slt storage API"

    slt["scratch"].attrs["kind"] = "workspace"
    slt["scratch"].attrs["description"] = "Raw user scratch area"
    slt["scratch"]["more_values"].attrs["unit"] = "arb. u."
    slt["scratch"]["more_values"].attrs["long_name"] = "Example raw values"

    root_attrs = slt.attrs.as_dict()
    scratch_attrs = slt["scratch"].attrs.as_dict()
    more_values_attrs = slt["scratch"]["more_values"].attrs.as_dict()
    return more_values_attrs, root_attrs, scratch_attrs


@app.cell
def _(more_values_attrs, root_attrs, scratch_attrs):
    {
        "root_attrs": root_attrs,
        "scratch_attrs": scratch_attrs,
        "scratch/more_values attrs": more_values_attrs,
    }
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Reading raw datasets and groups

    Raw datasets are represented by `SltDataset` handles.

    You can:

    - inspect them,
    - read slices,
    - convert them to NumPy arrays,
    - attach attributes.

    A raw group is represented by `SltGroup`.
    """)
    return


@app.cell
def _(slt):
    notes_ds = slt["notes"]
    scratch_group = slt["scratch"]
    more_values_ds = slt["scratch"]["more_values"]
    return more_values_ds, notes_ds, scratch_group


@app.cell
def _(more_values_ds, notes_ds, scratch_group):
    {
        "notes_handle": notes_ds,
        "scratch_group": scratch_group,
        "more_values_handle": more_values_ds,
    }
    return


@app.cell
def _(more_values_ds, notes_ds):
    notes_array = notes_ds.read()
    more_values_array = more_values_ds.read()
    {
        "notes": notes_array,
        "more_values": more_values_array,
    }
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Convenience path styles

    All of the following styles are supported for raw content:

    ```python
    slt["dataset"]
    slt["group/dataset"]
    slt["group"]["dataset"]
    ```

    This keeps the public API simple and pleasant for notebook use.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. Writing a SlothPy semantic xarray group

    Now we move to the scientifically more important case.

    A SlothPy semantic group is an HDF5 group that stores a valid xarray dataset with SlothPy metadata attached. These groups are typically written by SlothPy computational routines.

    For tutorial purposes, we will use the internal helper:

    ```python
    slt._write_slothpy_group(...)
    ```

    ### Important note

    This is **internal / advanced** API. Normal end users should mostly **read** semantic groups, not create them manually.
    """)
    return


@app.cell
def _(np, xr):
    rng = np.random.default_rng(42)

    fields = np.linspace(0.0, 7.0, 8)
    temperatures = np.array([2.0, 5.0, 10.0])
    orientations = np.arange(4)

    ds_mag = xr.Dataset(
        data_vars={
            "magnetisation": (
                ("field", "temperature", "orientation"),
                rng.random((fields.size, temperatures.size, orientations.size)),
                {"unit": "mu_B", "long_name": "magnetisation"},
            ),
            "orientation_weight": (
                ("orientation",),
                np.ones(orientations.size) / orientations.size,
                {"long_name": "powder weight"},
            ),
        },
        coords={
            "field": ("field", fields, {"unit": "T", "long_name": "magnetic field"}),
            "temperature": (
                "temperature",
                temperatures,
                {"unit": "K", "long_name": "temperature"},
            ),
            "orientation": ("orientation", orientations),
        },
        attrs={
            "title": "Demo magnetisation dataset",
            "source": "marimo tutorial",
        },
    )

    ds_mag
    return (ds_mag,)


@app.cell
def _(ds_mag, slt):
    mag_group = slt._write_slothpy_group(
        "magnetisation_001",
        ds_mag,
        slt_type="MAGNETISATION",
        primary="magnetisation",
        overwrite=True,
    )

    mag_group
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9. Inspecting the file after writing a semantic group

    Notice that the file can now contain a mix of:

    - raw root datasets,
    - raw groups,
    - SlothPy semantic xarray groups.

    This is one of the main strengths of the design.
    """)
    return


@app.cell
def _(slt):
    slt
    return


@app.cell
def _(slt):
    summary = {
        "root keys": slt.keys(),
        "root groups": slt.groups(),
        "root raw datasets": slt.datasets(),
        "slothpy semantic groups": slt.slothpy_groups(),
        "raw groups": slt.raw_groups(),
    }
    summary
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 10. Reading a semantic group as xarray

    When a group is a valid SlothPy semantic group:

    - `slt["group_name"]` gives a `SltGroup`
    - `slt["group_name"].to_dataset()` gives an `xarray.Dataset`
    - `slt["group_name"].to_xarray()` gives the **primary** xarray object
    - `slt["group_name"]["variable_name"]` gives an `xarray.DataArray`

    This is the key bridge between SlothPy storage and the wider scientific Python ecosystem.
    """)
    return


@app.cell
def _(slt):
    mag = slt["magnetisation_001"]
    mag
    return (mag,)


@app.cell
def _(mag):
    {
        "group": mag,
        "group type": mag.type,
        "group primary": mag.primary,
        "group keys": mag.keys(),
        "group dimensions": mag.dimensions(),
        "group variables": mag.variables(),
        "group coordinates": mag.coordinates(),
    }
    return


@app.cell
def _(mag):
    mag_dataset = mag.to_dataset()
    mag_primary = mag.to_xarray()
    temperature_coord = mag["temperature"]
    orientation_weight = mag["orientation_weight"]
    return mag_dataset, mag_primary, orientation_weight, temperature_coord


@app.cell
def _(mag_dataset):
    mag_dataset
    return


@app.cell
def _(mag_primary):
    mag_primary
    return


@app.cell
def _(orientation_weight, temperature_coord):
    {
        "temperature coordinate": temperature_coord,
        "orientation_weight": orientation_weight,
    }
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 11. Typical xarray operations

    Because semantic groups open as xarray objects, you can immediately use:

    - `.sel(...)`
    - `.isel(...)`
    - `.mean(...)`
    - `.weighted(...)`
    - `.to_dataframe()`
    - plotting, broadcasting, alignment, and all the rest of the xarray ecosystem

    Below are a few practical examples.
    """)
    return


@app.cell
def _(mag_primary, orientation_weight):
    subset_temperature = mag_primary.sel(temperature=5.0, method="nearest")
    powder_average = mag_primary.weighted(orientation_weight).mean("orientation")
    field_slice = mag_primary.sel(field=3.0, method="nearest")
    return field_slice, powder_average, subset_temperature


@app.cell
def _(subset_temperature):
    subset_temperature
    return


@app.cell
def _(powder_average):
    powder_average
    return


@app.cell
def _(field_slice):
    field_slice
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 12. Converting to a dataframe

    This is useful for:

    - quick tables,
    - export,
    - integration with pandas/polars workflows,
    - feeding plotting libraries that prefer tabular input.
    """)
    return


@app.cell
def _(mag, powder_average):
    df_primary = mag.to_dataframe().reset_index()
    df_powder = powder_average.to_dataframe(name="magnetisation").reset_index()
    return df_powder, df_primary


@app.cell
def _(df_primary):
    df_primary.head(12)
    return


@app.cell
def _(df_powder):
    df_powder.head(12)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 13. Lazy loading and optional chunking

    Semantic groups are opened through xarray, so access is naturally compatible with lazy workflows.

    If you want chunked/Dask-backed opening, you can request chunks:

    ```python
    mag_chunked = slt.group("magnetisation_001", chunks={"field": 4}).to_xarray()
    ```

    This is especially useful for large multidimensional datasets.

    In the next cell, we try it if Dask-backed chunking is available.
    """)
    return


@app.cell
def _(slt):
    try:
        mag_chunked = slt.group("magnetisation_001", chunks={"field": 4}).to_xarray()
        chunk_demo = {
            "status": "success",
            "object": mag_chunked,
            "chunks": getattr(mag_chunked, "chunks", None),
        }
    except Exception as exc:
        mag_chunked = None
        chunk_demo = {
            "status": "not available in this environment",
            "reason": repr(exc),
        }

    chunk_demo
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 14. Re-opening the file

    A typical workflow is:

    1. Create or compute data,
    2. save it to `.slt`,
    3. later open the same file again.

    This is done with:

    ```python
    slt2 = open_slt_file("demo_tutorial.slt")
    ```
    """)
    return


@app.cell
def _(SLT_PATH, open_slt_file):
    slt2 = open_slt_file(SLT_PATH)
    slt2
    return (slt2,)


@app.cell
def _(slt2):
    {
        "keys": slt2.keys(),
        "groups": slt2.groups(),
        "semantic groups": slt2.slothpy_groups(),
    }
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 15. Low-level HDF5 escape hatch

    Sometimes you want direct HDF5 access, for example to:

    - inspect raw structure,
    - interoperate with external HDF5 code,
    - perform advanced custom operations not covered by the high-level API.

    For that, SlothPy provides:

    ```python
    with slt.open_hdf5("r") as h5:
    ...
    ```
    """)
    return


@app.cell
def _(slt):
    with slt.open_hdf5("r") as h5:
        root_items = list(h5.keys())
        root_item_types = {name: type(h5[name]).__name__ for name in h5.keys()}
        mag_group_items = list(h5["magnetisation_001"].keys())

    {
        "root_items": root_items,
        "root_item_types": root_item_types,
        "magnetisation_001 raw group members": mag_group_items,
    }
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 16. Structured introspection with `to_node()`

    The high-level objects can also be converted to structured node descriptions.

    This is useful for:

    - GUIs,
    - custom viewers,
    - tree inspection,
    - serialization of metadata summaries.
    """)
    return


@app.cell
def _(mag, slt):
    file_node = slt.to_node()
    group_node = mag.to_node()
    return file_node, group_node


@app.cell
def _(file_node):
    file_node
    return


@app.cell
def _(group_node):
    group_node
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 17. Deleting raw content

    The convenient delete operations are:

    ```python
    del slt["root_dataset"]
    del slt["raw_group"]["dataset"]
    del slt["group_or_dataset"]
    ```

    For semantic SlothPy groups, the recommended operation is deleting/replacing the **whole group** from file level, not mutating pieces of it as raw HDF5 content.

    Below we demonstrate deletion on a temporary raw dataset.
    """)
    return


@app.cell
def _(slt):
    slt["scratch"]["temporary"] = [99, 98, 97]
    before = slt["scratch"].keys()
    del slt["scratch"]["temporary"]
    after = slt["scratch"].keys()

    {
        "scratch keys before deletion": before,
        "scratch keys after deletion": after,
    }
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 18. Important note about lazy file handles

    xarray objects often hold **lazy references** to the underlying file.

    That is a feature, but it also means:

    - if you open a semantic group lazily,
    - and then try to mutate the same `.slt` file immediately,
    - backend file handles may still exist.

    Your implementation includes cache release logic before write operations, which helps a lot in notebooks.

    ### Still, best practice is:

    - close xarray objects when you are done:

    ```python
    obj.close()
    ```

    - especially before **rewriting**, **deleting**, or **replacing** data in the same file
    - and **never mutate the same file while an active Dask computation is reading it**

    This is the safest mental model:

    1. **open lazily**
    2. **analyze / compute**
    3. **close / materialize if needed**
    4. **then write**
    """)
    return


@app.cell
def _(
    mag_dataset,
    mag_primary,
    orientation_weight,
    powder_average,
    subset_temperature,
    temperature_coord,
):
    to_close = [
        mag_dataset,
        mag_primary,
        temperature_coord,
        orientation_weight,
        subset_temperature,
        powder_average,
    ]

    close_report = []
    for obj in to_close:
        name = getattr(obj, "name", type(obj).__name__)
        try:
            obj.close()
            close_report.append((name, "closed"))
        except Exception as exc:
            close_report.append((name, f"no close / skipped: {exc!r}"))

    close_report
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 19. Summary of the public workflow

    ### Raw user content

    Use the simple mapping API:

    ```python
    slt["dataset"] = data
    slt["group/dataset"] = data
    slt["group"]["dataset"] = data
    ```

    ### Reading semantic scientific data

    If a group is a valid SlothPy semantic group:

    ```python
    group = slt["magnetisation_001"]
    dataset = group.to_dataset()
    array = group.to_xarray()
    temperature = group["temperature"]
    ```

    ### Scientific operations

    Once you have xarray objects, use the xarray ecosystem directly:

    ```python
    array.sel(...)
    array.isel(...)
    array.mean(...)
    array.weighted(...).mean(...)
    array.to_dataframe()
    ```

    ### HDF5 escape hatch

    For advanced low-level operations:

    ```python
    with slt.open_hdf5("r") as h5:
    ...
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 20. Final remarks

    This `.slt` design provides a balance between:

    - **simplicity for notebook users**
    - **good scientific Python interoperability**
    - **transparent HDF5 storage**
    - **rich multidimensional data access through xarray**

    In other words:

    - **HDF5** gives the storage layer,
    - **SlothPy** gives the domain-aware API,
    - **xarray** gives the scientific data model.

    That combination is a strong foundation for future SlothPy workflows.
    """)
    return


@app.cell
def _(slt):
    # Add demo for del
    del slt["notes"].attrs["kut"]
    # del group/dataset etc.
    return


if __name__ == "__main__":
    app.run()
