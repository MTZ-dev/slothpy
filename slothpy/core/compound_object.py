from os import path
from typing import Tuple, Union
import h5py
from h5py import File, Group, Dataset
from numpy import ndarray, array, float64, int64
import numpy as np
from ._slothpy_exceptions import (
    SltFileError,
    SltCompError,
    SltSaveError,
    SltReadError,
    SltInputError,
    SltPlotError,
)
from slothpy.general_utilities._constants import (
    RED,
    GREEN,
    YELLOW,
    BLUE,
    PURPLE,
    RESET,
)
from slothpy.magnetism._g_tensor import _g_tensor_and_axes_doublet
from slothpy.magnetism._magnetisation import _mth, _mag_3d
from slothpy.magnetism.susceptibility import _chitht, chit_tensorht, chit_3d
from slothpy.general_utilities._grids_over_hemisphere import (
    _lebedev_laikov_grid,
)
from slothpy.general_utilities.io import (
    _group_exists,
    get_soc_energies_cm_1,
    get_states_magnetic_momenta,
    get_states_total_angular_momneta,
    get_total_angular_momneta_matrix,
    get_magnetic_momenta_matrix,
)
from slothpy.magnetism.zeeman import (
    zeeman_splitting,
    get_zeeman_matrix,
    hemholtz_energyth,
    hemholtz_energy_3d,
)
from slothpy.angular_momentum.pseudo_spin_ito import (
    get_decomposition_in_z_total_angular_momentum_basis,
    get_decomposition_in_z_magnetic_momentum_basis,
    ito_real_decomp_matrix,
    ito_complex_decomp_matrix,
    get_soc_matrix_in_z_magnetic_momentum_basis,
    get_soc_matrix_in_z_total_angular_momentum_basis,
    get_zeeman_matrix_in_z_magnetic_momentum_basis,
    get_zeeman_matrix_in_z_total_angular_momentum_basis,
    matrix_from_ito_complex,
    matrix_from_ito_real,
)
from slothpy.general_utilities._math_expresions import _normalize_grid_vectors
from slothpy.general_utilities._auto_tune import _auto_tune

# Experimental imports for plotting
from cycler import cycler
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
import matplotlib.colors
import matplotlib.cm
import matplotlib.gridspec
from matplotlib.animation import PillowWriter
from matplotlib.widgets import Slider

# faster plot
import matplotlib.style as mplstyle

mplstyle.use("fast")
mpl.rcParams["path.simplify"] = True
mpl.rcParams["path.simplify_threshold"] = 1.0

mpl.use("Qt5Agg")

# Promote set_plain_error_reporting_mode and set_default_error_reporting_mode and rise it from system to main with doc string

###To do: orinetation print in zeeman splitting
###       Hemholtz 3D plot, animate 3D all properties,
###       coloured prints and errors in terminal, numpy docstrings,
###       new diemnsions 3D, np.arrays of orient in decompositions

### MATRIX FROM ITO FIX MATRIX TYPE!!!!

### Use os.cpu_count() to determine max numeber of cpu to be used
### MPI memory management chunksize and how to minimize memory per process not sharing with the main one

### Effective charge distribution from CFPs.

### Check files before computations to avoid errors after resulting with no return

### change algorithm for 3d sus to improve memory usage (probably dont store whole stencils but compute elements on the fly, maybe with
# properly constructed generator)

### Add shared memory support as in zeeman_shared_1leak xd but learn more about this, use shared memory manger, and probably add index to the iterator
# https://docs.python.org/3/library/multiprocessing.shared_memory.html, wrapper should form a tuple pointing to the  shared memory object (but try if x is something to try this)
# after tests you should restart computer to cler the leaks and buffers

## You should probably move loops over fields to the new iterator with shared memory arrays to speed everything up

## get_soc_magnetic_momenta_and_energies_from_hdf5 should be out of mag_3d there are seconds wasted for reading over and over


class Compound:
    """
    The core object constituting the API and access to the all methods.
    """

    @classmethod
    def _new(cls, filepath: str, filename: str):
        """
        This is a private method for initializing the Compound object that
        should be only used by the creation_functions.

        Parameters
        ----------
        filepath : str
            A path of the file that will be associated with the created
            instance of the Compound class.
        filename : str
            A name of the file that will be associated with the created
            instance of the Compound class.

        Returns
        -------
        Compound
            An instance of the Compound class.
        """

        filename += ".slt"
        hdf5_file = path.join(filepath, filename)
        obj = super().__new__(cls)
        obj._hdf5 = hdf5_file
        obj._get_hdf5_groups_datasets_and_attributes()

        return obj

    def __new__(cls, *args, **kwargs) -> None:
        """
        The definition of this method prevents direct instantialization of the
        Compound class.

        Raises
        ------
        TypeError
            Prevents Compound() from working.
        """

        raise TypeError(
            "The Compound object should not be instantiated "
            "directly. Use a Compound creation function instead."
        )

    def __repr__(self) -> str:
        """
        Performs the operation __repr__.

        Creates a representation of the Compound object using names and
        attributes of the groups contained in the associated .slt file.

        Returns
        -------
        str
            A representation in terms of the contents of the .slt file.
        """

        representation = (
            RED
            + "Compound "
            + RESET
            + "from "
            + GREEN
            + "File "
            + RESET
            + f'"{self._hdf5}" with the following '
            + BLUE
            + "Groups "
            + RESET
            + "of data:\n"
        )
        for group, attributes in self._groups.items():
            representation += BLUE + f"{group}" + RESET + f": {attributes}\n"

        if self._datasets:
            representation += "and " + PURPLE + "Datasets" + RESET + ":\n"
            for dataset in self._datasets:
                representation += PURPLE + f"{dataset}\n" + RESET

        return representation

    # Set __str__ the same as an object representation using __repr__.
    __str__ = __repr__

    def __setitem__(
        self,
        key: Union[
            str,
            Tuple[str, str],
            Tuple[str, str, str],
            Tuple[str, str, str, str],
        ],
        value: ndarray,
    ) -> None:
        """
        Performs the operation __setitem__.

        Provides a convenient method for setting groups and datasets in the
        .slt file associated with a Compund instance in an array-like manner.

        Parameters
        ----------
        key : Union[str, Tuple[str, str], Tuple[str, str, str],
          Tuple[str, str, str]]
            A string or a 2/3/4-tuple of strings representing a dataset or
            group/dataset/dataset atribute/group atribute (Description),
            respectively, to be created or added (to the existing group).
        value : np.ndarray
            An ArrayLike structure (can be converted to np.ndarray) that will
            be stored in the dataset or group/dataset provided by the key.

        Raises
        ------
        SltSaveError
            If setting the data set was unsuccessful.
        KeyError
            If the key is not a string or 2-tuple of strings.
        """
        value = array(value)

        if isinstance(key, str):
            self._set_single_dataset(key, value)
        elif (
            isinstance(key, tuple)
            and (len(key) in [2, 3, 4])
            and all(isinstance(k, str) for k in key)
        ):
            self._set_group_and_dataset(key, value)
        else:
            raise KeyError(
                "Invalid key type. It has to be str or a 2/3/4-tuple of str."
            )

    def __getitem__(self, key: Union[str, Tuple[str, str]]) -> ndarray:
        """
        Performs the operation __getitem__.

        Provides a convenient method for getting datasets from the .slt file
        associated with a Compund instance in an array-like manner.

        Parameters
        ----------
        key : Union[str, Tuple[str, str], Tuple[str, str, str]]
            A string or a 2-tuple of strings representing a dataset or
            group/dataset, respectively, to be read from the .slt file.

        Returns
        -------
        ndarray
            An array contained in the dataset associated with the provided key.

        Raises
        ------
        SltReadError
            If reading the data from dataset set was unsuccessful.
        KeyError
            If the key is not a string or 2-tuple of strings.
        """
        if isinstance(key, str):
            return self._get_data_from_dataset(key)

        if (
            isinstance(key, tuple)
            and len(key) >= 2
            and all(isinstance(k, str) for k in key)
        ):
            return self._get_data_from_group_dataset(key)

        else:
            raise KeyError(
                "Invalid key type. It has to be str or 2-tuple of str."
            )

    def _get_hdf5_groups_datasets_and_attributes(self):
        self._groups = {}
        self._datasets = []

        def collect_objects(name, obj):
            if isinstance(obj, Group):
                self._groups[name] = dict(obj.attrs)
            elif isinstance(obj, Dataset):
                self._datasets.append(name)

        with File(self._hdf5, "r") as file:
            file.visititems(collect_objects)

    def _set_single_dataset(self, name: str, value: ndarray):
        try:
            with File(self._hdf5, "r+") as file:
                new_dataset = file.create_dataset(
                    name, shape=value.shape, dtype=value.dtype
                )
                new_dataset[:] = value[:]
            self._get_hdf5_groups_datasets_and_attributes()
        except Exception as exc:
            raise SltSaveError(
                self._hdf5,
                exc,
                message=f'Failed to set a Dataset: "{name}" in the .slt file',
            ) from None

    def _set_group_and_dataset(
        self,
        names: Union[
            Tuple[str, str], Tuple[str, str, str], Tuple[str, str, str, str]
        ],
        value: ndarray,
    ):
        try:
            with File(self._hdf5, "r+") as file:
                if names[0] in file and isinstance(file[names[0]], Group):
                    group = file[names[0]]
                else:
                    group = file.create_group(names[0])
                if len(names) == 4:
                    group.attrs["Description"] = names[3]
                new_dataset = group.create_dataset(
                    names[1], shape=value.shape, dtype=value.dtype
                )
                new_dataset[:] = value[:]
                if len(names) >= 3:
                    new_dataset.attrs["Description"] = names[2]
            self._get_hdf5_groups_datasets_and_attributes()
        except Exception as exc:
            raise SltSaveError(
                self._hdf5,
                exc,
                message=(
                    f'Failed to set a Dataset: "{names[1]}" within the Group:'
                    f' "{names[0]}" in the .slt file'
                ),
            ) from None

    def _get_data_from_dataset(self, name: str) -> ndarray:
        try:
            with File(self._hdf5, "r") as file:
                value = file[name][:]
        except Exception as exc:
            raise SltReadError(
                self._hdf5,
                exc,
                message=(
                    f'Failed to get a Dataset: "{name}" from the .slt file'
                ),
            ) from None

        return value

    def _get_data_from_group_dataset(self, names: Tuple[str, str]) -> ndarray:
        try:
            with File(self._hdf5, "r") as file:
                value = file[names[0]][names[1]][:]
        except Exception as exc:
            raise SltReadError(
                self._hdf5,
                exc,
                message=(
                    f'Failed to get a Dataset: "{names[0]}/{names[1]}" from'
                    " the .slt file"
                ),
            ) from None

        return value

    def delete_group_dataset(self, first: str, second: str = None) -> None:
        """
        Deletes a group/dataset provided its full name/path from the .slt file.

        Parameters
        ----------
        first : str
            A name of the gorup or dataset to be deleted.
        second : str, optional
            A name of the particular dataset inside the group from the first
            argument to be deleted.

        Raises
        ------
        SltFileError
            If the deletion is unsuccessful.
        """

        try:
            with File(self._hdf5, "r+") as file:
                if second is None:
                    del file[first]
                else:
                    del file[first][second]
        except Exception as exc:
            raise SltFileError(
                self._hdf5,
                exc,
                message=(
                    f'Failed to delete  "{first}"'
                    + (f"/{second}" if second is not None else "")
                    + " from the .slt file"
                ),
            ) from None

        self._get_hdf5_groups_datasets_and_attributes()

    def calculate_g_tensor_and_axes_doublet(
        self, group: str, doublets: ndarray[int64], slt: str = None
    ) -> Tuple[ndarray[float64], ndarray[float64]]:
        """
        Calculates pseudo-g-tensor components (for S = 1/2) and
        main magnetic axes for a given list of doublet states.

        Parameters
        ----------
        group : str
            Name of a group containing results of relativistic ab initio
            calculations used for the computation of g-tensors.
        doublets : ndarray[int64]
            ArrayLike structure (can be converted to numpy.NDArray) of integers
            corresponding to doublet labels (numbers).
        slt : str, optional
            If given, the results will be saved using this name to the .slt
            file with the sufix: _g_tensors_axes, by default None.

        Returns
        -------
        Tuple[ndarray[float64], ndarray[float64]]
            The first array (g_tensor_list) contains a list g-tensors in
            a format [doublet_number, gx, gy, gz], the second one
            (magnetic_axes_list) contains respective rotation matrices.

        Raises
        ------
        SltSaveError
            If the name of the group already exists in the .slt file.
        SltInputError
            If input ArrayLike data cannot be converted to numpy.NDArrays.
        SltInputError
            If doublets are not one-diemsional array.
        SltCompError
            If the calculation of g-tensors is unsuccessful.
        SltFileError
            If the program is unable to correctly save results to .slt file.

        Notes
        -----
        Magnetic axes are returned in the form of rotation matrices that
        diagonalise the Abragam-Bleaney tensor (G = gg.T). Coordinates of the
        main axes XYZ in the initial xzy frame are columns of such matrices
        (0-X, 1-Y, 2-Z).
        """

        if slt is not None:
            slt_group_name = f"{slt}_g_tensors_axes"
            if _group_exists(self._hdf5, slt_group_name):
                raise SltSaveError(
                    self._hdf5,
                    NameError(""),
                    message="Unable to save the results. "
                    + BLUE
                    + "Group "
                    + RESET
                    + '"'
                    + BLUE
                    + slt_group_name
                    + RESET
                    + '" '
                    + "already exists. Delete it manually",
                ) from None
        try:
            doublets = array(doublets, dtype=int64)
        except Exception as exc:
            raise SltInputError(exc) from None

        if doublets.ndim != 1:
            raise SltInputError(
                ValueError("The list of doublets has to be a 1D array.")
            ) from None

        try:
            (
                g_tensor_list,
                magnetic_axes_list,
            ) = _g_tensor_and_axes_doublet(self._hdf5, group, doublets)
        except Exception as exc:
            raise SltCompError(
                self._hdf5,
                exc,
                "Failed to compute g-tensors and main magnetic axes from "
                + BLUE
                + "Group "
                + RESET
                + '"'
                + BLUE
                + f"{group}"
                + RESET
                + '".',
            ) from None

        if slt is not None:
            try:
                self[
                    slt_group_name,
                    f"{slt}_g_tensors",
                    (
                        "Dataset containing number of doublet and respective"
                        f" g-tensors from Group {group}."
                    ),
                    (
                        f"Group({slt}) containing g-tensors of doublets and"
                        f" their magnetic axes calculated from Group: {group}."
                    ),
                ] = g_tensor_list[:, :]

                self[
                    slt_group_name,
                    f"{slt}_axes",
                    (
                        "Dataset containing rotation matrices from the initial"
                        " coordinate system to the magnetic axes of respective"
                        f" g-tensors from Group: {group}."
                    ),
                ] = magnetic_axes_list[:, :, :]
            except Exception as exc:
                raise SltFileError(
                    self._hdf5,
                    exc,
                    "Failed to save g-tensors and magnetic axes to "
                    + BLUE
                    + "Group "
                    + RESET
                    + '"'
                    + BLUE
                    + slt_group_name
                    + RESET
                    + '".',
                ) from None

        return g_tensor_list, magnetic_axes_list

    def calculate_mth(
        self,
        group: str,
        fields: ndarray[float64],
        grid: Union[int, ndarray[float64]],
        temperatures: ndarray[float64],
        states_cutoff: int = 0,
        number_cpu: int = 0,
        number_threads: int = 1,
        slt: str = None,
        autotune: bool = False,
        _autotune_size: int = 2,
    ) -> ndarray[float64]:
        """
        Calculates powder-averaged or directional molar magnetisation M(T,H)
        for a given list of temperature and field values.

        Parameters
        ----------
        group : str
            Name of a group containing results of relativistic ab initio
            calculations used for the computation of the magnetisation.
        fields : ndarray[float64]
            ArrayLike structure (can be converted to numpy.NDArray) of field
            values (T) at which magnetisation will be computed.
        grid : Union[int, ndarray[float64]]
            If the grid is set to an integer from 0-11 then the prescribed
            Lebedev-Laikov grids over hemisphere will be used (see
            grids_over_hemisphere documentation), otherwise, user can provide
            an ArrayLike structure (can be converted to numpy.NDArray) with the
            convention: [[direction_x, direction_y, direction_z, weight],...]
            for powder-averaging. If one wants a calculation for a single,
            particular direction the list has to contain one entry like this:
            [[direction_x, direction_y, direction_z, 1.]].
        temperatures : ndarray[float64]
            ArrayLike structure (can be converted to numpy.NDArray) of
            temeperature values (K) at which magnetisation will be computed.
        states_cutoff : int, optional
            Number of states that will be taken into account for construction
            of Zeeman Hamiltonian. If set to zero, all available states from
            the file will be used., by default 0
        number_cpu : int, optional
            Number of logical CPUs to be assigned to perform the calculation.
            If set to zero, all available CPUs will be used., by default 0
        number_threads : int, optional
            Number of threads used in a multithreaded implementation of linear
            algebra libraries used during the calculation. Higher values
            benefit from the increasing size of matrices (states_cutoff) over
            the parallelization over CPUs., by default 1
        slt : str, optional
            If given the results will be saved in a group of this name to .slt
            file with sufix: _magnetisation., by default None
        autotune : bool, optional
            If True the program will automatically try to choose the best
            number of threads (and therefore parallel processes), for the given
            number of CPUs, to be used during the calculation. Note that this
            process can take a significant amount of time, so start to use it
            with medium-sized calculations (e.g. for states_cutoff > 400 with
            dense grids or a higher number of field values) where it becomes
            a necessity., by default False

        Returns
        -------
        ndarray[float64]
            The resulting mth_array gives magnetisation in Bohr magnetons and
            is in the form [temperatures, fields] - the first dimension runs
            over temperature values, and the second over fields.

        Raises
        ------
        SltSaveError
            If the name of the group already exists in the .slt file.
        SltInputError
            If input ArrayLike data cannot be converted to numpy.NDArrays.
        SltInputError
            If fields are not a one-diemsional array.
        SltInputError
            If temperatures are not a one-diemsional array.
        SltCompError
            If autotuning a number of processes and threads is unsuccessful.
        SltCompError
            If the calculation of magnetisation is unsuccessful.
        SltFileError
            If the program is unable to correctly save results to .slt file.

        See Also
        --------
        slothpy.lebedev_laikov_grid : For the description of the prescribed
        Lebedev-Laikov grids

        Notes
        -----
        Here, (number_cpu // number_threads) parallel processes are used to
        distribute the workload over the provided field values.
        """

        if slt is not None:
            slt_group_name = f"{slt}_magnetisation"
            if _group_exists(self._hdf5, slt_group_name):
                raise SltSaveError(
                    self._hdf5,
                    NameError(""),
                    message="Unable to save the results. "
                    + BLUE
                    + "Group "
                    + RESET
                    + '"'
                    + BLUE
                    + slt_group_name
                    + RESET
                    + '" '
                    + "already exists. Delete it manually.",
                ) from None
        try:
            fields = array(fields, dtype=float64)
            temperatures = array(temperatures, dtype=float64)
        except Exception as exc:
            raise SltInputError(exc) from None

        if fields.ndim != 1:
            raise SltInputError(
                ValueError("The list of fields has to be a 1D array.")
            ) from None

        if temperatures.ndim != 1:
            raise SltInputError(
                ValueError("The list of temperatures has to be a 1D array.")
            ) from None

        if isinstance(grid, int):
            grid = _lebedev_laikov_grid(grid)
        else:
            grid = _normalize_grid_vectors(grid)

        if autotune:
            try:
                number_cpu, number_threads = _auto_tune(
                    self._hdf5,
                    group,
                    fields.size,
                    states_cutoff,
                    grid.shape[0],
                    temperatures.size,
                    number_cpu,
                    _autotune_size,
                )
            except Exception as exc:
                raise SltCompError(
                    self._hdf5,
                    exc,
                    "Failed to autotune a number of processes and threads to"
                    " the data within "
                    + BLUE
                    + "Group "
                    + RESET
                    + '"'
                    + BLUE
                    + f"{group}"
                    + RESET
                    + '".',
                ) from None

        try:
            mth_array = _mth(
                self._hdf5,
                group,
                fields,
                grid,
                temperatures,
                states_cutoff,
                number_cpu,
                number_threads,
            )
        except Exception as exc:
            raise SltCompError(
                self._hdf5,
                exc,
                "Failed to compute M(T,H) from "
                + BLUE
                + "Group "
                + RESET
                + '"'
                + BLUE
                + f"{group}"
                + RESET
                + '".',
            ) from None

        if slt is not None:
            try:
                self[
                    slt_group_name,
                    f"{slt}_mth",
                    (
                        "Dataset containing M(T,H) magnetisation (T - rows, H"
                        f" - columns) calculated from group: {group}."
                    ),
                    (
                        f"Group({slt}) containing M(T,H) magnetisation"
                        f" calculated from group: {group}."
                    ),
                ] = mth_array[:, :]
                self[
                    slt_group_name,
                    f"{slt}_fields",
                    (
                        "Dataset containing magnetic field H values used in"
                        f" simulation of M(T,H) from group: {group}."
                    ),
                ] = fields[:]
                self[
                    slt_group_name,
                    f"{slt}_temperatures",
                    (
                        "Dataset containing temperature T values used in"
                        f" simulation of M(T,H) from group: {group}."
                    ),
                ] = temperatures[:]
            except Exception as exc:
                raise SltFileError(
                    self._hdf5,
                    exc,
                    "Failed to save M(T,H) to "
                    + BLUE
                    + "Group "
                    + RESET
                    + '"'
                    + BLUE
                    + slt_group_name
                    + RESET
                    + '".',
                ) from None

        return mth_array

    def calculate_mag_3d(
        self,
        group: str,
        fields: ndarray[float64],
        spherical_grid: int,
        temperatures: ndarray[float64],
        states_cutoff: int = 0,
        number_cpu: int = 0,
        number_threads: int = 1,
        slt: str = None,
        autotune: bool = False,
        _autotune_size: int = 1,
    ) -> ndarray[float64]:
        """
        Calculates 3D magnetisation over a spherical grid for a given list of
        temperature and field values.

        Parameters
        ----------
        group : str
            Name of a group containing results of relativistic ab initio
            calculations used for the computation of the 3D magnetisation.
        fields : ndarray[float64]
            ArrayLike structure (can be converted to numpy.NDArray) of field
            values (T) at which 3D magnetisation will be computed.
        spherical_grid : int
            Controls the density of the angular grid for the 3D magnetisation
            calculation. A grid of dimension (spherical_grid*2*spherical_grid)
            for spherical angles theta [0, pi], and phi [0, 2*pi] will be used.
        temperatures : ndarray[float64]
            ArrayLike structure (can be converted to numpy.NDArray) of
            temperature values (K) at which magnetisation will be computed.
        states_cutoff : int, optional
            Number of states that will be taken into account for construction
            of Zeeman Hamiltonian. If set to zero, all available states from
            the file will be used., by default 0, by default 0
        number_cpu : int, optional
            Number of logical CPUs to be assigned to perform the calculation.
            If set to zero, all available CPUs will be used., by default 0
        number_threads : int, optional
            Number of threads used in a multithreaded implementation of linear
            algebra libraries used during the calculation. Higher values
            benefit from the increasing size of matrices (states_cutoff) over
            the parallelization over CPUs., by default 1, by default 1
        slt : str, optional
            If given the results will be saved in a group of this name to .slt
            file with sufix: _3d_magnetisation., by default None
        autotune : bool, optional
            If True the program will automatically try to choose the best
            number of threads (and therefore parallel processes), for the given
            number of CPUs, to be used during the calculation. Note that this
            process can take a significant amount of time, so start to use it
            with medium-sized calculations (e.g. for states_cutoff > 300 with
            dense grids or a higher number of field values) where it becomes
            a necessity., by default False

        Returns
        -------
        ndarray[float64]
            The resulting mag_3d_array gives magnetisation in Bohr magnetons
            and is in the form [coordinates, fields, temperatures, mesh, mesh]
            - the first dimension runs over coordinates (0-x, 1-y, 2-z), the
            second over field values, and the third over temperatures. The last
            two dimensions are in a form of meshgrids over theta and phi, ready
            for 3D plots as xyz.

        Raises
        ------
        SltSaveError
            If the name of the group already exists in the .slt file.
        SltInputError
            If input ArrayLike data cannot be converted to numpy.NDArrays.
        SltInputError
            If fields are not a one-diemsional array.
        SltInputError
            If temperatures are not a one-diemsional array.
        SltInputError
            If spherical_grid is not a positive integer.
        SltCompError
            If autotuning a number of processes and threads is unsuccessful.
        SltCompError
            If the calculation of 3D magnetisation is unsuccessful.
        SltFileError
            If the program is unable to correctly save results to .slt file.
        Notes
        -----
        Here, (number_cpu // number_threads) parallel processes are used to
        distribute the workload over len(fields)*2*shperical_grid**2 tasks. Be
        aware that the resulting arrays and computations can quickly consume
        much memory (e.g. for calculation with 100 field values 1-10 T, 300
        temperatures 1-300 K, and spherical_grid = 60, the resulting array will
        take 3*100*300*2*60*60*8 bytes = 5.184 GB).
        """

        if slt is not None:
            slt_group_name = f"{slt}_3d_magnetisation"
            if _group_exists(self._hdf5, slt_group_name):
                raise SltSaveError(
                    self._hdf5,
                    NameError(""),
                    message="Unable to save the results. "
                    + BLUE
                    + "Group "
                    + RESET
                    + '"'
                    + BLUE
                    + slt_group_name
                    + RESET
                    + '" '
                    + "already exists. Delete it manually.",
                ) from None
        try:
            temperatures = array(temperatures, dtype=np.float64)
            fields = array(fields, dtype=np.float64)
        except Exception as exc:
            raise SltInputError(exc) from None

        if fields.ndim != 1:
            raise SltInputError(
                ValueError("The list of fields has to be a 1D array.")
            ) from None

        if temperatures.ndim != 1:
            raise SltInputError(
                ValueError("The list of temperatures has to be a 1D array.")
            ) from None

        if (not isinstance(spherical_grid, int)) or spherical_grid <= 0:
            raise SltInputError(
                ValueError("Spherical grid has to be a positive integer.")
            ) from None

        if autotune:
            try:
                number_cpu, number_threads = _auto_tune(
                    self._hdf5,
                    group,
                    fields.size * 2 * spherical_grid**2,
                    states_cutoff,
                    1,
                    temperatures.size,
                    number_cpu,
                    _autotune_size,
                )
            except Exception as exc:
                raise SltCompError(
                    self._hdf5,
                    exc,
                    "Failed to autotune a number of processes and threads to"
                    " the data within "
                    + BLUE
                    + "Group "
                    + RESET
                    + '"'
                    + BLUE
                    + f"{group}"
                    + RESET
                    + '".',
                ) from None

        try:
            mag_3d_array = _mag_3d(
                self._hdf5,
                group,
                states_cutoff,
                fields,
                spherical_grid,
                temperatures,
                number_cpu,
                number_threads,
            )
        except Exception as exc:
            raise SltCompError(
                self._hdf5,
                exc,
                "Failed to compute 3D magnetisation from "
                + BLUE
                + "Group "
                + RESET
                + '"'
                + BLUE
                + f"{group}"
                + RESET
                + '".',
            ) from None

        if slt is not None:
            try:
                self[
                    slt_group_name,
                    f"{slt}_mag_3d",
                    (
                        "Dataset containing 3D magnetisation as meshgird"
                        " (0-x,1-y,2-z) arrays over sphere (xyz, field,"
                        " temperature, meshgrid, meshgrid) calculated from"
                        f" group: {group}."
                    ),
                    (
                        f"Group({slt}) containing 3D magnetisation calculated"
                        f" from group: {group}."
                    ),
                ] = mag_3d_array[:, :, :, :, :]
                self[
                    slt_group_name,
                    f"{slt}_fields",
                    (
                        "Dataset containing magnetic field H values used in"
                        f" simulation of 3D magnetisation from group: {group}."
                    ),
                ] = fields[:]
                self[
                    slt_group_name,
                    f"{slt}_temperatures",
                    (
                        "Dataset containing temperature T values used in"
                        f" simulation of 3D magnetisation from group: {group}."
                    ),
                ] = temperatures[:]
            except Exception as exc:
                raise SltFileError(
                    self._hdf5,
                    exc,
                    "Failed to save 3D magnetisation to "
                    + BLUE
                    + "Group "
                    + RESET
                    + '"'
                    + BLUE
                    + slt_group_name
                    + RESET
                    + '".',
                ) from None

        return mag_3d_array

    def calculate_chitht(
        self,
        group: str,
        temperatures: ndarray[float64],
        fields: ndarray[float64],
        number_of_points: int,
        delta_h: float64 = 0.0001,
        states_cutoff: int = 0,
        number_cpu: int = 0,
        number_threads: int = 1,
        exp: bool = False,
        T: bool = True,
        grid: Union[int, np.ndarray[float64]] = None,
        slt: str = None,
        autotune: bool = False,
        _autotune_size: int = 1,
    ) -> ndarray[float64]:
        """
        Calculates powder-averaged or directional molar magnetic susceptibility
        chi(T)(H,T) for a given list of field and temperatures values.

        Parameters
        ----------
        group : str
            Name of a group containing results of relativistic ab initio
            calculations used for the computation of the magnetisation.
        temperatures : ndarray[float64]
            ArrayLike structure (can be converted to numpy.NDArray) of
            temeperature values (K) at which magnetic susceptibility will
            be computed.
        fields : ndarray[float64]
            ArrayLike structure (can be converted to numpy.NDArray) of field
            values (T) at which magnetic susceptibility will be computed.
        number_of_points : int
            Controls the number of points for numerical differentiation over
            the magnetic field values using the finite difference method with
            a symmetrical stencil. The total number of used points =
            (2 * num_of_opints + 1), therefore 1 is a minimum value to obtain
            the first derivative using 3 points - including the value at the
            point at which the derivative is taken. In this regard, the value 0
            triggers the experimentalist model for susceptibility.
        delta_h : float64, optional
            Value of field step used for numerical differentiation using finite
            difference method. 0.0001 (T) = 1 Oe is recommended as starting
            point., by default 0.0001
        states_cutoff : int, optional
            Number of states that will be taken into account for construction
            of Zeeman Hamiltonian. If set to zero, all available states from
            the file will be used., by default 0
        number_cpu : int, optional
            Number of logical CPUs to be assigned to perform the calculation.
            If set to zero, all available CPUs will be used., by default 0
        number_threads : int, optional
            Number of threads used in a multithreaded implementation of linear
            algebra libraries used during the calculation. Higher values
            benefit from the increasing size of matrices (states_cutoff) over
            the parallelization over CPUs., by default 1
        exp : bool, optional
            Turns on the experimentalist model for magnetic susceptibility.,
            by default False
        T : bool, optional
            Results are returned as a product with temperature chiT(H,T).,
            by default True
        grid : Union[int, np.ndarray[float64]], optional
            If the grid is set to an integer from 0-11 then the prescribed
            Lebedev-Laikov grids over the hemisphere will be used (see
            grids_over_hemisphere documentation), otherwise, the user can
            provide an ArrayLike structure (can be converted to numpy.NDArray)
            with the convention: [[direction_x, direction_y, direction_z,
            weight],...] for powder-averaging. If one wants a calculation for a
            single, particular direction the list has to contain one entry like
            this: [[direction_x, direction_y, direction_z, 1.]]. If not given
            the average is taken over xyz directions, which is sufficient for a
            second rank tensor., by default None
        slt : str, optional
            If given the results will be saved in a group of this name to .slt
            file with sufix: _susceptibility., by default None, by default None
        autotune : bool, optional
            If True the program will automatically try to choose the best
            number of threads (and therefore parallel processes), for the given
            number of CPUs, to be used during the calculation. Note that this
            process can take a significant amount of time, so start to use it
            with medium-sized calculations (e.g. for states_cutoff > 600 with
            a higher number of field values) where it becomes a necessity.,
            by default False

        Returns
        -------
        ndarray[float64]
            The resulting chitht_array gives magnetic susceptibility (or
            product with temperature) in cm^3 (or * K) and is in the form
            [fields, temperatures] - the first dimension runs over field
            values, and the second over temperatures.

        Raises
        ------
        SltSaveError
            If the name of the group already exists in the .slt file
        SltInputError
            If input ArrayLike data cannot be converted to numpy.NDArrays
        SltInputError
            If temperatures are not a one-diemsional array.
        SltInputError
            If fields are not a one-diemsional array.
        SltInputError
            If the  number of points for finite difference method is not
            a possitive integer.
        SltInputError
            If the field step for the finite difference method is not
            a possitive real number.
        SltCompError
            If autotuning a number of processes and threads is unsuccessful.
        SltCompError
            If the calculation of magnetisation is unsuccessful.
        SltCompError
            If the calculation of magnetisation is unsuccessful
        SltFileError
            If the program is unable to correctly save results to .slt file.
        """
        if slt is not None:
            slt_group_name = f"{slt}_susceptibility"
            if _group_exists(self._hdf5, slt_group_name):
                raise SltSaveError(
                    self._hdf5,
                    NameError(""),
                    message="Unable to save the results. "
                    + BLUE
                    + "Group "
                    + RESET
                    + '"'
                    + BLUE
                    + slt_group_name
                    + RESET
                    + '" '
                    + "already exists. Delete it manually.",
                ) from None
        try:
            fields = array(fields, dtype=float64)
            temperatures = array(temperatures, dtype=float64)
        except Exception as exc:
            raise SltInputError(exc) from None

        if temperatures.ndim != 1:
            raise SltInputError(
                ValueError("The list of temperatures has to be a 1D array.")
            ) from None

        if fields.ndim != 1:
            raise SltInputError(
                ValueError("The list of fields has to be a 1D array.")
            ) from None

        if (not isinstance(number_of_points, int)) or number_of_points < 0:
            raise SltInputError(
                ValueError(
                    "The number of points for the finite difference method has"
                    " to be a possitive integer."
                )
            ) from None

        if (not isinstance(delta_h, float)) or delta_h <= 0:
            raise SltInputError(
                ValueError(
                    "The field step for finite difference method has to be a"
                    " possitive number."
                )
            ) from None

        if isinstance(grid, int):
            grid = _lebedev_laikov_grid(grid)
        elif grid is not None:
            grid = _normalize_grid_vectors(grid)

        if autotune:
            if exp or number_of_points == 0:
                num_to_parallel = fields.size
            else:
                num_to_parallel = (2 * number_of_points + 1) * fields.size

            if grid is None:
                grid_shape = 3
            else:
                grid_shape = grid.shape[0]

            try:
                number_cpu, number_threads = _auto_tune(
                    self._hdf5,
                    group,
                    num_to_parallel,
                    states_cutoff,
                    grid_shape,
                    temperatures.shape[0],
                    number_cpu,
                    _autotune_size,
                )
            except Exception as exc:
                raise SltCompError(
                    self._hdf5,
                    exc,
                    "Failed to autotune a number of processes and threads to"
                    " the data within "
                    + BLUE
                    + "Group "
                    + RESET
                    + '"'
                    + BLUE
                    + f"{group}"
                    + RESET
                    + '".',
                ) from None

        if T:
            chi_name = "chiT(H,T)"
            chi_file = "chit"
        else:
            chi_name = "chi(H,T)"
            chi_file = "chi"

        try:
            chitht_array = _chitht(
                self._hdf5,
                group,
                temperatures,
                fields,
                number_of_points,
                delta_h,
                states_cutoff,
                number_cpu,
                number_threads,
                exp,
                T,
                grid,
            )
        except Exception as exc:
            raise SltCompError(
                self._hdf5,
                exc,
                f"Failed to compute {chi_name} from "
                + BLUE
                + "Group "
                + RESET
                + '"'
                + BLUE
                + f"{group}"
                + RESET
                + '".',
            ) from None

        if slt is not None:
            try:
                self[
                    slt_group_name,
                    f"{slt}_{chi_file}ht",
                    (
                        f"Dataset containing {chi_name} magnetic"
                        " susceptibility (H - rows, T - columns) calculated"
                        f" from group: {group}."
                    ),
                    (
                        f"Group({slt}) containing {chi_name} magnetic"
                        f" susceptibility calculated from group: {group}."
                    ),
                ] = chitht_array[:, :]
                self[
                    slt_group_name,
                    f"{slt}_fields",
                    (
                        "Dataset containing magnetic field H values used in"
                        f" simulation of {chi_name} from group: {group}."
                    ),
                ] = fields[:]
                self[
                    slt_group_name,
                    f"{slt}_temperatures",
                    (
                        "Dataset containing temperature T values used in"
                        f" simulation of {chi_name} from group: {group}."
                    ),
                ] = temperatures[:]
            except Exception as exc:
                raise SltFileError(
                    self._hdf5,
                    exc,
                    f"Failed to save {chi_name} to "
                    + BLUE
                    + "Group "
                    + RESET
                    + '"'
                    + BLUE
                    + slt_group_name
                    + RESET
                    + '".',
                ) from None

        return chitht_array

    def calculate_chit_tensorht(
        self,
        group: str,
        temperatures: ndarray[float64],
        fields: ndarray[float64],
        number_of_points: int,
        delta_h: float64 = 0.0001,
        states_cutoff: int = 0,
        number_cpu: int = 0,
        number_threads: int = 1,
        exp: bool = False,
        T: bool = True,
        rotation: ndarray[float64] = None,
        slt: str = None,
        autotune: bool = False,
        _autotune_size: int = 1,
    ) -> ndarray[float64]:
        fields = array(fields, dtype=np.float64)
        temperatures = array(temperatures, dtype=np.float64)

        try:
            chit_tensorht_array = chit_tensorht(
                self._hdf5,
                group,
                temperatures,
                fields,
                number_of_points,
                delta_h,
                states_cutoff,
                number_cpu,
                number_threads,
                exp,
                T,
                rotation,
            )
        except Exception as e:
            error_type = type(e).__name__
            error_message = str(e)
            raise Exception(
                "Error encountered while trying to compute chi_tensor(H,T)"
                f" from file: {self._hdf5} - group {group}: {error_type}:"
                f" {error_message}"
            )

        if slt is not None:
            try:
                with h5py.File(self._hdf5, "r+") as file:
                    new_group = file.create_group(
                        f"{slt}_susceptibility_tensor"
                    )
                    new_group.attrs["Description"] = (
                        f"Group({slt}) containing chiT_tensor(H,T) Van Vleck"
                        " susceptibility tensor calculated from group:"
                        f" {group}."
                    )
                    chit_tensorht_dataset = new_group.create_dataset(
                        f"{slt}_chit_tensorht",
                        shape=(
                            chit_tensorht_array.shape[0],
                            chit_tensorht_array.shape[1],
                            3,
                            3,
                        ),
                        dtype=np.float64,
                    )
                    chit_tensorht_dataset.attrs["Description"] = (
                        "Dataset containing chiT_tensor(H,T) Van Vleck"
                        " susceptibility tensor (H, T, 3, 3) calculated from"
                        f" group: {group}."
                    )
                    fields_dataset = new_group.create_dataset(
                        f"{slt}_fields",
                        shape=(fields.shape[0],),
                        dtype=np.float64,
                    )
                    fields_dataset.attrs["Description"] = (
                        "Dataset containing magnetic field H values used in"
                        " simulation of chiT_tensor(H,T) Van Vleck"
                        f" susceptibility tensor from group: {group}."
                    )
                    temperatures_dataset = new_group.create_dataset(
                        f"{slt}_temperatures",
                        shape=(temperatures.shape[0],),
                        dtype=np.float64,
                    )
                    temperatures_dataset.attrs["Description"] = (
                        "Dataset containing temperature T values used in"
                        " simulation of chiT_tensor(H,T) Van Vleck"
                        f" susceptibility tensor from group: {group}."
                    )

                    chit_tensorht_dataset[:, :] = chit_tensorht_array[:, :]
                    fields_dataset[:] = fields[:]
                    temperatures_dataset[:] = temperatures[:]

                self._get_hdf5_groups_datasets_and_attributes()

            except Exception as e:
                error_type = type(e).__name__
                error_message = str(e)
                raise Exception(
                    "Error encountered while trying to save chiT(H,T) to"
                    f" file: {self._hdf5} - group {slt}: {error_type}:"
                    f" {error_message}"
                )

        return chit_tensorht_array

    def soc_energies_cm_1(
        self, group: str, num_of_states: int = None, slt: str = None
    ) -> np.ndarray[np.float64]:
        """Returns energies in cm^(-1) of the given number of Spin-Orbit states.

        Args:
            group (str): Name of a group containing results of relativistic ab initio calculations used for the SOC energy simulation.
            num_of_states (int, optional): Number of states (couted from the ground one) whose energies will be calculated. If None or 0 all accessible states will be considered. Defaults to None.
            slt (str, optional): If not None the results will be saved using this name to .slt file with sufix: _soc_energies. Defaults to None.
        Raises:
            Exception: If the program is unable to get energies from .slt file.
            Exception: If the program is unable to correctly save the results to .slt file.

        Returns:
            np.ndarray[np.float64]: The resulting array gives SOC energies in cm^(-1) with indices corresponding to SO-state numbers.
        """

        try:
            soc_energies_array = get_soc_energies_cm_1(
                self._hdf5, group, num_of_states
            )
        except Exception as e:
            error_type = type(e).__name__
            error_message = str(e)
            raise Exception(
                "Error encountered while trying to get SOC energies from file:"
                f" {self._hdf5} - group {group}: {error_type}: {error_message}"
            )

        if slt is not None:
            try:
                with h5py.File(self._hdf5, "r+") as file:
                    new_group = file.create_group(f"{slt}_soc_energies")
                    new_group.attrs["Description"] = (
                        f"Group({slt}) containing SOC (Spin-Orbit Coupling)"
                        f" energies calculated from group: {group}."
                    )
                    soc_energies_dataset = new_group.create_dataset(
                        f"{slt}_soc_energies",
                        shape=(soc_energies_array.shape[0],),
                        dtype=np.float64,
                    )
                    soc_energies_dataset.attrs["Description"] = (
                        "Dataset containing SOC (Spin-Orbit Coupling) energies"
                        f" calculated from group: {group}."
                    )

                    soc_energies_dataset[:] = soc_energies_array[:]

                self._get_hdf5_groups_datasets_and_attributes()

            except Exception as e:
                error_type = type(e).__name__
                error_message = str(e)
                raise Exception(
                    "Error encountered while trying to save SOC (Spin-Orbit"
                    f" Coupling) energies to file: {self._hdf5} - group {slt}:"
                    f" {error_type}: {error_message}"
                )

        return soc_energies_array

    def states_magnetic_momenta(
        self,
        group: str,
        states: Union[int, np.ndarray[int]] = None,
        rotation=None,
        slt: str = None,
    ):
        states = np.array(states)

        try:
            states, magnetic_momenta_array = get_states_magnetic_momenta(
                self._hdf5, group, states, rotation
            )
        except Exception as e:
            error_type = type(e).__name__
            error_message = str(e)
            raise Exception(
                "Error encountered while trying to get states magnetic"
                f" momenta from file: {self._hdf5} - group {group}:"
                f" {error_type}: {error_message}"
            )

        if slt is not None:
            try:
                with h5py.File(self._hdf5, "r+") as file:
                    new_group = file.create_group(
                        f"{slt}_states_magnetic_momenta"
                    )
                    new_group.attrs["Description"] = (
                        f"Group({slt}) containing states magnetic momenta"
                        f" calculated from group: {group}."
                    )
                    magnetic_momenta_dataset = new_group.create_dataset(
                        f"{slt}_magnetic_momenta",
                        shape=(
                            magnetic_momenta_array.shape[0],
                            magnetic_momenta_array.shape[1],
                        ),
                        dtype=np.float64,
                    )
                    magnetic_momenta_dataset.attrs["Description"] = (
                        "Dataset containing states magnetic momenta"
                        f" (0-x,1-y,2-z) calculated from group: {group}."
                    )
                    states_dataset = new_group.create_dataset(
                        f"{slt}_states",
                        shape=(states.shape[0],),
                        dtype=np.int64,
                    )
                    states_dataset.attrs["Description"] = (
                        "Dataset containing indexes of states used in"
                        f" simulation of magnetic momenta from group: {group}."
                    )

                    magnetic_momenta_dataset[:] = magnetic_momenta_array[:]
                    states_dataset[:] = states[:]

                self._get_hdf5_groups_datasets_and_attributes()

            except Exception as e:
                error_type = type(e).__name__
                error_message = str(e)
                raise Exception(
                    "Error encountered while trying to save states magnetic"
                    f" momenta to file: {self._hdf5} - group {slt}:"
                    f" {error_type}: {error_message}"
                )

        return magnetic_momenta_array

    def states_total_angular_momenta(
        self,
        group: str,
        states: np.ndarray = None,
        rotation=None,
        slt: str = None,
    ):
        states = np.array(states)

        try:
            (
                states,
                total_angular_momenta_array,
            ) = get_states_total_angular_momneta(
                self._hdf5, group, states, rotation
            )
        except Exception as e:
            error_type = type(e).__name__
            error_message = str(e)
            raise Exception(
                "Error encountered while trying to get states total angular"
                f" momenta from file: {self._hdf5} - group {group}:"
                f" {error_type}: {error_message}"
            )

        if slt is not None:
            try:
                with h5py.File(self._hdf5, "r+") as file:
                    new_group = file.create_group(
                        f"{slt}_states_total_angular_momenta"
                    )
                    new_group.attrs["Description"] = (
                        f"Group({slt}) containing states total angular momenta"
                        f" calculated from group: {group}."
                    )
                    total_angular_momenta_dataset = new_group.create_dataset(
                        f"{slt}_total_angular_momenta",
                        shape=(
                            total_angular_momenta_array.shape[0],
                            total_angular_momenta_array.shape[1],
                        ),
                        dtype=np.float64,
                    )
                    total_angular_momenta_dataset.attrs["Description"] = (
                        "Dataset containing states total angular momenta"
                        f" (0-x,1-y,2-z) calculated from group: {group}."
                    )
                    states_dataset = new_group.create_dataset(
                        f"{slt}_states",
                        shape=(states.shape[0],),
                        dtype=np.int64,
                    )
                    states_dataset.attrs["Description"] = (
                        "Dataset containing indexes of states used in"
                        " simulation of total angular momenta from group:"
                        f" {group}."
                    )

                    total_angular_momenta_dataset[
                        :
                    ] = total_angular_momenta_array[:]
                    states_dataset[:] = states[:]

                self._get_hdf5_groups_datasets_and_attributes()

            except Exception as e:
                error_type = type(e).__name__
                error_message = str(e)
                raise Exception(
                    "Error encountered while trying to save states total"
                    f" angular momenta to file: {self._hdf5} - group {slt}:"
                    f" {error_type}: {error_message}"
                )

        return total_angular_momenta_array

    def calculate_zeeman_splitting(
        self,
        group: str,
        states_cutoff: int,
        num_of_states: int,
        fields: np.ndarray[np.float64],
        grid: np.ndarray[np.float64],
        num_cpu: int,
        num_threads: int,
        average: bool = False,
        slt: str = None,
    ) -> np.ndarray[np.float64]:
        """Calculates directional or powder-averaged Zeeman splitting for a given number of states and list of field values.

        Args:
            group (str): Name of a group containing results of relativistic ab initio calculations used for the computation Zeeman splitting.
            states_cutoff (int): Number of states that will be taken into account for construction of Zeeman Hamiltonian.
            num_of_states (int): Number of states whose energy splitting will be given in result array.
            fields (np.ndarray[np.float64]): ArrayLike structure (can be converted to numpy.NDArray) of field values (T) at which Zeeman splitting will be computed.
            grid (np.ndarray[np.float64]): ArrayLike (can be converted to numpy.NDArray) list of directions in the form: [[direction_x, direction_y, direction_z], ...] for which Zeeman splitting will
                be calculated. If the grid is set to integer from 0-11 then the prescribed Lebedev-Laikov grids over hemisphere will be used
                (see grids_over_hemisphere documentation) for powder-averaging of energy splitting.
            num_cpu (int): Number of physical CPUs to be assigned to perform calculation.
            num_threads (int): Number of threads used in multithreaded implementation of the linear algebra libraries used during calculation. Values higher than 2 benefit
                with the increasing size of matrices (states_cutoff) over the MPI parallelization over field values.
            average (bool, optional): Switch on powder-averaging using list of directions and weights in the form of ArrayLike structure: [[direction_x, direction_y, direction_z, weight],...]. Defaults to False.
            slt (str, optional): If not None the results will be saved using this name to .slt file with sufix: _zeeman. Defaults to None.

        Raises:
            Exception: If the calculation of Zeeman splitting is unsuccessful.
            Exception: If the program is unable to correctly save the results to .slt file.

        Returns:
            np.ndarray[np.float64]: The resulting array gives Zeeman splitting of (num_of_states) energy levels in cm^(-1) for each direction (or average) in the form (orientations, fields, energies) - the first dimension
              runs over different orientations, the second over field values.
        """

        fields = np.array(fields)

        if isinstance(grid, int):
            grid = lebedev_laikov_grid(grid)
            average = True
        else:
            grid = normalize_grid_vectors(grid)

        try:
            zeeman_array = zeeman_splitting(
                self._hdf5,
                group,
                states_cutoff,
                num_of_states,
                fields,
                grid,
                num_cpu,
                num_threads,
                average,
            )
        except Exception as e:
            error_type = type(e).__name__
            error_message = str(e)
            raise Exception(
                "Error encountered while trying to compute Zeeman splitting"
                f" from file: {self._hdf5} - group {group}: {error_type}:"
                f" {error_message}"
            )

        if slt is not None:
            try:
                with h5py.File(self._hdf5, "r+") as file:
                    new_group = file.create_group(f"{slt}_zeeman_splitting")
                    new_group.attrs["Description"] = (
                        f"Group({slt}) containing Zeeman splitting calculated"
                        f" from group: {group}."
                    )
                    zeeman_splitting_dataset = new_group.create_dataset(
                        f"{slt}_zeeman",
                        shape=zeeman_array.shape,
                        dtype=np.float64,
                    )
                    if average:
                        zeeman_splitting_dataset.attrs["Description"] = (
                            "Dataset containing Zeeman splitting averaged"
                            " over grid of directions with shape: (field,"
                            f" energy) calculated from group: {group}."
                        )
                    else:
                        zeeman_splitting_dataset.attrs["Description"] = (
                            "Dataset containing Zeeman splitting with shape:"
                            " (orientation, field, energy) calculated from"
                            f" group: {group}."
                        )
                    fields_dataset = new_group.create_dataset(
                        f"{slt}_fields",
                        shape=(fields.shape[0],),
                        dtype=np.float64,
                    )
                    fields_dataset.attrs["Description"] = (
                        "Dataset containing magnetic field H values used in"
                        f" simulation of Zeeman splitting from group: {group}."
                    )
                    if average:
                        orientations_dataset = new_group.create_dataset(
                            f"{slt}_orientations",
                            shape=(grid.shape[0], grid.shape[1]),
                            dtype=np.float64,
                        )
                        orientations_dataset.attrs["Description"] = (
                            "Dataset containing magnetic field orientation"
                            " grid with weights used in simulation of"
                            f" averaged Zeeman splitting from group: {group}."
                        )
                        orientations_dataset[:] = grid[:]
                    else:
                        orientations_dataset = new_group.create_dataset(
                            f"{slt}_orientations",
                            shape=(grid.shape[0], 3),
                            dtype=np.float64,
                        )
                        orientations_dataset.attrs["Description"] = (
                            "Dataset containing orientations of magnetic"
                            " field used in simulation of Zeeman splitting"
                            f" from group: {group}."
                        )
                        orientations_dataset[:] = grid[:, :3]

                    zeeman_splitting_dataset[:, :] = zeeman_array[:, :]
                    fields_dataset[:] = fields[:]

                self._get_hdf5_groups_datasets_and_attributes()

            except Exception as e:
                error_type = type(e).__name__
                error_message = str(e)
                raise Exception(
                    "Error encountered while trying to save Zeeman splitting"
                    f" to file: {self._hdf5} - group {slt}: {error_type}:"
                    f" {error_message}"
                )

        return zeeman_array

    def total_angular_momenta_matrix(
        self,
        group: str,
        states_cutoff: np.int64 = None,
        rotation=None,
        slt: str = None,
    ):
        if (not isinstance(states_cutoff, int)) or (states_cutoff < 0):
            raise ValueError(
                "Invalid states cutoff, set it to positive integer or 0 for"
                " all states."
            )

        try:
            total_angular_momenta_matrix_array = (
                get_total_angular_momneta_matrix(
                    self._hdf5, group, states_cutoff, rotation
                )
            )
        except Exception as e:
            error_type = type(e).__name__
            error_message = str(e)
            raise Exception(
                "Error encountered while trying to get total angular momenta"
                f" matrix from file: {self._hdf5} - group {group}:"
                f" {error_type}: {error_message}"
            )

        if slt is not None:
            try:
                with h5py.File(self._hdf5, "r+") as file:
                    new_group = file.create_group(
                        f"{slt}_total_angular_momenta_matrix"
                    )
                    total_angular_momenta_matrix_dataset = (
                        new_group.create_dataset(
                            f"{slt}_total_angular_momenta_matrix",
                            shape=total_angular_momenta_matrix_array.shape,
                            dtype=np.complex128,
                        )
                    )
                    total_angular_momenta_matrix_dataset.attrs[
                        "Description"
                    ] = (
                        "Dataset containing total angular momenta matrix"
                        f" (0-x, 1-y, 2-z) calculated from group: {group}."
                    )
                    states_dataset = new_group.create_dataset(
                        f"{slt}_states",
                        shape=(total_angular_momenta_matrix_array.shape[1],),
                        dtype=np.int64,
                    )
                    states_dataset.attrs["Description"] = (
                        "Dataset containing states indexes of total angular"
                        f" momenta matrix from group: {group}."
                    )

                    total_angular_momenta_matrix_dataset[
                        :
                    ] = total_angular_momenta_matrix_array[:]
                    states_dataset[:] = np.arange(
                        total_angular_momenta_matrix_array.shape[1],
                        dtype=np.int64,
                    )

                self._get_hdf5_groups_datasets_and_attributes()

            except Exception as e:
                error_type = type(e).__name__
                error_message = str(e)
                raise Exception(
                    "Error encountered while trying to save total angular"
                    f" momenta matrix to file: {self._hdf5} - group {slt}:"
                    f" {error_type}: {error_message}"
                )

        return total_angular_momenta_matrix_array

    def magnetic_momenta_matrix(
        self,
        group: str,
        states_cutoff: np.ndarray = None,
        rotation=None,
        slt: str = None,
    ):
        if (not isinstance(states_cutoff, int)) or (states_cutoff < 0):
            raise ValueError(
                "Invalid states cutoff, set it to positive integer or 0 for"
                " all states."
            )

        try:
            magnetic_momenta_matrix_array = get_magnetic_momenta_matrix(
                self._hdf5, group, states_cutoff, rotation
            )
        except Exception as e:
            error_type = type(e).__name__
            error_message = str(e)
            raise Exception(
                "Error encountered while trying to get total angular momenta"
                f" matrix from file: {self._hdf5} - group {group}:"
                f" {error_type}: {error_message}"
            )

        if slt is not None:
            try:
                with h5py.File(self._hdf5, "r+") as file:
                    new_group = file.create_group(
                        f"{slt}_magnetic_momenta_matrix"
                    )
                    new_group.attrs["Description"] = (
                        f"Group({slt}) containing magnetic momenta calculated"
                        f" from group: {group}."
                    )
                    magnetic_momenta_matrix_dataset = new_group.create_dataset(
                        f"{slt}_magnetic_momenta_matrix",
                        shape=magnetic_momenta_matrix_array.shape,
                        dtype=np.complex128,
                    )
                    magnetic_momenta_matrix_dataset.attrs["Description"] = (
                        "Dataset containing magnetic momenta matrix (0-x,"
                        f" 1-y, 2-z) calculated from group: {group}."
                    )
                    states_dataset = new_group.create_dataset(
                        f"{slt}_states",
                        shape=(magnetic_momenta_matrix_array.shape[1],),
                        dtype=np.int64,
                    )
                    states_dataset.attrs["Description"] = (
                        "Dataset containing states indexes of magnetic"
                        f" momenta matrix from group: {group}."
                    )

                    magnetic_momenta_matrix_dataset[
                        :
                    ] = magnetic_momenta_matrix_array[:]
                    states_dataset[:] = np.arange(
                        magnetic_momenta_matrix_array.shape[1], dtype=np.int64
                    )

                self._get_hdf5_groups_datasets_and_attributes()

            except Exception as e:
                error_type = type(e).__name__
                error_message = str(e)
                raise Exception(
                    "Error encountered while trying to save magnetic momenta"
                    f" matrix to file: {self._hdf5} - group {slt}:"
                    f" {error_type}: {error_message}"
                )

        return magnetic_momenta_matrix_array

    def decomposition_in_z_magnetic_momentum_basis(
        self, group, start_state, stop_state, rotation=None, slt: str = None
    ):
        if (not isinstance(stop_state, int)) or (stop_state < 0):
            raise ValueError(
                "Invalid states number, set it to positive integer or 0 for"
                " all states."
            )

        try:
            decomposition = get_decomposition_in_z_magnetic_momentum_basis(
                self._hdf5, group, start_state, stop_state, rotation
            )
        except Exception as e:
            error_type = type(e).__name__
            error_message = str(e)
            raise Exception(
                'Error encountered while trying to get decomposition in "z"'
                " magnetic momentum basis of SOC matrix from file:"
                f" {self._hdf5} - group {group}: {error_type}: {error_message}"
            )

        if slt is not None:
            try:
                with h5py.File(self._hdf5, "r+") as file:
                    new_group = file.create_group(
                        f"{slt}_magnetic_decomposition"
                    )
                    new_group.attrs["Description"] = (
                        f'Group({slt}) containing decomposition in "z"'
                        " magnetic momentum basis of SOC matrix calculated"
                        f" from group: {group}."
                    )
                    decomposition_dataset = new_group.create_dataset(
                        f"{slt}_magnetic_momenta_matrix",
                        shape=decomposition.shape,
                        dtype=np.float64,
                    )
                    decomposition_dataset.attrs["Description"] = (
                        "Dataset containing % decomposition (rows -"
                        ' SO-states, columns - basis) in "z" magnetic'
                        f" momentum basis of SOC matrix from group: {group}."
                    )
                    states_dataset = new_group.create_dataset(
                        f"{slt}_pseudo_spin_states",
                        shape=(decomposition.shape[0],),
                        dtype=np.float64,
                    )
                    states_dataset.attrs["Description"] = (
                        "Dataset containing Sz pseudo-spin states"
                        " corresponding to the decomposition of SOC matrix"
                        f" from group: {group}."
                    )

                    decomposition_dataset[:] = decomposition[:]
                    dim = (decomposition.shape[1] - 1) / 2
                    states_dataset[:] = np.arange(
                        -dim, dim + 1, step=1, dtype=np.float64
                    )

                self._get_hdf5_groups_datasets_and_attributes()

            except Exception as e:
                error_type = type(e).__name__
                error_message = str(e)
                raise Exception(
                    "Error encountered while trying to save decomposition in"
                    ' "z" magnetic momentum basis of SOC matrix to file:'
                    f" {self._hdf5} - group {slt}: {error_type}:"
                    f" {error_message}"
                )

        return decomposition

    def decomposition_in_z_total_angular_momentum_basis(
        self, group, start_state, stop_state, rotation=None, slt: str = None
    ):
        if (not isinstance(stop_state, int)) or (stop_state < 0):
            raise ValueError(
                "Invalid states number, set it to positive integer or 0 for"
                " all states."
            )

        try:
            decomposition = (
                get_decomposition_in_z_total_angular_momentum_basis(
                    self._hdf5, group, start_state, stop_state, rotation
                )
            )
        except Exception as e:
            error_type = type(e).__name__
            error_message = str(e)
            raise Exception(
                'Error encountered while trying to get decomposition in "z"'
                " total angular momentum basis of SOC matrix from file:"
                f" {self._hdf5} - group {group}: {error_type}: {error_message}"
            )

        if slt is not None:
            try:
                with h5py.File(self._hdf5, "r+") as file:
                    new_group = file.create_group(
                        f"{slt}_total_angular_decomposition"
                    )
                    new_group.attrs["Description"] = (
                        f'Group({slt}) containing decomposition in "z" total'
                        " angular momentum basis of SOC matrix calculated"
                        f" from group: {group}."
                    )
                    decomposition_dataset = new_group.create_dataset(
                        f"{slt}_magnetic_momenta_matrix",
                        shape=decomposition.shape,
                        dtype=np.float64,
                    )
                    decomposition_dataset.attrs["Description"] = (
                        "Dataset containing % decomposition (rows SO-states,"
                        ' columns - basis) in "z" total angular momentum'
                        f" basis of SOC matrix from group: {group}."
                    )
                    states_dataset = new_group.create_dataset(
                        f"{slt}_pseudo_spin_states",
                        shape=(decomposition.shape[0],),
                        dtype=np.float64,
                    )
                    states_dataset.attrs["Description"] = (
                        "Dataset containing Sz pseudo-spin states"
                        " corresponding to the decomposition of SOC matrix"
                        f" from group: {group}."
                    )

                    decomposition_dataset[:] = decomposition[:]
                    dim = (decomposition.shape[1] - 1) / 2
                    states_dataset[:] = np.arange(
                        -dim, dim + 1, step=1, dtype=np.float64
                    )

                self._get_hdf5_groups_datasets_and_attributes()

            except Exception as e:
                error_type = type(e).__name__
                error_message = str(e)
                raise Exception(
                    "Error encountered while trying to save decomposition in"
                    ' "z" total angular momentum basis of SOC matrix to file:'
                    f" {self._hdf5} - group {slt}: {error_type}:"
                    f" {error_message}"
                )

        return decomposition

    def soc_crystal_field_parameters(
        self,
        group,
        start_state,
        stop_state,
        order,
        even_order: bool = True,
        complex: bool = False,
        magnetic: bool = False,
        rotation=None,
        slt: str = None,
    ):
        if magnetic:
            try:
                soc_matrix = get_soc_matrix_in_z_magnetic_momentum_basis(
                    self._hdf5, group, start_state, stop_state, rotation
                )
            except Exception as e:
                error_type = type(e).__name__
                error_message = str(e)
                raise Exception(
                    'Error encountered while trying to get SOC matrix in "z"'
                    f" magnetic momentum basis from file: {self._hdf5} - group"
                    f" {group}: {error_type}: {error_message}"
                )

        else:
            try:
                soc_matrix = get_soc_matrix_in_z_total_angular_momentum_basis(
                    self._hdf5, group, start_state, stop_state, rotation
                )
            except Exception as e:
                error_type = type(e).__name__
                error_message = str(e)
                raise Exception(
                    'Error encountered while trying to get SOC matrix in "z"'
                    f" total angular momentum basis from file: {self._hdf5} -"
                    f" group {group}: {error_type}: {error_message}"
                )

        dim = (soc_matrix.shape[1] - 1) / 2

        if order > 2 * dim:
            raise ValueError(
                "Order of ITO parameters exeeds 2S. Set it less or equal."
            )

        if complex:
            try:
                cfp = ito_complex_decomp_matrix(soc_matrix, order, even_order)
            except Exception as e:
                error_type = type(e).__name__
                error_message = str(e)
                raise Exception(
                    "Error encountered while trying to ITO decompose SOC"
                    ' matrix in "z" magnetic momentum basis from file:'
                    f" {self._hdf5} - group {group}: {error_type}:"
                    f" {error_message}"
                )
        else:
            try:
                cfp = ito_real_decomp_matrix(soc_matrix, order, even_order)
            except Exception as e:
                error_type = type(e).__name__
                error_message = str(e)
                raise Exception(
                    "Error encountered while trying to ITO decompose SOC"
                    ' matrix in "z" total angular momentum basis from file:'
                    f" {self._hdf5} - group {group}: {error_type}:"
                    f" {error_message}"
                )

        cfp_return = cfp

        if slt is not None:
            cfp = np.array(cfp)

            try:
                with h5py.File(self._hdf5, "r+") as file:
                    new_group = file.create_group(
                        f"{slt}_soc_ito_decomposition"
                    )
                    new_group.attrs["Description"] = (
                        f'Group({slt}) containing ITO decomposition in "z"'
                        " pseudo-spin basis of SOC matrix calculated from"
                        f" group: {group}."
                    )
                    cfp_dataset = new_group.create_dataset(
                        f"{slt}_ito_parameters",
                        shape=cfp.shape,
                        dtype=cfp.dtype,
                    )
                    cfp_dataset.attrs["Description"] = (
                        'Dataset containing ITO decomposition in "z"'
                        " pseudo-spin basis of SOC matrix from group:"
                        f" {group}."
                    )
                    states_dataset = new_group.create_dataset(
                        f"{slt}_pseudo_spin_states",
                        shape=(1,),
                        dtype=np.float64,
                    )
                    states_dataset.attrs["Description"] = (
                        "Dataset containing S pseudo-spin number"
                        " corresponding to the decomposition of SOC matrix"
                        f" from group: {group}."
                    )

                    cfp_dataset[:] = cfp[:]
                    states_dataset[:] = dim

                self._get_hdf5_groups_datasets_and_attributes()

            except Exception as e:
                error_type = type(e).__name__
                error_message = str(e)
                raise Exception(
                    "Error encountered while trying to save ITO decomposition"
                    ' in "z" pseudo-spin basis of SOC matrix to file:'
                    f" {self._hdf5} - group {slt}: {error_type}:"
                    f" {error_message}"
                )

        return cfp_return

    def zeeman_matrix_ito_decpomosition(
        self,
        group,
        start_state,
        stop_state,
        field,
        orientation,
        order,
        imaginary: bool = False,
        magnetic: bool = False,
        rotation=None,
        slt: str = None,
    ):
        if magnetic:
            try:
                zeeman_matrix = get_zeeman_matrix_in_z_magnetic_momentum_basis(
                    self._hdf5,
                    group,
                    field,
                    orientation,
                    start_state,
                    stop_state,
                    rotation,
                )
            except Exception as e:
                error_type = type(e).__name__
                error_message = str(e)
                raise Exception(
                    "Error encountered while trying to get Zeeman matrix in"
                    f' "z" magnetic momentum basis from file: {self._hdf5} -'
                    f" group {group}: {error_type}: {error_message}"
                )
        else:
            try:
                zeeman_matrix = (
                    get_zeeman_matrix_in_z_total_angular_momentum_basis(
                        self._hdf5,
                        group,
                        field,
                        orientation,
                        start_state,
                        stop_state,
                        rotation,
                    )
                )
            except Exception as e:
                error_type = type(e).__name__
                error_message = str(e)
                raise Exception(
                    "Error encountered while trying to get Zeeman matrix in"
                    ' "z" total angular momentum basis from file:'
                    f" {self._hdf5} - group {group}: {error_type}:"
                    f" {error_message}"
                )

        dim = (zeeman_matrix.shape[1] - 1) / 2

        if order > 2 * dim:
            raise ValueError(
                "Order of ITO parameters exeeds 2S. Set it less or equal."
            )

        if imaginary:
            try:
                cfp = ito_complex_decomp_matrix(zeeman_matrix, order)
            except Exception as e:
                error_type = type(e).__name__
                error_message = str(e)
                raise Exception(
                    "Error encountered while trying to ITO decompose Zeeman"
                    ' matrix in "z" magnetic momentum basis from file:'
                    f" {self._hdf5} - group {group}: {error_type}:"
                    f" {error_message}"
                )
        else:
            try:
                cfp = ito_real_decomp_matrix(zeeman_matrix, order)
            except Exception as e:
                error_type = type(e).__name__
                error_message = str(e)
                raise Exception(
                    "Error encountered while trying to ITO decompose Zeeman"
                    ' matrix in "z" total angular momentum basis from file:'
                    f" {self._hdf5} - group {group}: {error_type}:"
                    f" {error_message}"
                )

        cfp_return = cfp

        if slt is not None:
            cfp = np.array(cfp)

            try:
                with h5py.File(self._hdf5, "r+") as file:
                    new_group = file.create_group(
                        f"{slt}_zeeman_ito_decomposition"
                    )
                    new_group.attrs["Description"] = (
                        f'Group({slt}) containing ITO decomposition in "z"'
                        " pseudo-spin basis of Zeeman matrix calculated from"
                        f" group: {group}."
                    )
                    cfp_dataset = new_group.create_dataset(
                        f"{slt}_ito_parameters",
                        shape=cfp.shape,
                        dtype=cfp.dtype,
                    )
                    cfp_dataset.attrs["Description"] = (
                        'Dataset containing ITO decomposition in "z"'
                        " pseudo-spin basis of Zeeman matrix from group:"
                        f" {group}."
                    )
                    states_dataset = new_group.create_dataset(
                        f"{slt}_pseudo_spin_states",
                        shape=(1,),
                        dtype=np.float64,
                    )
                    states_dataset.attrs["Description"] = (
                        "Dataset containing S pseudo-spin number"
                        " corresponding to the decomposition of Zeeman matrix"
                        f" from group: {group}."
                    )

                    cfp_dataset[:] = cfp[:]
                    states_dataset[:] = dim

                self._get_hdf5_groups_datasets_and_attributes()

            except Exception as e:
                error_type = type(e).__name__
                error_message = str(e)
                raise Exception(
                    "Error encountered while trying to save ITO decomposition"
                    ' in "z" pseudo-spin basis of Zeeman matrix to file:'
                    f" {self._hdf5} - group {slt}: {error_type}:"
                    f" {error_message}"
                )

        return cfp_return

    def zeeman_matrix(
        self, group: str, states_cutoff, field, orientation, slt: str = None
    ):
        if (not isinstance(states_cutoff, int)) or (states_cutoff < 0):
            raise ValueError(
                "Invalid states cutoff, set it to positive integer or 0 for"
                " all states."
            )

        try:
            zeeman_matrix_array = get_zeeman_matrix(
                self._hdf5, group, states_cutoff, field, orientation
            )
        except Exception as e:
            error_type = type(e).__name__
            error_message = str(e)
            raise Exception(
                "Error encountered while trying to get Zeeman matrix from"
                f" file: {self._hdf5} - group {group}: {error_type}:"
                f" {error_message}"
            )

        if slt is not None:
            try:
                with h5py.File(self._hdf5, "r+") as file:
                    new_group = file.create_group(f"{slt}_zeeman_matrix")
                    new_group.attrs["Description"] = (
                        f"Group({slt}) containing Zeeman matrix calculated"
                        f" from group: {group}."
                    )
                    zeeman_matrix_dataset = new_group.create_dataset(
                        f"{slt}_zeeman_matrix",
                        shape=zeeman_matrix_array.shape,
                        dtype=np.complex128,
                    )
                    zeeman_matrix_dataset.attrs["Description"] = (
                        "Dataset containing Zeeman matrix calculated from"
                        f" group: {group}."
                    )
                    states_dataset = new_group.create_dataset(
                        f"{slt}_states",
                        shape=(zeeman_matrix_array.shape[1],),
                        dtype=np.int64,
                    )
                    states_dataset.attrs["Description"] = (
                        "Dataset containing states indexes of Zeeman matrix"
                        f" from group: {group}."
                    )

                    zeeman_matrix_dataset[:] = zeeman_matrix_array[:]
                    states_dataset[:] = np.arange(
                        zeeman_matrix_array.shape[1], dtype=np.int64
                    )

                self._get_hdf5_groups_datasets_and_attributes()

            except Exception as e:
                error_type = type(e).__name__
                error_message = str(e)
                raise Exception(
                    "Error encountered while trying to save Zeeman matrix to"
                    f" file: {self._hdf5} - group {slt}: {error_type}:"
                    f" {error_message}"
                )

        return zeeman_matrix_array

    def matrix_from_ito(
        self,
        name,
        imaginary: bool = False,
        dataset: str = None,
        pseudo_spin: str = None,
        slt: str = None,
        matrix_type: str = None,
    ):
        if (
            (dataset is not None)
            and (pseudo_spin is not None)
            and pseudo_spin > 0
        ):
            try:
                J = pseudo_spin
                coefficients = self[f"{name}", f"{dataset}"]
                if imaginary:
                    matrix = matrix_from_ito_complex(J, coefficients)
                else:
                    matrix = matrix_from_ito_real(J, coefficients)

            except Exception as e:
                error_type_1 = type(e).__name__
                error_message_1 = str(e)
                error_print_1 = f"{error_type_1}: {error_message_1}"
                raise Exception(
                    "Failed to form matrix from ITO parameters.\n Error(s)"
                    " encountered while trying compute the matrix:"
                    f" {error_print_1}"
                )

        else:
            try:
                J = self[
                    f"{name}_zeeman_ito_decomposition",
                    f"{name}_pseudo_spin_states",
                ]
                coefficients = self[
                    f"{name}_zeeman_ito_decomposition",
                    f"{name}_ito_parameters",
                ]

            except Exception as e:
                error_type_2 = type(e).__name__
                error_message_2 = str(e)
                error_print_2 = f"{error_type_2}: {error_message_2}"
                try:
                    J = self[
                        f"{name}_soc_ito_decomposition",
                        f"{name}_pseudo_spin_states",
                    ]
                    coefficients = self[
                        f"{name}_soc_ito_decomposition",
                        f"{name}_ito_parameters",
                    ]

                except Exception as e:
                    error_type_3 = type(e).__name__
                    error_message_3 = str(e)
                    error_print_3 = f"{error_type_3}: {error_message_3}"
                    raise Exception(
                        "Failed to form matrix from ITO parameters.\n Error(s)"
                        " encountered while trying compute the matrix:"
                        f" {error_print_2}, {error_print_3}"
                    )

                else:
                    J_result = J[0]
                    if imaginary:
                        matrix = matrix_from_ito_complex(J[0], coefficients)
                    else:
                        matrix = matrix_from_ito_real(J[0], coefficients)

            else:
                J_result = J[0]
                if imaginary:
                    matrix = matrix_from_ito_complex(J[0], coefficients)
                else:
                    matrix = matrix_from_ito_real(J[0], coefficients)

        if slt is not None:
            try:
                with h5py.File(self._hdf5, "r+") as file:
                    new_group = file.create_group(f"{slt}_matrix")
                    new_group.attrs["Description"] = (
                        f"Group({slt}) containing matrix from ITO calculated"
                        f" from group: {name}."
                    )
                    matrix_dataset = new_group.create_dataset(
                        f"{slt}_matrix",
                        shape=matrix.shape,
                        dtype=np.complex128,
                    )
                    matrix_dataset.attrs["Description"] = (
                        "Dataset containing matrix from ITO calculated from"
                        f" group: {name}."
                    )
                    states_dataset = new_group.create_dataset(
                        f"{slt}_pseudo_spin_states",
                        shape=(1,),
                        dtype=np.float64,
                    )
                    states_dataset.attrs["Description"] = (
                        "Dataset containing S pseudo-spin number"
                        f" corresponding to the matrix from group: {name}."
                    )

                    matrix_dataset[:] = matrix[:]
                    states_dataset[:] = J_result

                self._get_hdf5_groups_datasets_and_attributes()

            except Exception as e:
                error_type = type(e).__name__
                error_message = str(e)
                raise Exception(
                    "Error encountered while trying to save matrix from ITO"
                    f" to file: {self._hdf5} - group {slt}: {error_type}:"
                    f" {error_message}"
                )

        return matrix

    def soc_zeem_in_angular_magnetic_momentum_basis(
        self,
        group,
        start_state,
        stop_state,
        matrix_type,
        basis_type,
        rotation=None,
        field=None,
        orientation=None,
        slt: str = None,
    ):
        if (matrix_type not in ["zeeman", "soc"]) or (
            basis_type not in ["angular", "magnetic"]
        ):
            raise ValueError(
                'Only valid matrix_type are "soc" or "zeeman" and basis_type'
                ' are "angular" or "magnetic"'
            )

        if matrix_type == "zeeman" and (
            (field is None) or (orientation is None)
        ):
            raise ValueError(
                "For Zeeman matrix provide filed value and orientation."
            )

        try:
            if matrix_type == "zeeman":
                if basis_type == "angular":
                    matrix = (
                        get_zeeman_matrix_in_z_total_angular_momentum_basis(
                            self._hdf5,
                            group,
                            field,
                            orientation,
                            start_state,
                            stop_state,
                            rotation,
                        )
                    )
                elif basis_type == "magnetic":
                    matrix = get_zeeman_matrix_in_z_magnetic_momentum_basis(
                        self._hdf5,
                        group,
                        field,
                        orientation,
                        start_state,
                        stop_state,
                        rotation,
                    )
            elif matrix_type == "soc":
                if basis_type == "angular":
                    matrix = get_soc_matrix_in_z_total_angular_momentum_basis(
                        self._hdf5, group, start_state, stop_state, rotation
                    )
                elif basis_type == "magnetic":
                    matrix = get_soc_matrix_in_z_magnetic_momentum_basis(
                        self._hdf5, group, start_state, stop_state, rotation
                    )
        except Exception as e:
            error_type = type(e).__name__
            error_message = str(e)
            raise Exception(
                f"Error encountered while trying to get {matrix_type} matrix"
                f" from file in {basis_type} momentum basis: {self._hdf5} -"
                f" group {group}: {error_type}: {error_message}"
            )

        if slt is not None:
            try:
                with h5py.File(self._hdf5, "r+") as file:
                    new_group = file.create_group(
                        f"{slt}_{matrix_type}_matrix_in_{basis_type}_basis"
                    )
                    new_group.attrs["Description"] = (
                        f"Group({slt}) containing {matrix_type} matrix in"
                        f' {basis_type} momentum "z" basis calculated from'
                        f" group: {group}."
                    )
                    matrix_dataset = new_group.create_dataset(
                        f"{slt}_matrix",
                        shape=matrix.shape,
                        dtype=np.complex128,
                    )
                    matrix_dataset.attrs["Description"] = (
                        f"Dataset containing {matrix_type} matrix in"
                        f' {basis_type} momentum "z" basis calculated from'
                        f" group: {group}."
                    )
                    states_dataset = new_group.create_dataset(
                        f"{slt}_states",
                        shape=(matrix.shape[1],),
                        dtype=np.int64,
                    )
                    states_dataset.attrs["Description"] = (
                        "Dataset containing states indexes of"
                        f' {matrix_type} matrix in {basis_type} momentum "z"'
                        f" basis from group: {group}."
                    )

                    matrix_dataset[:] = matrix[:]
                    states_dataset[:] = np.arange(
                        matrix.shape[1], dtype=np.int64
                    )

                self._get_hdf5_groups_datasets_and_attributes()

            except Exception as e:
                error_type = type(e).__name__
                error_message = str(e)
                raise Exception(
                    "Error encountered while trying to save"
                    f' {matrix_type} matrix in {basis_type} momentum "z" basis'
                    f" to file: {self._hdf5} - group {slt}: {error_type}:"
                    f" {error_message}"
                )

        return matrix

    def calculate_chit_3d(
        self,
        group: str,
        fields: np.ndarray,
        states_cutoff: int,
        temperatures: np.ndarray,
        num_cpu: int,
        num_threads: int,
        num_of_points: int,
        delta_h: np.float64,
        spherical_grid: int,
        exp: bool = False,
        T: bool = True,
        slt: str = None,
    ):
        temperatures = np.array(temperatures, dtype=np.float64)
        fields = np.array(fields, dtype=np.float64)

        try:
            x, y, z = chit_3d(
                self._hdf5,
                group,
                fields,
                states_cutoff,
                temperatures,
                num_cpu,
                num_threads,
                num_of_points,
                delta_h,
                spherical_grid,
                exp,
                T,
            )
        except Exception as e:
            error_type = type(e).__name__
            error_message = str(e)
            raise Exception(
                "Error encountered while trying to compute 3D magnetic"
                f" susceptibility from file: {self._hdf5} - group {group}:"
                f" {error_type}: {error_message}"
            )

        if slt is not None:
            if T:
                chi_file = "chit"
            else:
                chi_file = "chi"

            try:
                with h5py.File(self._hdf5, "r+") as file:
                    new_group = file.create_group(f"{slt}_3d_susceptibility")
                    new_group.attrs["Description"] = (
                        f"Group({slt}) containing 3D magnetic susceptibility"
                        f" calculated from group: {group}."
                    )
                    chit_3d_dataset = new_group.create_dataset(
                        f"{slt}_{chi_file}_3d",
                        shape=(
                            3,
                            x.shape[0],
                            x.shape[1],
                            x.shape[2],
                            x.shape[3],
                        ),
                        dtype=np.float64,
                    )
                    chit_3d_dataset.attrs["Description"] = (
                        "Dataset containing 3D magnetic susceptibility as"
                        " meshgird (0-x,1-y,2-z) arrays over sphere ((xyz,"
                        " field, temperature, meshgrid, meshgrid) calculated"
                        f" from group: {group}."
                    )
                    fields_dataset = new_group.create_dataset(
                        f"{slt}_fields",
                        shape=(fields.shape[0],),
                        dtype=np.float64,
                    )
                    fields_dataset.attrs["Description"] = (
                        "Dataset containing magnetic field H values used in"
                        " simulation of 3D magnetic susceptibility from"
                        f" group: {group}."
                    )
                    temperatures_dataset = new_group.create_dataset(
                        f"{slt}_temperatures",
                        shape=(temperatures.shape[0],),
                        dtype=np.float64,
                    )
                    temperatures_dataset.attrs["Description"] = (
                        "Dataset containing temperature T values used in"
                        " simulation of 3D magnetic susceptibility from"
                        f" group: {group}."
                    )

                    chit_3d_dataset[0, :, :, :, :] = x[:, :, :, :]
                    chit_3d_dataset[1, :, :, :, :] = y[:, :, :, :]
                    chit_3d_dataset[2, :, :, :, :] = z[:, :, :, :]

                    temperatures_dataset[:] = temperatures
                    fields_dataset[:] = fields

                self._get_hdf5_groups_datasets_and_attributes()

            except Exception as e:
                error_type = type(e).__name__
                error_message = str(e)
                raise Exception(
                    "Error encountered while trying to save 3D magnetic"
                    f" susceptibility to file: {self._hdf5} - group {slt}:"
                    f" {error_type}: {error_message}"
                )

        return x, y, z

    def calculate_hemholtz_energy_3d(
        self,
        group: str,
        states_cutoff: int,
        fields: np.ndarray,
        spherical_grid: int,
        temperatures: np.ndarray,
        num_cpu: int,
        num_threads: int,
        internal_energy: bool = False,
        slt: str = None,
    ):
        temperatures = np.array(temperatures, dtype=np.float64)
        fields = np.array(fields, dtype=np.float64)

        try:
            x, y, z = hemholtz_energy_3d(
                self._hdf5,
                group,
                states_cutoff,
                fields,
                spherical_grid,
                temperatures,
                num_cpu,
                num_threads,
                internal_energy,
            )
        except Exception as e:
            error_type = type(e).__name__
            error_message = str(e)
            raise Exception(
                "Error encountered while trying to compute 3D magnetisation"
                f" from file: {self._hdf5} - group {group}: {error_type}:"
                f" {error_message}"
            )

        if slt is not None:
            try:
                with h5py.File(self._hdf5, "r+") as file:
                    new_group = file.create_group(f"{slt}_3d_hemholtz_energy")
                    new_group.attrs["Description"] = (
                        f"Group({slt}) containing 3D hemholtz_energy"
                        f" calculated from group: {group}."
                    )
                    hemholtz_energy_3d_dataset = new_group.create_dataset(
                        f"{slt}_energy_3d",
                        shape=(
                            3,
                            x.shape[0],
                            x.shape[1],
                            x.shape[2],
                            x.shape[3],
                        ),
                        dtype=np.float64,
                    )
                    hemholtz_energy_3d_dataset.attrs["Description"] = (
                        "Dataset containing 3D hemholtz_energy as meshgird"
                        " (0-x,1-y,2-z) arrays over sphere (xyz, field,"
                        " temperature, meshgrid, meshgrid) calculated from"
                        f" group: {group}."
                    )
                    fields_dataset = new_group.create_dataset(
                        f"{slt}_fields",
                        shape=(fields.shape[0],),
                        dtype=np.float64,
                    )
                    fields_dataset.attrs["Description"] = (
                        "Dataset containing magnetic field H values used in"
                        " simulation of 3D hemholtz_energy from group:"
                        f" {group}."
                    )
                    temperatures_dataset = new_group.create_dataset(
                        f"{slt}_temperatures",
                        shape=(temperatures.shape[0],),
                        dtype=np.float64,
                    )
                    temperatures_dataset.attrs["Description"] = (
                        "Dataset containing temperature T values used in"
                        " simulation of 3D hemholtz_energy from group:"
                        f" {group}."
                    )

                    hemholtz_energy_3d_dataset[0, :, :, :, :] = x[:, :, :, :]
                    hemholtz_energy_3d_dataset[1, :, :, :, :] = y[:, :, :, :]
                    hemholtz_energy_3d_dataset[2, :, :, :, :] = z[:, :, :, :]
                    temperatures_dataset[:] = temperatures
                    fields_dataset[:] = fields

                self._get_hdf5_groups_datasets_and_attributes()

            except Exception as e:
                error_type = type(e).__name__
                error_message = str(e)
                raise Exception(
                    "Error encountered while trying to save 3D"
                    f" hemholtz_energy to file: {self._hdf5} - group {slt}:"
                    f" {error_type}: {error_message}"
                )

        return x, y, z

    def calculate_hemholtz_energyth(
        self,
        group: str,
        states_cutoff: np.int64,
        fields: np.ndarray,
        grid: np.ndarray,
        temperatures: np.ndarray,
        num_cpu: int,
        num_threads: int,
        internal_energy: bool = False,
        slt: str = None,
    ):
        fields = np.array(fields)
        temperatures = np.array(temperatures)

        if isinstance(grid, int):
            grid = lebedev_laikov_grid(grid)
        else:
            grid = np.array(grid)

        try:
            hemholtz_energyth_array = hemholtz_energyth(
                self._hdf5,
                group,
                states_cutoff,
                fields,
                grid,
                temperatures,
                num_cpu,
                num_threads,
                internal_energy,
            )
        except Exception as e:
            error_type = type(e).__name__
            error_message = str(e)
            raise Exception(
                "Error encountered while trying to compute E(T,H) from file:"
                f" {self._hdf5} - group {group}: {error_type}: {error_message}"
            )

        if slt is not None:
            try:
                with h5py.File(self._hdf5, "r+") as file:
                    new_group = file.create_group(f"{slt}_hemholtz_energy")
                    new_group.attrs["Description"] = (
                        f"Group({slt}) containing E(T,H) Hemholtz energy"
                        f" calculated from group: {group}."
                    )
                    hemholtz_energyth_dataset = new_group.create_dataset(
                        f"{slt}_eth",
                        shape=(
                            hemholtz_energyth_array.shape[0],
                            hemholtz_energyth_array.shape[1],
                        ),
                        dtype=np.float64,
                    )
                    hemholtz_energyth_dataset.attrs["Description"] = (
                        "Dataset containing E(T,H) Hemholtz energy (T - rows,"
                        f" H - columns) calculated from group: {group}."
                    )
                    fields_dataset = new_group.create_dataset(
                        f"{slt}_fields",
                        shape=(fields.shape[0],),
                        dtype=np.float64,
                    )
                    fields_dataset.attrs["Description"] = (
                        "Dataset containing magnetic field H values used in"
                        f" simulation of E(T,H) from group: {group}."
                    )
                    temperatures_dataset = new_group.create_dataset(
                        f"{slt}_temperatures",
                        shape=(temperatures.shape[0],),
                        dtype=np.float64,
                    )
                    temperatures_dataset.attrs["Description"] = (
                        "Dataset containing temperature T values used in"
                        f" simulation of E(T,H) from group: {group}."
                    )

                    hemholtz_energyth_dataset[:, :] = hemholtz_energyth_array[
                        :, :
                    ]
                    fields_dataset[:] = fields[:]
                    temperatures_dataset[:] = temperatures[:]

                self._get_hdf5_groups_datasets_and_attributes()

            except Exception as e:
                error_type = type(e).__name__
                error_message = str(e)
                raise Exception(
                    "Error encountered while trying to save E(T,H) to file:"
                    f" {self._hdf5} - group {slt}: {error_type}:"
                    f" {error_message}"
                )

        return hemholtz_energyth_array

    ####Experimental plotting
    @staticmethod
    def colour_map(name):
        """
        Creates matplotlib colour map object.

        Args:
            name: (str or lst) one of defined names for colour maps: BuPi, rainbow, dark_rainbow, light_rainbow,
            light_rainbow_alt, BuOr, BuYl, BuRd, GnYl, PrOr, GnRd, funmat, NdCoN322bpdo, NdCoNO222bpdo, NdCoI22bpdo,
            viridis, plasma, inferno, magma, cividis or list of colour from which colour map will be created by
            interpolation of colours between ones on a list; for predefined names modifiers can be applyed: _l loops
            the list in a way that it starts and ends with the same colour, _r reverses the list

        Returns:

        """
        cmap_list = []
        reverse = False
        loop = False
        if name[-2:] == "_l":
            name = name[:-2]
            loop = True
        try:
            if name[-2:] == "_r":
                reverse = True
                name = name[:-2]
            if type(name) == list:
                cmap_list = name
            elif name == "BuPi":
                cmap_list = [
                    "#0091ad",
                    "#1780a1",
                    "#2e6f95",
                    "#455e89",
                    "#5c4d7d",
                    "#723c70",
                    "#a01a58",
                    "#b7094c",
                ]
            elif name == "rainbow":
                cmap_list = [
                    "#ff0000",
                    "#ff8700",
                    "#ffd300",
                    "#deff0a",
                    "#a1ff0a",
                    "#0aff99",
                    "#0aefff",
                    "#147df5",
                    "#580aff",
                    "#be0aff",
                ]
            elif name == "dark_rainbow":
                cmap_list = [
                    "#F94144",
                    "#F3722C",
                    "#F8961E",
                    "#F9844A",
                    "#F9C74F",
                    "#90BE6D",
                    "#43AA8B",
                    "#4D908E",
                    "#577590",
                    "#277DA1",
                ]
            elif name == "light_rainbow":
                cmap_list = [
                    "#FFADAD",
                    "#FFD6A5",
                    "#FDFFB6",
                    "#CAFFBF",
                    "#9BF6FF",
                    "#A0C4FF",
                    "#BDB2FF",
                    "#FFC6FF",
                ]
            elif name == "light_rainbow_alt":
                cmap_list = [
                    "#FBF8CC",
                    "#FDE4CF",
                    "#FFCFD2",
                    "#F1C0E8",
                    "#CFBAF0",
                    "#A3C4F3",
                    "#90DBF4",
                    "#8EECF5",
                    "#98F5E1",
                    "#B9FBC0",
                ]
            elif name == "BuOr":
                cmap_list = [
                    "#03045e",
                    "#023e8a",
                    "#0077b6",
                    "#0096c7",
                    "#00b4d8",
                    "#ff9e00",
                    "#ff9100",
                    "#ff8500",
                    "#ff6d00",
                    "#ff5400",
                ]
            elif name == "BuRd":
                cmap_list = [
                    "#033270",
                    "#1368aa",
                    "#4091c9",
                    "#9dcee2",
                    "#fedfd4",
                    "#f29479",
                    "#ef3c2d",
                    "#cb1b16",
                    "#65010c",
                ]
            elif name == "BuYl":
                cmap_list = [
                    "#184e77",
                    "#1e6091",
                    "#1a759f",
                    "#168aad",
                    "#34a0a4",
                    "#52b69a",
                    "#76c893",
                    "#99d98c",
                    "#b5e48c",
                    "#d9ed92",
                ]
            elif name == "GnYl":
                cmap_list = [
                    "#007f5f",
                    "#2b9348",
                    "#55a630",
                    "#80b918",
                    "#aacc00",
                    "#bfd200",
                    "#d4d700",
                    "#dddf00",
                    "#eeef20",
                    "#ffff3f",
                ]
            elif name == "PrOr":
                cmap_list = [
                    "#240046",
                    "#3c096c",
                    "#5a189a",
                    "#7b2cbf",
                    "#9d4edd",
                    "#ff9e00",
                    "#ff9100",
                    "#ff8500",
                    "#ff7900",
                    "#ff6d00",
                ]
            elif name == "GnRd":
                cmap_list = [
                    "#005C00",
                    "#2D661B",
                    "#2A850E",
                    "#27A300",
                    "#A9FFA5",
                    "#FFA5A5",
                    "#FF0000",
                    "#BA0C0C",
                    "#751717",
                    "#5C0000",
                ]
            elif name == "funmat":
                cmap_list = [
                    "#1f6284",
                    "#277ba5",
                    "#2f94c6",
                    "#49a6d4",
                    "#6ab6dc",
                    "#ffe570",
                    "#ffe15c",
                    "#ffda33",
                    "#ffd20a",
                    "#e0b700",
                ]
            elif name == "NdCoN322bpdo":
                cmap_list = [
                    "#00268f",
                    "#0046ff",
                    "#009cf4",
                    "#E5E4E2",
                    "#ede76d",
                    "#ffb900",
                    "#b88700",
                ]
            elif name == "NdCoNO222bpdo":
                cmap_list = [
                    "#A90F97",
                    "#E114C9",
                    "#f9bbf2",
                    "#77f285",
                    "#11BB25",
                    "#0C831A",
                ]
            elif name == "NdCoI22bpdo":
                cmap_list = [
                    "#075F5F",
                    "#0B9898",
                    "#0fd1d1",
                    "#FAB3B3",
                    "#d10f0f",
                    "#720808",
                ]
            if cmap_list:
                if reverse:
                    cmap_list.reverse()
                if loop:
                    new_cmap_list = cmap_list.copy()
                    for i in range(len(cmap_list)):
                        new_cmap_list.append(cmap_list[-(i + 1)])
                    cmap_list = new_cmap_list
                cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
                    "", cmap_list
                )
            elif name == "viridis":
                cmap = matplotlib.cm.viridis
                if reverse:
                    cmap = matplotlib.cm.viridis_r
            elif name == "plasma":
                cmap = matplotlib.cm.plasma
                if reverse:
                    cmap = matplotlib.cm.plasma_r
            elif name == "inferno":
                cmap = matplotlib.cm.inferno
                if reverse:
                    cmap = matplotlib.cm.inferno_r
            elif name == "magma":
                cmap = matplotlib.cm.magma
                if reverse:
                    cmap = matplotlib.cm.magma_r
            elif name == "cividis":
                cmap = matplotlib.cm.cividis
                if reverse:
                    cmap = matplotlib.cm.cividis_r
            else:
                print(
                    f"""There is no such colour map as {name} use one of those: BuPi, rainbow, dark_rainbow, light_rainbow, 
            light_rainbow_alt, BuOr, BuYl, BuRd, GnYl, PrOr, GnRd, funmat, NdCoN322bpdo, NdCoNO222bpdo, NdCoI22bpdo,
            viridis, plasma, inferno, magma, cividis or enter list of colours"""
                )
            return cmap
        except Exception as e:
            error_type = type(e).__name__
            error_message = str(e)
            raise Exception(
                "Error encountered while trying to find palette/colour map:"
                f" {error_type}: {error_message}"
            )

    @staticmethod
    def custom_colour_cycler(number_of_colours: int, cmap1: str, cmap2: str):
        """
        Creates colour cycler from two colour maps in alternating pattern, suitable for use in matplotlib plots.

        Args:
            number_of_colours: (int) number of colour in cycle
            cmap1: (str or lst) colour map name or list of colours (valid input for Compoud.colour_map())
            cmap2: (str or lst) colour map name or list of colours (valid input for Compoud.colour_map())

        Returns:
            cycler object created based on two input colourmaps
        """
        if number_of_colours % 2 == 0:
            increment = 0
            lst1 = Compound.colour_map(cmap1)(
                np.linspace(0, 1, int(number_of_colours / 2))
            )
            lst2 = Compound.colour_map(cmap2)(
                np.linspace(0, 1, int(number_of_colours / 2))
            )
            colour_cycler_list = []
            while increment < number_of_colours:
                if increment % 2 == 0:
                    colour_cycler_list.append(lst1[int(increment / 2)])
                else:
                    colour_cycler_list.append(lst2[int((increment - 1) / 2)])
                increment += 1
        else:
            increment = 0
            lst1 = Compound.colour_map(cmap1)(
                np.linspace(0, 1, int((number_of_colours / 2) + 1))
            )
            lst2 = Compound.colour_map(cmap2)(
                np.linspace(0, 1, int(number_of_colours / 2))
            )
            colour_cycler_list = []
            while increment < number_of_colours:
                if increment % 2 == 0:
                    colour_cycler_list.append(lst1[int(increment / 2)])
                else:
                    colour_cycler_list.append(lst2[int((increment - 1) / 2)])
                increment += 1
        return cycler(color=colour_cycler_list)

    def plot_mth(
        self,
        group: str,
        show=True,
        origin=False,
        save=False,
        colour_map_name="rainbow",
        xlim=(),
        ylim=(),
        xticks=1,
        yticks=0,
        field="B",
    ):
        """
        Function that creates graphs of M(H,T) given name of the group in HDF5 file, graphs can be optionally shown,
        saved, colour palettes can be changed. If origin=True it returns data packed into a dictionary for exporting
        to Origin.

            Args:
                group (str): name of a group in HDF5 file
                show (bool): determines if matplotlib graph is created
                and shown if True
                origin (bool): determines if function should return raw data
                save (bool): determines if matplotlib graph should be saved, saved graphs are TIFF files
                colour_map_name (str) or (list): sets colours used to create graphs, valid options are returned by
                Compound.colour_map staticmethod
                xlim (tuple): tuple of two or one numbers that set corresponding axe limits
                ylim (tuple): tuple of two or one numbers that set corresponding axe limits
                xticks (int): frequency of x major ticks
                yticks (int): frequency of x major ticks
                field ('B' or 'H'): chooses field type and unit: Tesla for B and kOe for H
            Returns:
                if origin=True:
                    dict[origin_column (str), data (np.array)]: contains data used to create graph in origin
        """
        try:
            """Getting data from hdf5 or sloth file"""
            mth = self[f"{group}_magnetisation", f"{group}_mth"]
            fields = self[f"{group}_magnetisation", f"{group}_fields"]
            if field == "H":
                fields *= 10
                xticks *= 10
            temps = self[f"{group}_magnetisation", f"{group}_temperatures"]
            """Creates dataset suitable to be exported to Origin"""
            data = {"data_x": fields, "data_y": mth, "comment": temps}
        except Exception as e:
            error_type = type(e).__name__
            error_message = str(e)
            raise Exception(
                "Error encountered while trying to get data to create graph"
                f" of M(H, T): {self._hdf5} - group {group}: {error_type}:"
                f" {error_message}"
            )
        if show:
            try:
                """Plotting in matplotlib"""
                fig, ax = plt.subplots()
                """Defining colour maps for graphs"""
                colour = iter(
                    Compound.colour_map(colour_map_name)(
                        np.linspace(0, 1, len(temps))
                    )
                )
                """Creating a plot"""
                for i, mh in enumerate(mth):
                    c = next(colour)
                    ax.plot(
                        fields, mh, linewidth=2, c=c, label=f"{temps[i]} K"
                    )

                if yticks:
                    ax.yaxis.set_major_locator(MultipleLocator(yticks))
                ax.xaxis.set_major_locator(MultipleLocator(xticks))
                ax.xaxis.set_minor_locator(AutoMinorLocator(2))
                ax.yaxis.set_minor_locator(AutoMinorLocator(2))
                ax.tick_params(which="major", length=7)
                ax.tick_params(which="minor", length=3.5)
                if field == "B":
                    ax.set_xlabel(r"$B\ /\ \mathrm{T}$")
                elif field == "H":
                    ax.set_xlabel(r"$H\ /\ \mathrm{kOe}$")
                ax.set_ylabel(r"$M\ /\ \mathrm{\mu_{B}}$")
                if xlim:
                    if len(xlim) == 2:
                        ax.set_ylim(xlim[0], xlim[1])
                    else:
                        ax.set_ylim(xlim[0])
                else:
                    if len(temps) > 17:
                        ax.set_xlim(0)
                        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

                    else:
                        ax.set_xlim(0, fields[-1] + 0.3 * fields[-1])
                        ax.legend()
                if ylim:
                    if len(ylim) == 2:
                        ax.set_ylim(ylim[0], ylim[1])
                    else:
                        ax.set_ylim(ylim[0])
                else:
                    ax.set_ylim(0)
                plt.tight_layout()
                plt.show()
            except Exception as e:
                error_type = type(e).__name__
                error_message = str(e)
                raise Exception(
                    "Error encountered while trying to create graph of M(H,"
                    f" T): {self._hdf5} - group {group}: {error_type}:"
                    f" {error_message}"
                )
            if save:
                try:
                    """Saving plot figure"""
                    fig.savefig(f"mgh_{group}.tiff", dpi=300)
                except Exception as e:
                    error_type = type(e).__name__
                    error_message = str(e)
                    raise Exception(
                        "Error encountered while trying to save graph of M(H,"
                        f" T): {self._hdf5} - group {group}: {error_type}:"
                        f" {error_message}"
                    )
        if origin:
            return data

    def plot_chitht(
        self,
        group,
        show=True,
        origin=False,
        save=False,
        colour_map_name="funmat",
        xlim=(),
        ylim=(),
        xticks=100,
        yticks=0,
        field="B",
    ):
        """
        Function that creates graphs of chiT(H,T) or chi(H,T) depending on content of HDF5 file, given name of the group
        in HDF5 file, graphs can be optionally shown, saved, colour palettes can be changed. If origin=True it returns
        data packed into a dictionary for exporting to Origin.

            Args:
                group (str): name of a group in HDF5 file
                show (bool): determines if matplotlib graph is created
                and shown if True
                origin (bool): determines if function should return raw data
                save (bool): determines if matplotlib graph should be saved, saved graphs are TIFF files
                colour_map_name (str): sets colours used to create graphs, valid options are returned by
                Compound.colour_map staticmethod
                xlim (tuple): tuple of two or one numbers that set corresponding axe limits
                ylim (tuple): tuple of two or one numbers that set corresponding axe limits
                xticks (int): frequency of x major ticks
                yticks (int): frequency of x major ticks
                field ('B' or 'H'): chooses field type and unit: Tesla for B and kOe for H
            Returns:
                if origin=True:
                    dict[origin_column (str), data (np.array)]: contains data used to create graph in origin
        """
        try:
            """Getting data from hdf5 or sloth file"""
            try:
                chi = self[f"{group}_susceptibility", f"{group}_chiht"]
                T = False
            except:
                chi = self[f"{group}_susceptibility", f"{group}_chitht"]
                T = True
            fields = self[f"{group}_susceptibility", f"{group}_fields"]
            if field == "H":
                fields *= 10
            temps = self[f"{group}_susceptibility", f"{group}_temperatures"]
            """Creates dataset suitable to be exported to Origin"""
            data = {"data_x": temps, "data_y": chi, "comment": fields, "T": T}
        except Exception as e:
            error_type = type(e).__name__
            error_message = str(e)
            raise Exception(
                "Error encountered while trying to get data to create graph"
                f" of chiT(H,T) or chi(H,T): {self._hdf5} - group {group}:"
                f" {error_type}: {error_message}"
            )
        if show:
            try:
                """Plotting in matplotlib"""
                fig, ax = plt.subplots()
                """Defining colour maps for graphs"""
                colour = iter(
                    Compound.colour_map(colour_map_name)(
                        np.linspace(0, 1, len(fields))
                    )
                )
                """Creating a plot"""
                for i, ch in enumerate(chi):
                    c = next(colour)
                    ax.plot(
                        temps,
                        ch,
                        linewidth=2,
                        c=c,
                        label=(
                            f'{round(fields[i], 2)} {"kOe" if field == "H" else "T"}'
                        ),
                    )
                ax.xaxis.set_major_locator(MultipleLocator(xticks))
                if yticks:
                    ax.yaxis.set_major_locator(MultipleLocator(yticks))
                ax.xaxis.set_minor_locator(AutoMinorLocator(2))
                ax.yaxis.set_minor_locator(AutoMinorLocator(2))
                ax.tick_params(which="major", length=7)
                ax.tick_params(which="minor", length=3.5)
                ax.set_xlabel(r"$T\ /\ \mathrm{K}$")
                if T:
                    ax.set_ylabel(
                        r"$\chi_{\mathrm{M}}T\ /\ \mathrm{cm^{3}mol^{-1}K}$"
                    )
                else:
                    ax.set_ylabel(
                        r"$\chi_{\mathrm{M}}\ /\ \mathrm{cm^{3}mol^{-1}}$"
                    )
                if xlim:
                    if len(xlim) == 2:
                        ax.set_ylim(xlim[0], xlim[1])
                    else:
                        ax.set_ylim(xlim[0])
                else:
                    ax.set_xlim(0, temps[-1])
                if ylim:
                    if len(ylim) == 2:
                        ax.set_ylim(ylim[0], ylim[1])
                    else:
                        ax.set_ylim(ylim[0])
                else:
                    ax.set_ylim(0)
                ax.legend()
                plt.tight_layout()
                plt.show()
            except Exception as e:
                error_type = type(e).__name__
                error_message = str(e)
                raise Exception(
                    "Error encountered while trying to create graph of"
                    f" chiT(H,T) or chi(H,T): {self._hdf5} - group {group}:"
                    f" {error_type}: {error_message}"
                )
            if save:
                try:
                    """Saving plot figure"""
                    fig.savefig(f"chitht_{group}.tiff", dpi=300)
                except Exception as e:
                    error_type = type(e).__name__
                    error_message = str(e)
                    raise Exception(
                        "Error encountered while trying to save graph of"
                        f" chiT(H,T) or chi(H,T): {self._hdf5} - group"
                        f" {group}: {error_type}: {error_message}"
                    )
        if origin:
            return data

    def plot_zeeman(
        self,
        group: str,
        show=True,
        origin=False,
        save=False,
        colour_map_name1="BuPi",
        colour_map_name2="BuPi_r",
        single=False,
        xlim=(),
        ylim=(),
        xticks=1,
        yticks=0,
        field="B",
    ):
        """
        Function that creates graphs of E(H,orientation) given name of the group in HDF5 file, graphs can be optionally shown,
        saved, colour palettes can be changed. If origin=True it returns data packed into a dictionary for exporting
        to Origin.

            Args:
                group (str): name of a group in HDF5 file
                show (bool): determines if matplotlib graph is created
                and shown if True
                origin (bool): determines if function should return raw data
                save (bool): determines if matplotlib graph should be saved, saved graphs are TIFF files
                colour_map_name1 (str) or (list): sets colours used to create graphs, valid options are returned by
                Compound.colour_map staticmethod
                colour_map_name2 (str) or (list): sets colours used to create graphs, valid options are returned by
                Compound.colour_map staticmethod
                single (bool): determines if graph should be created for each orientation given separately
                xlim (tuple): tuple of two or one numbers that set corresponding axe limits
                ylim (tuple): tuple of two or one numbers that set corresponding axe limits
                xticks (int): frequency of x major ticks
                yticks (int): frequency of y major ticks
                field ('B' or 'H'): chooses field type and unit: Tesla for B and kOe for H
            Returns:
                if origin=True:
                    dict[origin_column (str), data (np.array)]: contains data used to create graph in origin
        """
        try:
            """Getting data from hdf5 or sloth file"""
            zeeman = self[f"{group}_zeeman_splitting", f"{group}_zeeman"]
            fields = self[f"{group}_zeeman_splitting", f"{group}_fields"]
            if field == "H":
                fields *= 10
                xticks *= 10
            orientations = self[
                f"{group}_zeeman_splitting", f"{group}_orientations"
            ]
            """Creates dataset suitable to be exported to Origin"""
            for i, orientation in enumerate(orientations):
                data = {
                    f"data_x{i}": fields,
                    "data_y": mth,
                    f"comment{i}": orientation,
                }
        except Exception as e:
            error_type = type(e).__name__
            error_message = str(e)
            raise Exception(
                "Error encountered while trying to get data to create graph"
                f" of E(H,orientation): {self._hdf5} - group {group}:"
                f" {error_type}: {error_message}"
            )
        if show:
            try:
                """Plotting in matplotlib"""
                if not single:
                    number_of_plots = len(orientations)
                    if number_of_plots % 5 == 0:
                        fig = plt.figure(
                            figsize=(16, 3.2 * (number_of_plots / 5))
                        )
                        gs = matplotlib.gridspec.GridSpec(
                            int(number_of_plots / 5), 5
                        )
                        devisor = 5
                    elif number_of_plots % 3 == 0:
                        fig = plt.figure(
                            figsize=(9.6, 3.2 * (number_of_plots / 3))
                        )
                        gs = matplotlib.gridspec.GridSpec(
                            int(number_of_plots / 3), 3
                        )
                        devisor = 3
                    elif number_of_plots % 2 == 0:
                        fig = plt.figure(
                            figsize=(6.4, 3.2 * (number_of_plots / 2))
                        )
                        gs = matplotlib.gridspec.GridSpec(
                            int(number_of_plots / 2), 2
                        )
                        devisor = 2
                    else:
                        fig = plt.figure(figsize=(6.4, 3.2 * number_of_plots))
                        gs = matplotlib.gridspec.GridSpec(1, number_of_plots)
                        devisor = 1
                    """Creating a plot"""
                    for i, zee in enumerate(zeeman):
                        if i % devisor != 0:
                            plt.rc(
                                "axes",
                                prop_cycle=Compound.custom_colour_cycler(
                                    len(zeeman[0][0]),
                                    colour_map_name1,
                                    colour_map_name2,
                                ),
                            )
                            multiple_plots = fig.add_subplot(
                                gs[i // devisor, i % devisor]
                            )
                            plt.plot(fields, zee, linewidth=0.75)
                            multiple_plots.xaxis.set_major_locator(
                                MultipleLocator(xticks * 2)
                            )
                            if yticks:
                                multiple_plots.yaxis.set_major_locator(
                                    MultipleLocator(yticks)
                                )
                            multiple_plots.xaxis.set_minor_locator(
                                AutoMinorLocator(2)
                            )
                            multiple_plots.yaxis.set_minor_locator(
                                AutoMinorLocator(2)
                            )
                            multiple_plots.tick_params(
                                which="major",
                                left=False,
                                labelleft=False,
                                length=7,
                            )
                            multiple_plots.tick_params(
                                which="minor", left=False, length=3.5
                            )
                            plt.title(
                                "Orientation"
                                f" [{round(orientations[i][0], 3)} {round(orientations[i][1], 3)} {round(orientations[i][2], 3)}]"
                            )
                            if xlim:
                                if len(xlim) == 2:
                                    multiple_plots.set_xlim(xlim[0], xlim[1])
                                else:
                                    multiple_plots.set_xlim(xlim[0])
                            if ylim:
                                if len(ylim) == 2:
                                    multiple_plots.set_ylim(ylim[0], ylim[1])
                                else:
                                    multiple_plots.set_ylim(ylim[0])

                        else:
                            if (i // devisor) == 0:
                                plt.rc(
                                    "axes",
                                    prop_cycle=Compound.custom_colour_cycler(
                                        len(zeeman[0][0]),
                                        colour_map_name1,
                                        colour_map_name2,
                                    ),
                                )
                                multiple_plots = fig.add_subplot(
                                    gs[i // devisor, i % devisor]
                                )
                                plt.plot(fields, zee, linewidth=0.75)
                                multiple_plots.xaxis.set_major_locator(
                                    MultipleLocator(xticks * 2)
                                )
                                if yticks:
                                    multiple_plots.yaxis.set_major_locator(
                                        MultipleLocator(yticks)
                                    )
                                multiple_plots.xaxis.set_minor_locator(
                                    AutoMinorLocator(2)
                                )
                                multiple_plots.tick_params(
                                    which="major", length=7
                                )
                                multiple_plots.tick_params(
                                    which="minor", length=3.5
                                )
                                multiple_plots.yaxis.set_minor_locator(
                                    AutoMinorLocator(2)
                                )
                                plt.title(
                                    "Orientation"
                                    f" [{round(orientations[i][0], 3)} {round(orientations[i][1], 3)} {round(orientations[i][2], 3)}]"
                                )
                                if xlim:
                                    if len(xlim) == 2:
                                        multiple_plots.set_xlim(
                                            xlim[0], xlim[1]
                                        )
                                    else:
                                        multiple_plots.set_xlim(xlim[0])
                                if ylim:
                                    if len(ylim) == 2:
                                        multiple_plots.set_ylim(
                                            ylim[0], ylim[1]
                                        )
                                    else:
                                        multiple_plots.set_ylim(ylim[0])
                            else:
                                plt.rc(
                                    "axes",
                                    prop_cycle=Compound.custom_colour_cycler(
                                        len(zeeman[0][0]),
                                        colour_map_name1,
                                        colour_map_name2,
                                    ),
                                )
                                multiple_plots = fig.add_subplot(
                                    gs[i // devisor, i % devisor]
                                )
                                plt.plot(fields, zee, linewidth=0.75)
                                multiple_plots.xaxis.set_major_locator(
                                    MultipleLocator(xticks * 2)
                                )
                                if yticks:
                                    multiple_plots.yaxis.set_major_locator(
                                        MultipleLocator(yticks)
                                    )
                                multiple_plots.xaxis.set_minor_locator(
                                    AutoMinorLocator(2)
                                )
                                multiple_plots.tick_params(
                                    which="major", length=7
                                )
                                multiple_plots.tick_params(
                                    which="minor", length=3.5
                                )
                                multiple_plots.yaxis.set_minor_locator(
                                    AutoMinorLocator(2)
                                )
                                plt.title(
                                    "Orientation"
                                    f" [{round(orientations[i][0], 3)} {round(orientations[i][1], 3)} {round(orientations[i][2], 3)}]"
                                )
                                if xlim:
                                    if len(xlim) == 2:
                                        multiple_plots.set_xlim(
                                            xlim[0], xlim[1]
                                        )
                                    else:
                                        multiple_plots.set_xlim(xlim[0])
                                if ylim:
                                    if len(ylim) == 2:
                                        multiple_plots.set_ylim(
                                            ylim[0], ylim[1]
                                        )
                                    else:
                                        multiple_plots.set_ylim(ylim[0])
                    if field == "B":
                        fig.supxlabel(r"$B\ /\ \mathrm{T}$")
                    if field == "H":
                        fig.supxlabel(r"$H\ /\ \mathrm{kOe}$")
                    fig.supylabel(r"$\mathrm{Energy\ /\ cm^{-1}}$")
                    plt.tight_layout()
                    plt.show()
                elif single:
                    for i, zee in enumerate(zeeman):
                        plt.rc(
                            "axes",
                            prop_cycle=Compound.custom_colour_cycler(
                                len(zeeman[0][0]),
                                colour_map_name1,
                                colour_map_name2,
                            ),
                        )
                        fig, ax = plt.subplots()
                        ax.plot(fields, zee, linewidth=0.75)
                        plt.title(f"Orientation {orientations[i]}")
                        if field == "B":
                            ax.set_xlabel(r"$B\ /\ \mathrm{T}$")
                        elif field == "H":
                            ax.set_xlabel(r"$H\ /\ \mathrm{kOe}$")
                        ax.set_ylabel(r"$\mathrm{Energy\ /\ cm^{-1}}$")
                        ax.tick_params(which="major", length=7)
                        ax.tick_params(which="minor", length=3.5)
                        ax.xaxis.set_major_locator(MultipleLocator(xticks))
                        if yticks:
                            ax.yaxis.set_major_locator(MultipleLocator(yticks))
                        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
                        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
                        if xlim:
                            if len(xlim) == 2:
                                ax.set_xlim(xlim[0], xlim[1])
                            else:
                                ax.set_xlim(xlim[0])
                        if ylim:
                            if len(ylim) == 2:
                                ax.set_ylim(ylim[0], ylim[1])
                            else:
                                ax.set_ylim(ylim[0])
                        plt.tight_layout()
                        plt.show()
                        if save:
                            try:
                                """Saving plot figure"""
                                fig.savefig(
                                    (
                                        f"zeeman_{group}_Orientation"
                                        f" {orientations[i]}.tiff"
                                    ),
                                    dpi=300,
                                )
                            except Exception as e:
                                error_type = type(e).__name__
                                error_message = str(e)
                                raise Exception(
                                    "Error encountered while trying to save"
                                    " graph of E(H,orientation):"
                                    f" {self._hdf5} - group {group}:"
                                    f" {error_type}: {error_message}"
                                )

            except Exception as e:
                error_type = type(e).__name__
                error_message = str(e)
                raise Exception(
                    "Error encountered while trying to create graph of"
                    f" E(H,orientation): {self._hdf5} - group {group}:"
                    f" {error_type}: {error_message}"
                )
            if save and not single:
                try:
                    """Saving plot figure"""
                    fig.savefig(f"zeeman_{group}.tiff", dpi=300)
                except Exception as e:
                    error_type = type(e).__name__
                    error_message = str(e)
                    raise Exception(
                        "Error encountered while trying to save graph of"
                        f" E(H,orientation): {self._hdf5} - group {group}:"
                        f" {error_type}: {error_message}"
                    )
        if origin:
            return data

    def plot_3d(
        self,
        group: str,
        data_type: str,
        field_i: int,
        temp_i: int,
        show=True,
        save=False,
        colour_map_name="dark_rainbow_r_l",
        lim_scalar=1.0,
        ticks=1.0,
        r_density=0,
        c_density=0,
        axis_off=False,
    ):
        """
        Function that creates 3d plots of data dependent on field (B[T]) and temperature(T[K])

        Parameters
        ----------
        group: str
            name of a group from hdf5 file for which plot will be created
        data_type: str
            type of data that will be used to create plot, can only be one from 3 types: susceptibility,
            hemholtz_energy or magnetisation
        field_i: int
            index of field from dataset that will be used for plot
        temp_i: int
            index of temperature from dataset that will be used for plot
        show: bool = True
            determines if plot is shown, currently there is no reason to setting it to False
        save: bool = False
            determines if plot is saved, name of the file will be in following format: f'{group}_3d_{data_type}.tiff'
        colour_map_name: str or list = 'dark_rainbow_r_l'
            input of Compound.colour_map function
        lim_scalar: float = 1.
            scalar used to set limits of axes, smaller values magm
        ticks: float = 1.
        r_density: int = 0
            determines rcount of 3D plot
        c_density: int = 0
            determines ccount of 3D plot
        axis_off: bool = False
            determines if axes are turned off

        Returns
        -------

        """
        T = False
        if data_type == "chit":
            x = self[f"{group}_3d_susceptibility", f"{group}_chit_3d"][
                0, field_i, temp_i, :, :
            ]
            y = self[f"{group}_3d_susceptibility", f"{group}_chit_3d"][
                1, field_i, temp_i, :, :
            ]
            z = self[f"{group}_3d_susceptibility", f"{group}_chit_3d"][
                2, field_i, temp_i, :, :
            ]
            description = (
                "ChiT dependance on direction,"
                f" B={self[f'{group}_3d_susceptibility', f'{group}_fields'][field_i]} T={self[f'{group}_3d_susceptibility', f'{group}_temperatures'][temp_i]}"
            )
            T = True
        elif data_type == "chi":
            x = self[f"{group}_3d_susceptibility", f"{group}_chi_3d"][
                0, field_i, temp_i, :, :
            ]
            y = self[f"{group}_3d_susceptibility", f"{group}_chi_3d"][
                1, field_i, temp_i, :, :
            ]
            z = self[f"{group}_3d_susceptibility", f"{group}_chi_3d"][
                2, field_i, temp_i, :, :
            ]
            description = (
                "Chi dependance on direction,"
                f" B={self[f'{group}_3d_susceptibility', f'{group}_fields'][field_i]} T,"
                f" T={self[f'{group}_3d_susceptibility', f'{group}_temperatures'][temp_i]} K"
            )
        elif data_type == "hemholtz_energy":
            x = self[f"{group}_3d_hemholtz_energy", f"{group}_energy_3d"][
                0, field_i, temp_i, :, :
            ]
            y = self[f"{group}_3d_hemholtz_energy", f"{group}_energy_3d"][
                1, field_i, temp_i, :, :
            ]
            z = self[f"{group}_3d_hemholtz_energy", f"{group}_energy_3d"][
                2, field_i, temp_i, :, :
            ]
            description = (
                "Energy dependence on direction,"
                f" B={self[f'{group}_3d_hemholtz_energy', f'{group}_fields'][field_i]} T,"
                f" T={self[f'{group}_3d_hemholtz_energy', f'{group}_temperatures'][temp_i]} K"
            )
        elif data_type == "magnetisation":
            x = self[f"{group}_3d_magnetisation", f"{group}_mag_3d"][
                0, field_i, temp_i, :, :
            ]
            y = self[f"{group}_3d_magnetisation", f"{group}_mag_3d"][
                1, field_i, temp_i, :, :
            ]
            z = self[f"{group}_3d_magnetisation", f"{group}_mag_3d"][
                2, field_i, temp_i, :, :
            ]
            description = (
                "Magnetisation dependence on direction,"
                f" B={self[f'{group}_3d_magnetisation', f'{group}_fields'][field_i]} T,"
                f" T={self[f'{group}_3d_magnetisation', f'{group}_temperatures'][temp_i]} K"
            )
        title = description
        if show:
            fig = plt.figure()
            ax = fig.add_subplot(projection="3d")
            max_array = np.array([np.max(x), np.max(y), np.max(z)])
            lim = np.max(max_array)
            norm = plt.Normalize(z.min(), z.max())
            colors = Compound.colour_map(colour_map_name)(norm(z))
            rcount, ccount, _ = colors.shape
            if not r_density:
                r_density = rcount
            if not c_density:
                c_density = ccount
            (surface,) = ax.plot_surface(
                x,
                y,
                z,
                rcount=r_density,
                ccount=c_density,
                facecolors=colors,
                shade=False,
            )
            surface.set_facecolor((0, 0, 0, 0))
            ax.set_xlim(-lim * lim_scalar, lim * lim_scalar)
            ax.set_ylim(-lim * lim_scalar, lim * lim_scalar)
            ax.set_zlim(-lim * lim_scalar, lim * lim_scalar)
            # Important order of operations!
            if data_type == "susceptibility":
                if T:
                    ax.set_xlabel(
                        r"$\chi_{\mathrm{M}}T\ /\ \mathrm{cm^{3}mol^{-1}K}$",
                        labelpad=20 * len(str(ticks)) / 4,
                    )
                    ax.set_ylabel(
                        r"$\chi_{\mathrm{M}}T\ /\ \mathrm{cm^{3}mol^{-1}K}$",
                        labelpad=20 * len(str(ticks)) / 4,
                    )
                    ax.set_zlabel(
                        r"$\chi_{\mathrm{M}}T\ /\ \mathrm{cm^{3}mol^{-1}K}$",
                        labelpad=20 * len(str(ticks)) / 4,
                    )
                else:
                    ax.set_xlabel(
                        r"$\chi_{\mathrm{M}}\ /\ \mathrm{cm^{3}mol^{-1}}$",
                        labelpad=20 * len(str(ticks)) / 4,
                    )
                    ax.set_ylabel(
                        r"$\chi_{\mathrm{M}}\ /\ \mathrm{cm^{3}mol^{-1}}$",
                        labelpad=20 * len(str(ticks)) / 4,
                    )
                    ax.set_zlabel(
                        r"$\chi_{\mathrm{M}}\ /\ \mathrm{cm^{3}mol^{-1}}$",
                        labelpad=20 * len(str(ticks)) / 4,
                    )
            elif data_type == "hemholtz_energy":
                ax.set_xlabel(
                    r"$E\ /\ \mathrm{cm^{-1}}$",
                    labelpad=20 * len(str(ticks)) / 4,
                )
                ax.set_ylabel(
                    r"$E\ /\ \mathrm{cm^{-1}}$",
                    labelpad=20 * len(str(ticks)) / 4,
                )
                ax.set_zlabel(
                    r"$E\ /\ \mathrm{cm^{-1}}$",
                    labelpad=20 * len(str(ticks)) / 4,
                )
            elif data_type == "magnetisation":
                ax.set_xlabel(
                    r"$M\ /\ \mathrm{\mu_{B}}$",
                    labelpad=10 * len(str(ticks)) / 4,
                )
                ax.set_ylabel(
                    r"$M\ /\ \mathrm{\mu_{B}}$",
                    labelpad=10 * len(str(ticks)) / 4,
                )
                ax.set_zlabel(
                    r"$M\ /\ \mathrm{\mu_{B}}$",
                    labelpad=10 * len(str(ticks)) / 4,
                )
            if ticks == 0:
                for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
                    axis.set_ticklabels([])
                    axis._axinfo["axisline"]["linewidth"] = 1
                    axis._axinfo["axisline"]["color"] = (0, 0, 0)
                    axis._axinfo["grid"]["linewidth"] = 0.5
                    axis._axinfo["grid"]["linestyle"] = "-"
                    axis._axinfo["grid"]["color"] = (0, 0, 0)
                    axis._axinfo["tick"]["inward_factor"] = 0.0
                    axis._axinfo["tick"]["outward_factor"] = 0.0
                    axis.set_pane_color((0.95, 0.95, 0.95))
            else:
                ax.xaxis.set_minor_locator(AutoMinorLocator(2))
                ax.yaxis.set_minor_locator(AutoMinorLocator(2))
                ax.zaxis.set_minor_locator(AutoMinorLocator(2))
                if not (not T and ticks == 1):
                    ax.xaxis.set_major_locator(MultipleLocator(ticks))
                    ax.yaxis.set_major_locator(MultipleLocator(ticks))
                    ax.zaxis.set_major_locator(MultipleLocator(ticks))
            ax.grid(False)

            ax.set_box_aspect([1, 1, 1])
            plt.title(title)
            if axis_off:
                plt.axis("off")
            # plt.tight_layout()
            plt.show()

            if save:
                if axis_off:
                    fig.savefig(
                        f"{group}_3d_{data_type}.tiff",
                        transparent=True,
                        dpi=600,
                    )
                fig.savefig(f"{group}_3d_{data_type}.tiff", dpi=600)

    def animate_3d(
        self,
        group: str,
        data_type: str,
        animation_variable: str,
        filename: str,
        i_start=0,
        i_end=30,
        i_constant=0,
        colour_map_name="dark_rainbow_r_l",
        lim_scalar=1,
        ticks=1,
        r_density=0,
        c_density=0,
        axis_off=False,
        fps=15,
        dpi=100,
        bar=True,
        bar_scale=False,
        bar_colour_map_name="dark_rainbow_r",
        temp_rounding=0,
        field_rounding=0,
    ):
        T = False
        if data_type == "chit":
            x0 = self[f"{group}_3d_susceptibility", f"{group}_chit_3d"][0]
            y0 = self[f"{group}_3d_susceptibility", f"{group}_chit_3d"][1]
            z0 = self[f"{group}_3d_susceptibility", f"{group}_chit_3d"][2]
            fields = self[f"{group}_3d_susceptibility", f"{group}_fields"]
            temps = self[f"{group}_3d_susceptibility", f"{group}_temperatures"]
            T = True
        elif data_type == "chi":
            x0 = self[f"{group}_3d_susceptibility", f"{group}_chi_3d"][0]
            y0 = self[f"{group}_3d_susceptibility", f"{group}_chi_3d"][1]
            z0 = self[f"{group}_3d_susceptibility", f"{group}_chi_3d"][2]
            fields = self[f"{group}_3d_susceptibility", f"{group}_fields"]
            temps = self[f"{group}_3d_susceptibility", f"{group}_temperatures"]
        elif data_type == "hemholtz_energy":
            x0 = self[f"{group}_3d_hemholtz_energy", f"{group}_energy_3d"][0]
            y0 = self[f"{group}_3d_hemholtz_energy", f"{group}_energy_3d"][1]
            z0 = self[f"{group}_3d_hemholtz_energy", f"{group}_energy_3d"][2]
            fields = self[f"{group}_3d_hemholtz_energy", f"{group}_fields"]
            temps = self[
                f"{group}_3d_hemholtz_energy", f"{group}_temperatures"
            ]
        elif data_type == "magnetisation":
            x0 = self[f"{group}_3d_magnetisation", f"{group}_mag_3d"][0]
            y0 = self[f"{group}_3d_magnetisation", f"{group}_mag_3d"][1]
            z0 = self[f"{group}_3d_magnetisation", f"{group}_mag_3d"][2]
            fields = self[f"{group}_3d_magnetisation", f"{group}_fields"]
            temps = self[f"{group}_3d_magnetisation", f"{group}_temperatures"]
        else:
            raise ValueError
        if animation_variable == "temperature":
            description = f"B={fields[i_constant]:.4f} T"
        else:
            description = f"T={temps[i_constant]:.4f} K"
        title = description

        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        if bar:
            colour = iter(
                Compound.colour_map(bar_colour_map_name)(
                    np.linspace(0, 1, i_end - i_start)
                )
            )
            indicator = np.linspace(0, 1, i_end - i_start)
            if bar_scale:

                def my_ticks(x, pos):
                    if animation_variable == "temperature":
                        return f"{round(x * temps[-1], temp_rounding)} K"
                    else:
                        return f"{round(x * fields[-1], field_rounding)} T"

        writer = PillowWriter(fps=fps)
        with writer.saving(fig, f"{filename}.gif", dpi):
            if animation_variable == "temperature":
                for i_temp in range(i_start, i_end):
                    x = x0[i_constant, i_temp, :, :]
                    y = y0[i_constant, i_temp, :, :]
                    z = z0[i_constant, i_temp, :, :]
                    if data_type == "chit":
                        ax.set_xlabel(
                            (
                                r"$\chi_{\mathrm{M}}T\ /\ \mathrm{cm^{3}mol^{-1}K}$"
                            ),
                            labelpad=20 * len(str(ticks)) / 4,
                        )
                        ax.set_ylabel(
                            (
                                r"$\chi_{\mathrm{M}}T\ /\ \mathrm{cm^{3}mol^{-1}K}$"
                            ),
                            labelpad=20 * len(str(ticks)) / 4,
                        )
                        ax.set_zlabel(
                            (
                                r"$\chi_{\mathrm{M}}T\ /\ \mathrm{cm^{3}mol^{-1}K}$"
                            ),
                            labelpad=20 * len(str(ticks)) / 4,
                        )
                    elif data_type == "chi":
                        ax.set_xlabel(
                            r"$\chi_{\mathrm{M}}\ /\ \mathrm{cm^{3}mol^{-1}}$",
                            labelpad=20 * len(str(ticks)) / 4,
                        )
                        ax.set_ylabel(
                            r"$\chi_{\mathrm{M}}\ /\ \mathrm{cm^{3}mol^{-1}}$",
                            labelpad=20 * len(str(ticks)) / 4,
                        )
                        ax.set_zlabel(
                            r"$\chi_{\mathrm{M}}\ /\ \mathrm{cm^{3}mol^{-1}}$",
                            labelpad=20 * len(str(ticks)) / 4,
                        )
                    elif data_type == "hemholtz_energy":
                        ax.set_xlabel(
                            r"$E\ /\ \mathrm{cm^{-1}}$",
                            labelpad=20 * len(str(ticks)) / 4,
                        )
                        ax.set_ylabel(
                            r"$E\ /\ \mathrm{cm^{-1}}$",
                            labelpad=20 * len(str(ticks)) / 4,
                        )
                        ax.set_zlabel(
                            r"$E\ /\ \mathrm{cm^{-1}}$",
                            labelpad=20 * len(str(ticks)) / 4,
                        )
                    elif data_type == "magnetisation":
                        ax.set_xlabel(
                            r"$M\ /\ \mathrm{\mu_{B}}$",
                            labelpad=10 * len(str(ticks)) / 4,
                        )
                        ax.set_ylabel(
                            r"$M\ /\ \mathrm{\mu_{B}}$",
                            labelpad=10 * len(str(ticks)) / 4,
                        )
                        ax.set_zlabel(
                            r"$M\ /\ \mathrm{\mu_{B}}$",
                            labelpad=10 * len(str(ticks)) / 4,
                        )
                    max_array = np.array([np.max(x), np.max(y), np.max(z)])
                    lim = np.max(max_array)
                    norm = plt.Normalize(z.min(), z.max())
                    colors = Compound.colour_map(colour_map_name)(norm(z))
                    rcount, ccount, _ = colors.shape
                    if not r_density:
                        r_density = rcount
                    if not c_density:
                        c_density = ccount
                    surface = ax.plot_surface(
                        x,
                        y,
                        z,
                        rcount=r_density,
                        ccount=c_density,
                        facecolors=colors,
                        shade=False,
                    )
                    surface.set_facecolor((0, 0, 0, 0))
                    ax.set_xlim(-lim * lim_scalar, lim * lim_scalar)
                    ax.set_ylim(-lim * lim_scalar, lim * lim_scalar)
                    ax.set_zlim(-lim * lim_scalar, lim * lim_scalar)
                    # Important order of operations!

                    if ticks == 0:
                        for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
                            axis.set_ticklabels([])
                            axis._axinfo["axisline"]["linewidth"] = 1
                            axis._axinfo["axisline"]["color"] = (0, 0, 0)
                            axis._axinfo["grid"]["linewidth"] = 0.5
                            axis._axinfo["grid"]["linestyle"] = "-"
                            axis._axinfo["grid"]["color"] = (0, 0, 0)
                            axis._axinfo["tick"]["inward_factor"] = 0.0
                            axis._axinfo["tick"]["outward_factor"] = 0.0
                            axis.set_pane_color((0.95, 0.95, 0.95))
                    else:
                        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
                        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
                        ax.zaxis.set_minor_locator(AutoMinorLocator(2))
                        if not (not T and ticks == 1):
                            ax.xaxis.set_major_locator(MultipleLocator(ticks))
                            ax.yaxis.set_major_locator(MultipleLocator(ticks))
                            ax.zaxis.set_major_locator(MultipleLocator(ticks))
                    ax.grid(False)

                    ax.set_box_aspect([1, 1, 1])
                    ax.set_title(title)
                    if axis_off:
                        plt.axis("off")

                    if bar:
                        c = next(colour)
                        axins = ax.inset_axes([0, 0.6, 0.098, 0.2])
                        axins.bar(
                            1, indicator[i_temp - i_start], width=0.2, color=c
                        )
                        axins.set_ylim(0, 1)
                        if not bar_scale:
                            axins.text(
                                1,
                                1,
                                s=f"{round(temps[-1], temp_rounding)} K",
                                verticalalignment="bottom",
                                horizontalalignment="center",
                            )
                            axins.text(
                                1,
                                -0.03,
                                s=f"{round(temps[0], temp_rounding)} K",
                                verticalalignment="top",
                                horizontalalignment="center",
                            )
                            axins.axison = False
                        if bar_scale:
                            axins.get_xaxis().set_visible(False)
                            axins.xaxis.set_tick_params(labelbottom=False)
                            axins.yaxis.set_major_formatter(
                                matplotlib.ticker.FuncFormatter(my_ticks)
                            )

                    writer.grab_frame()
                    plt.cla()

            elif animation_variable == "field":
                for i_field in range(i_start, i_end):
                    x = x0[i_field, i_constant, :, :]
                    y = y0[i_field, i_constant, :, :]
                    z = z0[i_field, i_constant, :, :]
                    if data_type == "chit":
                        ax.set_xlabel(
                            (
                                r"$\chi_{\mathrm{M}}T\ /\ \mathrm{cm^{3}mol^{-1}K}$"
                            ),
                            labelpad=20 * len(str(ticks)) / 4,
                        )
                        ax.set_ylabel(
                            (
                                r"$\chi_{\mathrm{M}}T\ /\ \mathrm{cm^{3}mol^{-1}K}$"
                            ),
                            labelpad=20 * len(str(ticks)) / 4,
                        )
                        ax.set_zlabel(
                            (
                                r"$\chi_{\mathrm{M}}T\ /\ \mathrm{cm^{3}mol^{-1}K}$"
                            ),
                            labelpad=20 * len(str(ticks)) / 4,
                        )
                    elif data_type == "chi":
                        ax.set_xlabel(
                            r"$\chi_{\mathrm{M}}\ /\ \mathrm{cm^{3}mol^{-1}}$",
                            labelpad=20 * len(str(ticks)) / 4,
                        )
                        ax.set_ylabel(
                            r"$\chi_{\mathrm{M}}\ /\ \mathrm{cm^{3}mol^{-1}}$",
                            labelpad=20 * len(str(ticks)) / 4,
                        )
                        ax.set_zlabel(
                            r"$\chi_{\mathrm{M}}\ /\ \mathrm{cm^{3}mol^{-1}}$",
                            labelpad=20 * len(str(ticks)) / 4,
                        )
                    elif data_type == "hemholtz_energy":
                        ax.set_xlabel(
                            r"$E\ /\ \mathrm{cm^{-1}}$",
                            labelpad=20 * len(str(ticks)) / 4,
                        )
                        ax.set_ylabel(
                            r"$E\ /\ \mathrm{cm^{-1}}$",
                            labelpad=20 * len(str(ticks)) / 4,
                        )
                        ax.set_zlabel(
                            r"$E\ /\ \mathrm{cm^{-1}}$",
                            labelpad=20 * len(str(ticks)) / 4,
                        )
                    elif data_type == "magnetisation":
                        ax.set_xlabel(
                            r"$M\ /\ \mathrm{\mu_{B}}$",
                            labelpad=10 * len(str(ticks)) / 4,
                        )
                        ax.set_ylabel(
                            r"$M\ /\ \mathrm{\mu_{B}}$",
                            labelpad=10 * len(str(ticks)) / 4,
                        )
                        ax.set_zlabel(
                            r"$M\ /\ \mathrm{\mu_{B}}$",
                            labelpad=10 * len(str(ticks)) / 4,
                        )
                    title = description
                    max_array = np.array([np.max(x), np.max(y), np.max(z)])
                    lim = np.max(max_array)
                    norm = plt.Normalize(z.min(), z.max())
                    colors = Compound.colour_map(colour_map_name)(norm(z))
                    rcount, ccount, _ = colors.shape
                    if not r_density:
                        r_density = rcount
                    if not c_density:
                        c_density = ccount
                    surface = ax.plot_surface(
                        x,
                        y,
                        z,
                        rcount=r_density,
                        ccount=c_density,
                        facecolors=colors,
                        shade=False,
                    )
                    surface.set_facecolor((0, 0, 0, 0))
                    ax.set_xlim(-lim * lim_scalar, lim * lim_scalar)
                    ax.set_ylim(-lim * lim_scalar, lim * lim_scalar)
                    ax.set_zlim(-lim * lim_scalar, lim * lim_scalar)
                    # Important order of operations!

                    if ticks == 0:
                        for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
                            axis.set_ticklabels([])
                            axis._axinfo["axisline"]["linewidth"] = 1
                            axis._axinfo["axisline"]["color"] = (0, 0, 0)
                            axis._axinfo["grid"]["linewidth"] = 0.5
                            axis._axinfo["grid"]["linestyle"] = "-"
                            axis._axinfo["grid"]["color"] = (0, 0, 0)
                            axis._axinfo["tick"]["inward_factor"] = 0.0
                            axis._axinfo["tick"]["outward_factor"] = 0.0
                            axis.set_pane_color((0.95, 0.95, 0.95))
                    else:
                        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
                        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
                        ax.zaxis.set_minor_locator(AutoMinorLocator(2))
                        if not (not T and ticks == 1):
                            ax.xaxis.set_major_locator(MultipleLocator(ticks))
                            ax.yaxis.set_major_locator(MultipleLocator(ticks))
                            ax.zaxis.set_major_locator(MultipleLocator(ticks))
                    ax.grid(False)

                    ax.set_box_aspect([1, 1, 1])
                    plt.title(title)
                    if axis_off:
                        plt.axis("off")

                    if bar:
                        c = next(colour)
                        axins = ax.inset_axes([0, 0.6, 0.098, 0.2])
                        axins.bar(
                            1, indicator[i_field - i_start], width=0.2, color=c
                        )
                        axins.set_ylim(0, 1)

                        if not bar_scale:
                            axins.text(
                                1,
                                1,
                                s=f"{round(fields[-1], field_rounding)} T",
                                verticalalignment="bottom",
                                horizontalalignment="center",
                            )
                            axins.text(
                                1,
                                -0.03,
                                s=f"{round(fields[0], field_rounding)} T",
                                verticalalignment="top",
                                horizontalalignment="center",
                            )
                            axins.axison = False
                        if bar_scale:
                            axins.get_xaxis().set_visible(False)
                            axins.xaxis.set_tick_params(labelbottom=False)
                            axins.yaxis.set_major_formatter(
                                matplotlib.ticker.FuncFormatter(my_ticks)
                            )

                    writer.grab_frame()
                    plt.cla()
        plt.close()

    def interactive_plot_3d(
        self,
        group: str,
        data_type: str,
        colour_map_name="dark_rainbow_r",
        T_slider_colour="#77f285",
        B_slider_colour="#794285",
        temp_bar_colour_map_name="BuRd",
        field_bar_colour_map_name="BuPi",
        lim_scalar=1,
        ticks=1,
        bar=True,
        axis_off=False,
    ):
        field_i, temp_i = 0, 0

        T = False
        if data_type == "chit":
            x0 = self[f"{group}_3d_susceptibility", f"{group}_chit_3d"][0]
            y0 = self[f"{group}_3d_susceptibility", f"{group}_chit_3d"][1]
            z0 = self[f"{group}_3d_susceptibility", f"{group}_chit_3d"][2]
            fields = self[f"{group}_3d_susceptibility", f"{group}_fields"]
            temps = self[f"{group}_3d_susceptibility", f"{group}_temperatures"]
            T = True
        if data_type == "chi":
            x0 = self[f"{group}_3d_susceptibility", f"{group}_chi_3d"][0]
            y0 = self[f"{group}_3d_susceptibility", f"{group}_chi_3d"][1]
            z0 = self[f"{group}_3d_susceptibility", f"{group}_chi_3d"][2]
            fields = self[f"{group}_3d_susceptibility", f"{group}_fields"]
            temps = self[f"{group}_3d_susceptibility", f"{group}_temperatures"]

        elif data_type == "hemholtz_energy":
            x0 = self[f"{group}_3d_hemholtz_energy", f"{group}_energy_3d"][0]
            y0 = self[f"{group}_3d_hemholtz_energy", f"{group}_energy_3d"][1]
            z0 = self[f"{group}_3d_hemholtz_energy", f"{group}_energy_3d"][2]
            fields = self[f"{group}_3d_hemholtz_energy", f"{group}_fields"]
            temps = self[
                f"{group}_3d_hemholtz_energy", f"{group}_temperatures"
            ]

        elif data_type == "magnetisation":
            x0 = self[f"{group}_3d_magnetisation", f"{group}_mag_3d"][0]
            y0 = self[f"{group}_3d_magnetisation", f"{group}_mag_3d"][1]
            z0 = self[f"{group}_3d_magnetisation", f"{group}_mag_3d"][2]
            fields = self[f"{group}_3d_magnetisation", f"{group}_fields"]
            temps = self[f"{group}_3d_magnetisation", f"{group}_temperatures"]

        fig = plt.figure()
        global ax
        ax = fig.add_subplot(projection="3d")

        colour1 = Compound.colour_map(temp_bar_colour_map_name)(
            np.linspace(0, 1, len(temps))
        )
        colour2 = Compound.colour_map(field_bar_colour_map_name)(
            np.linspace(0, 1, len(fields))
        )

        indicator1 = np.linspace(0, 1, len(temps))
        indicator2 = np.linspace(0, 1, len(fields))

        x = x0[field_i, temp_i, :, :]
        y = y0[field_i, temp_i, :, :]
        z = z0[field_i, temp_i, :, :]

        max_array = np.array([np.max(x), np.max(y), np.max(z)])
        lim = np.max(max_array)
        norm = plt.Normalize(z.min(), z.max())
        colors = Compound.colour_map(colour_map_name)(norm(z))
        rcount, ccount, _ = colors.shape
        r_density = rcount
        c_density = ccount
        surface = ax.plot_surface(
            x,
            y,
            z,
            rcount=r_density,
            ccount=c_density,
            facecolors=colors,
            shade=False,
        )
        surface.set_facecolor((0, 0, 0, 0))
        ax.set_xlim(-lim * lim_scalar, lim * lim_scalar)
        ax.set_ylim(-lim * lim_scalar, lim * lim_scalar)
        ax.set_zlim(-lim * lim_scalar, lim * lim_scalar)
        # Important order of operations!
        if data_type in "chit":
            if T:
                ax.set_xlabel(
                    r"$\chi_{\mathrm{M}}T\ /\ \mathrm{cm^{3}mol^{-1}K}$",
                    labelpad=20 * len(str(ticks)) / 4,
                )
                ax.set_ylabel(
                    r"$\chi_{\mathrm{M}}T\ /\ \mathrm{cm^{3}mol^{-1}K}$",
                    labelpad=20 * len(str(ticks)) / 4,
                )
                ax.set_zlabel(
                    r"$\chi_{\mathrm{M}}T\ /\ \mathrm{cm^{3}mol^{-1}K}$",
                    labelpad=20 * len(str(ticks)) / 4,
                )
            else:
                ax.set_xlabel(
                    r"$\chi_{\mathrm{M}}\ /\ \mathrm{cm^{3}mol^{-1}}$",
                    labelpad=20 * len(str(ticks)) / 4,
                )
                ax.set_ylabel(
                    r"$\chi_{\mathrm{M}}\ /\ \mathrm{cm^{3}mol^{-1}}$",
                    labelpad=20 * len(str(ticks)) / 4,
                )
                ax.set_zlabel(
                    r"$\chi_{\mathrm{M}}\ /\ \mathrm{cm^{3}mol^{-1}}$",
                    labelpad=20 * len(str(ticks)) / 4,
                )
        elif data_type == "hemholtz_energy":
            ax.set_xlabel(
                r"$E\ /\ \mathrm{cm^{-1}}$", labelpad=20 * len(str(ticks)) / 4
            )
            ax.set_ylabel(
                r"$E\ /\ \mathrm{cm^{-1}}$", labelpad=20 * len(str(ticks)) / 4
            )
            ax.set_zlabel(
                r"$E\ /\ \mathrm{cm^{-1}}$", labelpad=20 * len(str(ticks)) / 4
            )
        elif data_type == "magnetisation":
            ax.set_xlabel(
                r"$M\ /\ \mathrm{\mu_{B}}$", labelpad=20 * len(str(ticks)) / 4
            )
            ax.set_ylabel(
                r"$M\ /\ \mathrm{\mu_{B}}$", labelpad=20 * len(str(ticks)) / 4
            )
            ax.set_zlabel(
                r"$M\ /\ \mathrm{\mu_{B}}$", labelpad=20 * len(str(ticks)) / 4
            )
        if ticks == 0:
            for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
                axis.set_ticklabels([])
                axis._axinfo["axisline"]["linewidth"] = 1
                axis._axinfo["axisline"]["color"] = (0, 0, 0)
                axis._axinfo["grid"]["linewidth"] = 0.5
                axis._axinfo["grid"]["linestyle"] = "-"
                axis._axinfo["grid"]["color"] = (0, 0, 0)
                axis._axinfo["tick"]["inward_factor"] = 0.0
                axis._axinfo["tick"]["outward_factor"] = 0.0
                axis.set_pane_color((0.95, 0.95, 0.95))
        else:
            ax.xaxis.set_minor_locator(AutoMinorLocator(2))
            ax.yaxis.set_minor_locator(AutoMinorLocator(2))
            ax.zaxis.set_minor_locator(AutoMinorLocator(2))
            if not (not T and ticks == 1):
                ax.xaxis.set_major_locator(MultipleLocator(ticks))
                ax.yaxis.set_major_locator(MultipleLocator(ticks))
                ax.zaxis.set_major_locator(MultipleLocator(ticks))
        ax.grid(False)

        ax.set_box_aspect([1, 1, 1])
        # plt.title(title)
        fig.subplots_adjust(left=0.1)
        if bar:
            c = colour1[temp_i]
            axins = ax.inset_axes([-0.05, 0.7, 0.1, 0.2])
            axins.bar(1, indicator1[temp_i], width=0.2, color=c)
            axins.set_ylim(0, 1)
            axins.text(
                1,
                1,
                s=f"{round(temps[-1], 1)} K",
                verticalalignment="bottom",
                horizontalalignment="center",
            )
            axins.text(
                1,
                -0.03,
                s=f"{round(temps[0], 1)} K",
                verticalalignment="top",
                horizontalalignment="center",
            )

            # axins.get_xaxis().set_visible(False)
            # axins.xaxis.set_tick_params(labelbottom=False)

            axins.axison = False

            c = colour2[field_i]
            axins2 = ax.inset_axes([-0.05, 0.2, 0.1, 0.2])
            axins2.bar(1, indicator2[field_i], width=0.2, color=c)
            axins2.set_ylim(0, 1)
            axins2.text(
                1,
                1,
                s=f"{round(fields[-1], 1)} T",
                verticalalignment="bottom",
                horizontalalignment="center",
            )
            axins2.text(
                1,
                -0.03,
                s=f"{round(fields[0], 1)} T",
                verticalalignment="top",
                horizontalalignment="center",
            )
            axins2.axison = False

        ax_temp = fig.add_axes([0.05, 0.6, 0.1, 0.2])
        ax_temp.axison = False

        ax_field = fig.add_axes([0.05, 0.3, 0.1, 0.2])
        ax_field.axison = False

        slider_temp = Slider(
            ax_temp,
            "T [index]",
            valmin=0,
            valmax=temps.size - 1,
            orientation="vertical",
            valstep=1,
            initcolor=None,
            color=T_slider_colour,
        )
        slider_field = Slider(
            ax_field,
            "B [index]",
            valmin=0,
            valmax=fields.size - 1,
            orientation="vertical",
            initcolor=None,
            valstep=1,
            color=B_slider_colour,
        )

        def slider_update(val):
            temp_i = slider_temp.val
            field_i = slider_field.val

            ax.cla()
            x = x0[field_i, temp_i, :, :]
            y = y0[field_i, temp_i, :, :]
            z = z0[field_i, temp_i, :, :]

            max_array = np.array([np.max(x), np.max(y), np.max(z)])
            lim = np.max(max_array)
            norm = plt.Normalize(z.min(), z.max())
            colors = Compound.colour_map(colour_map_name)(norm(z))
            rcount, ccount, _ = colors.shape
            r_density = rcount
            c_density = ccount
            surface = ax.plot_surface(
                x,
                y,
                z,
                rcount=r_density,
                ccount=c_density,
                facecolors=colors,
                shade=False,
            )
            surface.set_facecolor((0, 0, 0, 0))
            ax.set_xlim(-lim * lim_scalar, lim * lim_scalar)
            ax.set_ylim(-lim * lim_scalar, lim * lim_scalar)
            ax.set_zlim(-lim * lim_scalar, lim * lim_scalar)
            ax.set_title(f"B={fields[field_i]:.4f} T, T={temps[temp_i]:.4f} K")
            # Important order of operations!
            # if data_type in 'chit':
            #     if T:
            #         ax.set_xlabel(r'$\chi_{\mathrm{M}}T\ /\ \mathrm{cm^{3}mol^{-1}K}$',
            #                       labelpad=20 * len(str(ticks)) / 4)
            #         ax.set_ylabel(r'$\chi_{\mathrm{M}}T\ /\ \mathrm{cm^{3}mol^{-1}K}$',
            #                       labelpad=20 * len(str(ticks)) / 4)
            #         ax.set_zlabel(r'$\chi_{\mathrm{M}}T\ /\ \mathrm{cm^{3}mol^{-1}K}$',
            #                       labelpad=20 * len(str(ticks)) / 4)
            #     else:
            #         ax.set_xlabel(r'$\chi_{\mathrm{M}}\ /\ \mathrm{cm^{3}mol^{-1}}$', labelpad=20 * len(str(ticks)) / 4)
            #         ax.set_ylabel(r'$\chi_{\mathrm{M}}\ /\ \mathrm{cm^{3}mol^{-1}}$', labelpad=20 * len(str(ticks)) / 4)
            #         ax.set_zlabel(r'$\chi_{\mathrm{M}}\ /\ \mathrm{cm^{3}mol^{-1}}$', labelpad=20 * len(str(ticks)) / 4)
            # elif data_type == 'hemholtz_energy':
            #     ax.set_xlabel(r'$E\ /\ \mathrm{cm^{-1}}$', labelpad=20 * len(str(ticks)) / 4)
            #     ax.set_ylabel(r'$E\ /\ \mathrm{cm^{-1}}$', labelpad=20 * len(str(ticks)) / 4)
            #     ax.set_zlabel(r'$E\ /\ \mathrm{cm^{-1}}$', labelpad=20 * len(str(ticks)) / 4)
            # elif data_type == 'magnetisation':
            #     ax.set_xlabel(r'$M\ /\ \mathrm{\mu_{B}}$', labelpad=10 * len(str(ticks)) / 4)
            #     ax.set_ylabel(r'$M\ /\ \mathrm{\mu_{B}}$', labelpad=10 * len(str(ticks)) / 4)
            #     ax.set_zlabel(r'$M\ /\ \mathrm{\mu_{B}}$', labelpad=10 * len(str(ticks)) / 4)
            # if ticks == 0:
            #     for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
            #         axis.set_ticklabels([])
            #         axis._axinfo['axisline']['linewidth'] = 1
            #         axis._axinfo['axisline']['color'] = (0, 0, 0)
            #         axis._axinfo['grid']['linewidth'] = 0.5
            #         axis._axinfo['grid']['linestyle'] = "-"
            #         axis._axinfo['grid']['color'] = (0, 0, 0)
            #         axis._axinfo['tick']['inward_factor'] = 0.0
            #         axis._axinfo['tick']['outward_factor'] = 0.0
            #         axis.set_pane_color((0.95, 0.95, 0.95))
            # else:
            #     ax.xaxis.set_minor_locator(AutoMinorLocator(2))
            #     ax.yaxis.set_minor_locator(AutoMinorLocator(2))
            #     ax.zaxis.set_minor_locator(AutoMinorLocator(2))
            #     if not (not T and ticks == 1):
            #         ax.xaxis.set_major_locator(MultipleLocator(ticks))
            #         ax.yaxis.set_major_locator(MultipleLocator(ticks))
            #         ax.zaxis.set_major_locator(MultipleLocator(ticks))
            ax.grid(False)
            #
            # ax.set_box_aspect([1, 1, 1])
            # plt.title(title)
            fig.subplots_adjust(left=0.1)
            if bar:
                c = colour1[temp_i]
                axins = ax.inset_axes([-0.05, 0.7, 0.1, 0.2])
                axins.bar(1, indicator1[temp_i], width=0.2, color=c)
                axins.set_ylim(0, 1)
                axins.text(
                    1,
                    1,
                    s=f"{round(temps[-1], 1)} K",
                    verticalalignment="bottom",
                    horizontalalignment="center",
                )
                axins.text(
                    1,
                    -0.03,
                    s=f"{round(temps[0], 1)} K",
                    verticalalignment="top",
                    horizontalalignment="center",
                )

                # axins.get_xaxis().set_visible(False)
                # axins.xaxis.set_tick_params(labelbottom=False)

                axins.axison = False

                c = colour2[field_i]
                axins2 = ax.inset_axes([-0.05, 0.2, 0.1, 0.2])
                axins2.bar(1, indicator2[field_i], width=0.2, color=c)
                axins2.set_ylim(0, 1)
                axins2.text(
                    1,
                    1,
                    s=f"{round(fields[-1], 1)} T",
                    verticalalignment="bottom",
                    horizontalalignment="center",
                )
                axins2.text(
                    1,
                    -0.03,
                    s=f"{round(fields[0], 1)} T",
                    verticalalignment="top",
                    horizontalalignment="center",
                )
                axins2.axison = False

            fig.canvas.draw()

        slider_temp.on_changed(slider_update)
        slider_field.on_changed(slider_update)
        if axis_off:
            plt.axis("off")
        # plt.tight_layout()
        plt.show()
