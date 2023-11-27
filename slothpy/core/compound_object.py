# SlothPy
# Copyright (C) 2023 Mikolaj Tadeusz Zychowicz (MTZ)

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from os import path
from functools import partial
from typing import Tuple, Union
from h5py import File, Group, Dataset
from numpy import (
    ndarray,
    array,
    float64,
    int64,
    complex128,
    linspace,
    arange,
    max,
    newaxis,
    allclose,
    identity,
)
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from matplotlib.gridspec import GridSpec
from matplotlib.animation import PillowWriter
from matplotlib.widgets import Slider
from matplotlib.ticker import FuncFormatter
from matplotlib.pyplot import (
    plot,
    figure,
    subplots,
    rc,
    tight_layout,
    title,
    cla,
    close,
)
from slothpy.core._slothpy_exceptions import (
    SltFileError,
    SltCompError,
    SltSaveError,
    SltReadError,
    SltInputError,
    SltPlotError,
)
from slothpy._general_utilities._constants import (
    RED,
    GREEN,
    BLUE,
    PURPLE,
    YELLOW,
    RESET,
)
from slothpy._magnetism._g_tensor import _g_tensor_and_axes_doublet

from slothpy._magnetism._magnetisation_dask_2 import _mth
from slothpy._magnetism._magnetisation import _mag_3d
from slothpy._magnetism._susceptibility import (
    _chitht,
    _chitht_tensor,
    _chit_3d,
)
from slothpy._magnetism._zeeman import (
    _zeeman_splitting,
    _get_zeeman_matrix,
    _helmholtz_energyth,
    _helmholtz_energy_3d,
)
from slothpy._general_utilities._grids_over_hemisphere import (
    lebedev_laikov_grid,
)
from slothpy._general_utilities._io import (
    _group_exists,
    _get_soc_energies_cm_1,
    _get_states_magnetic_momenta,
    _get_states_total_angular_momenta,
    _get_total_angular_momneta_matrix,
    _get_magnetic_momenta_matrix,
)
from slothpy._angular_momentum._pseudo_spin_ito import (
    _get_decomposition_in_z_pseudo_spin_basis,
    _ito_real_decomp_matrix,
    _ito_complex_decomp_matrix,
    _get_soc_matrix_in_z_pseudo_spin_basis,
    _get_zeeman_matrix_in_z_pseudo_spin_basis,
    _matrix_from_ito_complex,
    _matrix_from_ito_real,
)
from slothpy._general_utilities._math_expresions import (
    _normalize_grid_vectors,
    _normalize_orientations,
    _normalize_orientation,
)
from slothpy._general_utilities._auto_tune import _auto_tune
from slothpy._general_utilities._ploting_utilities import (
    colour_map,
    _custom_colour_cycler,
)
from slothpy._general_utilities._ploting_utilities import _display_plot


class Compound:
    """
    The core object constituting the API and access to all the methods.
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
        value : ndarray
            An ArrayLike structure (can be converted to ndarray) that will
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
            file with the suffix: _g_tensors_axes, by default None.

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

        Note
        ----
        Magnetic axes are returned in the form of rotation matrices that
        diagonalise the Abragam-Bleaney tensor (G = gg.T). Coordinates of the
        main axes XYZ in the initial xzy frame are columns of such matrices
        (0-X, 1-Y, 2-Z).

        See Also
        --------
        slothpy.exporting.table_energy_and_g,
        slothpy.exporting.axes_in_mol2, slothpy.exporting.axes_in_xyz
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
                    "Dataset containing number of doublet and respective"
                    f" g-tensors from Group {group}.",
                    f"Group({slt}) containing g-tensors of doublets and"
                    f" their magnetic axes calculated from Group: {group}.",
                ] = g_tensor_list[:, :]

                self[
                    slt_group_name,
                    f"{slt}_axes",
                    "Dataset containing rotation matrices from the initial"
                    " coordinate system to the magnetic axes of respective"
                    f" g-tensors from Group: {group}.",
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
            [[direction_x, direction_y, direction_z, 1.]]. Custom grids will be
            automatically normalized.
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
            file with suffix: _magnetisation., by default None
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

        Note
        -----
        Here, (number_cpu // number_threads) parallel processes are used to
        distribute the workload over the provided field values.

        See Also
        --------
        slothpy.Compound.plot_mth,
        slothpy.lebedev_laikov_grid : For the description of the prescribed
                                      Lebedev-Laikov grids.
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
            grid = lebedev_laikov_grid(grid)
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

        # try:
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
        # except Exception as exc:
        #     raise SltCompError(
        #         self._hdf5,
        #         exc,
        #         "Failed to compute M(T,H) from "
        #         + BLUE
        #         + "Group "
        #         + RESET
        #         + '"'
        #         + BLUE
        #         + f"{group}"
        #         + RESET
        #         + '".',
        #     ) from None

        if slt is not None:
            try:
                self[
                    slt_group_name,
                    f"{slt}_mth",
                    "Dataset containing M(T,H) magnetisation (T - rows, H"
                    f" - columns) calculated from group: {group}.",
                    f"Group({slt}) containing M(T,H) magnetisation"
                    f" calculated from group: {group}.",
                ] = mth_array[:, :]
                self[
                    slt_group_name,
                    f"{slt}_fields",
                    "Dataset containing magnetic field H values used in"
                    f" simulation of M(T,H) from group: {group}.",
                ] = fields[:]
                self[
                    slt_group_name,
                    f"{slt}_temperatures",
                    "Dataset containing temperature T values used in"
                    f" simulation of M(T,H) from group: {group}.",
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
        rotation: ndarray[float64] = None,
        slt: str = None,
        autotune: bool = False,
        _autotune_size: int = 2,
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
            temperature values (K) at which 3D magnetisation will be computed.
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
        rotation : ndarray[float64], optional
            A (3,3) orthogonal rotation matrix used to rotate momenta matrices.
            Note that the inverse matrix has to be given to rotate the
            reference frame instead. It is useful here to orient your 3D plots
            more conveniently., by default None
        slt : str, optional
            If given the results will be saved in a group of this name to .slt
            file with suffix: _3d_magnetisation., by default None
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

        Note
        -----
        Here, (number_cpu // number_threads) parallel processes are used to
        distribute the workload over len(fields)*2*shperical_grid**2 tasks. Be
        aware that the resulting arrays and computations can quickly consume
        much memory (e.g. for a calculation with 100 field values 1-10 T, 300
        temperatures 1-300 K, and spherical_grid = 60, the resulting array will
        take 3*100*300*2*60*60*8 bytes = 5.184 GB).

        See Also
        --------
        slothpy.Compound.plot_3d, slothpy.Compound.interactive_plot_3d,
        slothpy.Compound.animate_3d
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
            temperatures = array(temperatures, dtype=float64)
            fields = array(fields, dtype=float64)
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
                    1,  # Single grid point in the inner loop
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
                fields,
                spherical_grid,
                temperatures,
                states_cutoff,
                number_cpu,
                number_threads,
                rotation,
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
                    "Dataset containing 3D magnetisation as meshgird"
                    " (0-x,1-y,2-z) arrays over sphere (xyz, field,"
                    " temperature, meshgrid, meshgrid) calculated from"
                    f" group: {group}.",
                    f"Group({slt}) containing 3D magnetisation calculated"
                    f" from group: {group}.",
                ] = mag_3d_array[:, :, :, :, :]
                self[
                    slt_group_name,
                    f"{slt}_fields",
                    "Dataset containing magnetic field H values used in"
                    f" simulation of 3D magnetisation from group: {group}.",
                ] = fields[:]
                self[
                    slt_group_name,
                    f"{slt}_temperatures",
                    "Dataset containing temperature T values used in"
                    f" simulation of 3D magnetisation from group: {group}.",
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
        number_of_points: int = 1,
        delta_h: float = 0.0001,
        states_cutoff: int = 0,
        number_cpu: int = 0,
        number_threads: int = 1,
        exp: bool = False,
        T: bool = True,
        grid: Union[int, ndarray[float64]] = None,
        slt: str = None,
        autotune: bool = False,
        _autotune_size: int = 2,
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
        number_of_points : int, optional
            Controls the number of points for numerical differentiation over
            the magnetic field values using the finite difference method with
            a symmetrical stencil. The total number of used points =
            (2 * num_of_opints + 1), therefore 1 is a minimum value to obtain
            the first derivative using 3 points - including the value at the
            point at which the derivative is taken. In this regard, the value 0
            triggers the experimentalist model for susceptibility.,
            by default 1
        delta_h : float64, optional
            Value of field step used for numerical differentiation using finite
            difference method. 0.0001 (T) = 1 Oe is recommended as a starting
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
        grid : Union[int, ndarray[float64]], optional
            If the grid is set to an integer from 0-11 then the prescribed
            Lebedev-Laikov grids over the hemisphere will be used (see
            grids_over_hemisphere documentation), otherwise, the user can
            provide an ArrayLike structure (can be converted to numpy.NDArray)
            with the convention: [[direction_x, direction_y, direction_z,
            weight],...] for powder-averaging. If one wants a calculation for a
            single, particular direction the list has to contain one entry like
            this: [[direction_x, direction_y, direction_z, 1.]]. If not given
            the average is taken over xyz directions, which is sufficient for a
            second rank tensor. Custom grids will be automatically normalized.,
            by default None
        slt : str, optional
            If given the results will be saved in a group of this name to .slt
            file with suffix: _susceptibility., by default None
        autotune : bool, optional
            If True the program will automatically try to choose the best
            number of threads (and therefore parallel processes), for the given
            number of CPUs, to be used during the calculation. Note that this
            process can take a significant amount of time, so start to use it
            with medium-sized calculations (e.g. for states_cutoff > 300 with
            a higher number of field values and number_of_points) where it
            becomes a necessity., by default False

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
            If the calculation of magnetic susceptibility is unsuccessful.
        SltFileError
            If the program is unable to correctly save results to .slt file.

        Note
        -----
        Here, (number_cpu // number_threads) parallel processes are used to
        distribute the workload over fields.size*(2*number_of_points+1) tasks.

        See Also
        --------
        slothpy.Compound.plot_chitht
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
            grid = lebedev_laikov_grid(grid)
        elif grid is not None:
            grid = _normalize_grid_vectors(grid)

        if autotune:
            if exp:
                num_to_parallel = fields.size
            else:
                num_to_parallel = (2 * number_of_points + 1) * fields.size

            if grid is None:
                grid_shape = 3  # xyz grid in the inner loop
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
                    f"Dataset containing {chi_name} magnetic"
                    " susceptibility (H - rows, T - columns) calculated"
                    f" from group: {group}.",
                    f"Group({slt}) containing {chi_name} magnetic"
                    f" susceptibility calculated from group: {group}.",
                ] = chitht_array[:, :]
                self[
                    slt_group_name,
                    f"{slt}_fields",
                    "Dataset containing magnetic field H values used in"
                    " simulation of magnetic susceptibility from group:"
                    f" {group}.",
                ] = fields[:]
                self[
                    slt_group_name,
                    f"{slt}_temperatures",
                    "Dataset containing temperature T values used in"
                    f" simulation of {chi_name} from group: {group}.",
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
        number_of_points: int = 1,
        delta_h: float = 0.0001,
        states_cutoff: int = 0,
        number_cpu: int = 0,
        number_threads: int = 1,
        exp: bool = False,
        T: bool = True,
        rotation: ndarray[float64] = None,
        slt: str = None,
        autotune: bool = False,
        _autotune_size: int = 2,
    ) -> ndarray[float64]:
        """
        Calculates magnetic susceptibility chi(H,T) (Van Vleck) tensor for
        a given list of field and temperature values.

        Parameters
        ----------
        group : str
            Name of a group containing results of relativistic ab initio
            calculations used for the computation of the magnetisation.
        temperatures : ndarray[float64]
            ArrayLike structure (can be converted to numpy.NDArray) of
            temeperature values (K) at which magnetic susceptibility tensor
            will be computed.
        fields : ndarray[float64]
            ArrayLike structure (can be converted to numpy.NDArray) of field
            values (T) at which magnetic susceptibility tensor will be
            computed.
        number_of_points : int, optional
            Controls the number of points for numerical differentiation over
            the magnetic field values using the finite difference method with
            a symmetrical stencil. The total number of used points =
            (2 * num_of_opints + 1), therefore 1 is a minimum value to obtain
            the first derivative using 3 points - including the value at the
            point at which the derivative is taken. In this regard, the value 0
            triggers the experimentalist model for susceptibility.,
            by default 1
        delta_h : float64, optional
            Value of field step used for numerical differentiation using finite
            difference method. 0.0001 (T) = 1 Oe is recommended as a starting
            point., by default 0.0001,
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
        rotation : ndarray[float64], optional
            A (3,3) orthogonal rotation matrix used to rotate momenta matrices.
            Note that the inverse matrix has to be given to rotate the
            reference frame instead., by default None
        slt : str, optional
            If given the results will be saved in a group of this name to .slt
            file with suffix: _susceptibility_tensor., by default None
        autotune : bool, optional
            If True the program will automatically try to choose the best
            number of threads (and therefore parallel processes), for the given
            number of CPUs, to be used during the calculation. Note that this
            process can take a significant amount of time, so start to use it
            with medium-sized calculations (e.g. for states_cutoff > 500 with
            a higher number of field values and number_of_points) where it
            becomes a necessity., by default False

        Returns
        -------
        ndarray[float64]
            The resulting array gives magnetic susceptibility (Van Vleck)
            tensors (or products with temperature) in cm^3 (or * K) and is in
            the form [fields, temperatures, 3x3 tensor] - the first dimension
            runs over field values, the second over temperatures, and the last
            two accomodate 3x3 tensors.

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
            a possitive integer
        SltInputError
            If the field step for the finite difference method is not
            a possitive real number.
        SltCompError
            If autotuning a number of processes and threads is unsuccessful.
        SltCompError
            If the calculation of magnetic susceptibility tensor is
            unsuccessful.
        SltFileError
            If the program is unable to correctly save results to .slt file.

        Note
        -----
        Here, (number_cpu // number_threads) parallel processes are used to
        distribute the workload over fields.size*(2*number_of_points+1) tasks.
        """
        if slt is not None:
            slt_group_name = f"{slt}_susceptibility_tensor"
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

        if autotune:
            if exp:
                num_to_parallel = fields.size
            else:
                num_to_parallel = (2 * number_of_points + 1) * fields.size

            try:
                number_cpu, number_threads = _auto_tune(
                    self._hdf5,
                    group,
                    num_to_parallel,
                    states_cutoff,
                    9,  # Size of 3x3 tensor in the inner loop
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
            chitht_tensor_array = _chitht_tensor(
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
        except Exception as exc:
            raise SltCompError(
                self._hdf5,
                exc,
                f"Failed to compute {chi_name} tensor from "
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
                    f"{slt}_{chi_file}ht_tensor",
                    f"Dataset containing {chi_name}_tensor Van Vleck"
                    " susceptibility tensor (H, T, 3, 3) calculated from"
                    f" group: {group}.",
                    f"Group({slt}) containing {chi_name}_tensor Van Vleck"
                    " susceptibility tensor calculated from group:"
                    f" {group}.",
                ] = chitht_tensor_array[:, :, :, :]
                self[
                    slt_group_name,
                    f"{slt}_fields",
                    "Dataset containing magnetic field H values used in"
                    f" simulation of {chi_name}_tensor Van Vleck"
                    f" susceptibility tensor from group: {group}.",
                ] = fields[:]
                self[
                    slt_group_name,
                    f"{slt}_temperatures",
                    "Dataset containing temperature T values used in"
                    f" simulation of {chi_name}_tensor Van Vleck"
                    f" susceptibility tensor from group: {group}.",
                ] = temperatures[:]
            except Exception as exc:
                raise SltFileError(
                    self._hdf5,
                    exc,
                    f"Failed to save {chi_name} tensor to "
                    + BLUE
                    + "Group "
                    + RESET
                    + '"'
                    + BLUE
                    + slt_group_name
                    + RESET
                    + '".',
                ) from None

        return chitht_tensor_array

    def calculate_chit_3d(
        self,
        group: str,
        temperatures: ndarray[float64],
        fields: ndarray[float64],
        spherical_grid: int,
        number_of_points: int = 1,
        delta_h: float = 0.0001,
        states_cutoff: int = 0,
        number_cpu: int = 0,
        number_threads: int = 1,
        exp: bool = False,
        T: bool = True,
        rotation: ndarray[float64] = None,
        slt: str = None,
        autotune: bool = False,
        _autotune_size: int = 2,
    ) -> ndarray[float64]:
        """
        Calculates 3D magnetic susceptibility over a spherical grid for a given
        list of temperature and field values.

        Parameters
        ----------
        group : str
            Name of a group containing results of relativistic ab initio
            calculations used for the computation of the 3D magnetic
            susceptibility.
        temperatures : ndarray[float64]
            ArrayLike structure (can be converted to numpy.NDArray) of
            temperature values (K) at which 3D magnetic susceptibility will be
            computed.
        fields : ndarray[float64]
            ArrayLike structure (can be converted to numpy.NDArray) of field
            values (T) at which 3D magnetic susceptibility will be computed.
        spherical_grid : int
            Controls the density of the angular grid for the 3D susceptibility
            calculation. A grid of dimension (spherical_grid*2*spherical_grid)
            for spherical angles theta [0, pi], and phi [0, 2*pi] will be used.
        number_of_points : int, optional
            Controls the number of points for numerical differentiation over
            the magnetic field values using the finite difference method with
            a symmetrical stencil. The total number of used points =
            (2 * num_of_opints + 1), therefore 1 is a minimum value to obtain
            the first derivative using 3 points - including the value at the
            point at which the derivative is taken. In this regard, the value 0
            triggers the experimentalist model for susceptibility.,
            by default 1
        delta_h : float64, optional
            Value of field step used for numerical differentiation using finite
            difference method. 0.0001 (T) = 1 Oe is recommended as a starting
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
        rotation : ndarray[float64], optional
            A (3,3) orthogonal rotation matrix used to rotate momenta matrices.
            Note that the inverse matrix has to be given to rotate the
            reference frame instead. It is useful here to orient your 3D plots
            more conveniently., by default None
        slt : str, optional
            If given the results will be saved in a group of this name to .slt
            file with suffix: _3d_magnetisation., by default None
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
            The resulting chi_3d_array gives magnetic susceptibility (or
            product with temperature) in cm^3 (or * K) and is in the form
            [coordinates, fields, temperatures, mesh, mesh] - the first
            dimension runs over coordinates (0-x, 1-y, 2-z), the second over
            field values, and the third over temperatures. The last two
            dimensions are in a form of meshgrids over theta and phi, ready
            for 3D plots as xyz.

        Raises
        ------
        SltSaveError
            If the name of the group already exists in the .slt file.
        SltInputError
            If input ArrayLike data cannot be converted to numpy.NDArrays.
        SltInputError
            If temperatures are not a one-diemsional array.
        SltInputError
            If fields are not a one-diemsional array.
        SltInputError
            If spherical_grid is not a positive integer.
        SltInputError
            If the  number of points for finite difference method is not
            a possitive integer.
        SltInputError
            If the field step for the finite difference method is not
            a possitive real number.
        SltCompError
            If autotuning a number of processes and threads is unsuccessful.
        SltCompError
            If the calculation of 3D magnetic susceptibility is unsuccessful.
        SltFileError
            If the program is unable to correctly save results to .slt file.

        Note
        -----
        Here, (number_cpu // number_threads) parallel processes are used to
        distribute the workload over len(fields)*(2*number_of_points + 1)
        *2*shperical_grid**2 tasks. Be aware that the resulting arrays and
        computations can quickly consume much memory (e.g. for calculation with
        100 field values 1-10 T, 300 temperatures 1-300 K, number_of_points=3,
        and spherical_grid = 60, the intermediate array (before numerical
        differentiation) will take 7*100*300*2*60*60*8 bytes = 12.096 GB).

        See Also
        --------
        slothpy.Compound.plot_3d, slothpy.Compound.interactive_plot_3d,
        slothpy.Compound.animate_3d
        """
        temperatures = array(temperatures, dtype=float64)
        fields = array(fields, dtype=float64)

        if slt is not None:
            slt_group_name = f"{slt}_3d_susceptibility"
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

        if (not isinstance(spherical_grid, int)) or spherical_grid <= 0:
            raise SltInputError(
                ValueError("Spherical grid has to be a positive integer.")
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

        if autotune:
            if exp:
                num_to_parallel = fields.size * 2 * spherical_grid**2
            else:
                num_to_parallel = (
                    (2 * number_of_points + 1)
                    * fields.size
                    * 2
                    * spherical_grid**2
                )

            try:
                number_cpu, number_threads = _auto_tune(
                    self._hdf5,
                    group,
                    num_to_parallel,
                    states_cutoff,
                    1,  # Single grid point in the inner loop
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
            chit_3d_array = _chit_3d(
                self._hdf5,
                group,
                temperatures,
                fields,
                spherical_grid,
                number_of_points,
                delta_h,
                states_cutoff,
                number_cpu,
                number_threads,
                exp,
                T,
                rotation,
            )
        except Exception as exc:
            raise SltCompError(
                self._hdf5,
                exc,
                "Failed to compute 3D magnetic susceptibility"
                f" {chi_name} from "
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
                    f"{slt}_{chi_file}_3d",
                    "Dataset containing 3D magnetic susceptibility"
                    f" {chi_name} as meshgird (0-x,1-y,2-z) arrays over"
                    " sphere ((xyz, field, temperature, meshgrid,"
                    f" meshgrid) calculated from group: {group}.",
                    f"Group({slt}) containing 3D magnetic susceptibility"
                    f" {chi_name} calculated from group: {group}.",
                ] = chit_3d_array[:, :, :, :, :]
                self[
                    slt_group_name,
                    f"{slt}_fields",
                    "Dataset containing magnetic field H values used in"
                    " simulation of 3D magnetic susceptibility"
                    f" {chi_name} from group: {group}.",
                ] = fields[:]
                self[
                    slt_group_name,
                    f"{slt}_temperatures",
                    "Dataset containing temperature T values used in"
                    " simulation of 3D magnetic susceptibility"
                    f" {chi_name} from group: {group}.",
                ] = temperatures[:]
            except Exception as exc:
                raise SltFileError(
                    self._hdf5,
                    exc,
                    f"Failed to save 3D magnetic susceptibility {chi_name} to "
                    + BLUE
                    + "Group "
                    + RESET
                    + '"'
                    + BLUE
                    + slt_group_name
                    + RESET
                    + '".',
                ) from None

        return chit_3d_array

    def calculate_helmholtz_energyth(
        self,
        group: str,
        fields: ndarray[float64],
        grid: ndarray[float64],
        temperatures: ndarray[float64],
        states_cutoff: int,
        number_cpu: int,
        number_threads: int,
        internal_energy: bool = False,
        slt: str = None,
        autotune: bool = False,
        _autotune_size: int = 2,
    ) -> ndarray[float64]:
        """
        Calculates powder-averaged or directional Helmholtz (or internal)
        energy for a given list of temperature and field values.

        Parameters
        ----------
        group : str
            Name of a group containing results of relativistic ab initio
            calculations used for the computation of the energy.
        fields : ndarray[float64]
            ArrayLike structure (can be converted to numpy.NDArray) of field
            values (T) at which energy will be computed.
        grid : ndarray[float64]
            If the grid is set to an integer from 0-11 then the prescribed
            Lebedev-Laikov grids over hemisphere will be used (see
            grids_over_hemisphere documentation), otherwise, user can provide
            an ArrayLike structure (can be converted to numpy.NDArray) with the
            convention: [[direction_x, direction_y, direction_z, weight],...]
            for powder-averaging. If one wants a calculation for a single,
            particular direction the list has to contain one entry like this:
            [[direction_x, direction_y, direction_z, 1.]]. Custom grids will be
            automatically normalized.
        temperatures : ndarray[float64]
            ArrayLike structure (can be converted to numpy.NDArray) of
            temeperature values (K) at which energy will be computed
        states_cutoff : int
            Number of states that will be taken into account for construction
            of Zeeman Hamiltonian. If set to zero, all available states from
            the file will be used., by default 0
        number_cpu : int
            Number of logical CPUs to be assigned to perform the calculation.
            If set to zero, all available CPUs will be used., by default 0
        number_threads : int
            Number of threads used in a multithreaded implementation of linear
            algebra libraries used during the calculation. Higher values
            benefit from the increasing size of matrices (states_cutoff) over
            the parallelization over CPUs., by default 1
        internal_energy : bool, optional
            Turns on the calculation of internal energy., by default False
        slt : str, optional
            If given the results will be saved in a group of this name to .slt
            file with suffix: _helmholtz_energy or _internal_energy.,
            by default None
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
            The resulting eth_array gives energy in cm-1 and is in the form
            [temperatures, fields] - the first dimension runs over temperature
            values, and the second over fields.

        Raises
        ------
        SltSaveError
            If the name of the group already exists in the .slt file.
        SltInputError
            If input ArrayLike data cannot be converted to numpy.NDArrays.
        SltInputError
            If fields are not a one-diemsional array
        SltInputError
            If temperatures are not a one-diemsional array
        SltCompError
            If autotuning a number of processes and threads is unsuccessful.
        SltCompError
            If the calculation of energy is unsuccessful.
        SltFileError
            If the program is unable to correctly save results to .slt file.

        Note
        -----
        Here, (number_cpu // number_threads) parallel processes are used to
        distribute the workload over the provided field values.

        See Also
        --------
        slothpy.Compound.plot_helmholtz_energyth
        slothpy.lebedev_laikov_grid : For the description of the prescribed
                                      Lebedev-Laikov grids.
        """
        if internal_energy:
            group_suffix = "_internal_energy"
            name = "internal"
        else:
            group_suffix = "_helmholtz_energy"
            name = "Helmholtz"

        if slt is not None:
            slt_group_name = f"{slt}{group_suffix}"
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
            grid = lebedev_laikov_grid(grid)
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
                    True,
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
            energyth_array = _helmholtz_energyth(
                self._hdf5,
                group,
                fields,
                grid,
                temperatures,
                states_cutoff,
                number_cpu,
                number_threads,
                internal_energy,
            )
        except Exception as exc:
            raise SltCompError(
                self._hdf5,
                exc,
                f"Failed to compute {name} energy from "
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
                    f"{slt}_eth",
                    f"Dataset containing E(T,H) {name} energy (T - rows,"
                    f" H - columns) calculated from group: {group}.",
                    f"Group({slt}) containing E(T,H) {name} energy"
                    f" calculated from group: {group}.",
                ] = energyth_array[:, :]
                self[
                    slt_group_name,
                    f"{slt}_fields",
                    "Dataset containing magnetic field H values used in"
                    f" simulation of E(T,H) {name} energy from group:"
                    f" {group}.",
                ] = fields[:]
                self[
                    slt_group_name,
                    f"{slt}_temperatures",
                    "Dataset containing temperature T values used in"
                    f" simulation of E(T,H) {name} energy from group:"
                    f" {group}.",
                ] = temperatures[:]
            except Exception as exc:
                raise SltFileError(
                    self._hdf5,
                    exc,
                    f"Failed to save {name} energy to "
                    + BLUE
                    + "Group "
                    + RESET
                    + '"'
                    + BLUE
                    + slt_group_name
                    + RESET
                    + '".',
                ) from None

        return energyth_array

    def calculate_helmholtz_energy_3d(
        self,
        group: str,
        fields: ndarray[float64],
        spherical_grid: int,
        temperatures: ndarray[float64],
        states_cutoff: int = 0,
        number_cpu: int = 0,
        number_threads: int = 1,
        internal_energy: bool = False,
        rotation: ndarray[float64] = None,
        slt: str = None,
        autotune: bool = False,
        _autotune_size: int = 2,
    ) -> ndarray[float64]:
        """
        Calculates 3D Helmholtz (or internal) energy over a spherical grid for
        a given list of temperature and field values.

        Parameters
        ----------
        group : str
            Name of a group containing results of relativistic ab initio
            calculations used for the computation of the 3D energy.
        fields : ndarray[float64]
            ArrayLike structure (can be converted to numpy.NDArray) of field
            values (T) at which 3D energy will be computed.
        spherical_grid : int
            Controls the density of the angular grid for the 3D magnetisation
            calculation. A grid of dimension (spherical_grid*2*spherical_grid)
            for spherical angles theta [0, pi], and phi [0, 2*pi] will be used.
        temperatures : ndarray[float64]
            ArrayLike structure (can be converted to numpy.NDArray) of
            temperature values (K) at which 3D energy will be computed.
        states_cutoff : int, optional
            Number of states that will be taken into account for construction
            of Zeeman Hamiltonian. If set to zero, all available states from
            the file will be used., by default 0,
        number_cpu : int, optional
            Number of logical CPUs to be assigned to perform the calculation.
            If set to zero, all available CPUs will be used., by default 0
        number_threads : int, optional
            Number of threads used in a multithreaded implementation of linear
            algebra libraries used during the calculation. Higher values
            benefit from the increasing size of matrices (states_cutoff) over
            the parallelization over CPUs., by default 1
        internal_energy : bool, optional
            Turns on the calculation of internal energy., by default False
        rotation : ndarray[float64], optional
            A (3,3) orthogonal rotation matrix used to rotate momenta matrices.
            Note that the inverse matrix has to be given to rotate the
            reference frame instead. It is useful here to orient your 3D plots
            more conveniently., by default None
        slt : str, optional
            If given the results will be saved in a group of this name to .slt
            file with suffix: _3d_helmholtz_energy or _3d_internal_energy.,
            by default None
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
            The resulting energy_3d_array gives energy in cm-1
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
            If the calculation of 3D energy is unsuccessful.
        SltFileError
            If the program is unable to correctly save results to .slt file.

        Note
        -----
        Here, (number_cpu // number_threads) parallel processes are used to
        distribute the workload over len(fields)*2*shperical_grid**2 tasks. Be
        aware that the resulting arrays and computations can quickly consume
        much memory (e.g. for a calculation with 100 field values 1-10 T, 300
        temperatures 1-300 K, and spherical_grid = 60, the resulting array will
        take 3*100*300*2*60*60*8 bytes = 5.184 GB).

        See Also
        --------
        slothpy.Compound.plot_3d, slothpy.Compound.interactive_plot_3d,
        slothpy.Compound.animate_3d
        """
        if internal_energy:
            group_suffix = "_3d_internal_energy"
            name = "internal"
        else:
            group_suffix = "_3d_helmholtz_energy"
            name = "Helmholtz"

        if slt is not None:
            slt_group_name = f"{slt}{group_suffix}"
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
            temperatures = array(temperatures, dtype=float64)
            fields = array(fields, dtype=float64)
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
                    1,  # Single grid point in the inner loop
                    temperatures.size,
                    number_cpu,
                    _autotune_size,
                    True,
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
            energy_3d_array = _helmholtz_energy_3d(
                self._hdf5,
                group,
                fields,
                spherical_grid,
                temperatures,
                states_cutoff,
                number_cpu,
                number_threads,
                internal_energy,
                rotation,
            )
        except Exception as exc:
            raise SltCompError(
                self._hdf5,
                exc,
                f"Failed to compute 3D {name} energy from "
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
                    f"{slt}_energy_3d",
                    "Dataset containing 3D {name} energy as meshgird"
                    " (0-x,1-y,2-z) arrays over sphere (xyz, field,"
                    " temperature, meshgrid, meshgrid) calculated from"
                    f" group: {group}.",
                    f"Group({slt}) containing 3D {name}_energy"
                    f" calculated from group: {group}.",
                ] = energy_3d_array[:, :, :, :, :]
                self[
                    slt_group_name,
                    f"{slt}_fields",
                    "Dataset containing magnetic field H values used in"
                    f" simulation of 3D {name} energy from group:"
                    f" {group}.",
                ] = fields[:]
                self[
                    slt_group_name,
                    f"{slt}_temperatures",
                    "Dataset containing temperature T values used in"
                    f" simulation of 3D {name} energy from group:"
                    f" {group}.",
                ] = temperatures[:]
            except Exception as exc:
                raise SltFileError(
                    self._hdf5,
                    exc,
                    f"Failed to save 3D {name} energy to "
                    + BLUE
                    + "Group "
                    + RESET
                    + '"'
                    + BLUE
                    + slt_group_name
                    + RESET
                    + '".',
                ) from None

        return energy_3d_array

    def calculate_zeeman_splitting(
        self,
        group: str,
        number_of_states: int,
        fields: ndarray[float64],
        grid: ndarray[float64],
        states_cutoff: int = 0,
        number_cpu: int = 0,
        number_threads: int = 1,
        average: bool = False,
        slt: str = None,
        autotune: bool = False,
        _autotune_size: int = 2,
    ) -> ndarray[float64]:
        """
        Calculates directional or powder-averaged Zeeman splitting for a given
        number of states and list of field values.

        Parameters
        ----------
        group : str
            Name of a group containing results of relativistic ab initio
            calculations used for the computation of the Zeeman splitting.
        number_of_states : int
            Number of states whose energy splitting will be given in the
            result array.
        fields : ndarray[float64]
            ArrayLike structure (can be converted to numpy.NDArray) of field
            values (T) at which Zeeman splitting will be computed.
        grid : ndarray[float64]
            If the grid is set to an integer from 0-11 then the prescribed
            Lebedev-Laikov grids over hemisphere will be used (see
            grids_over_hemisphere documentation) and powder-averaging will be
            turned on, otherwise, user can provide an ArrayLike structure (can
            be converted to numpy.NDArray) with the convention: [[direction_x,
            direction_y, direction_z, weight],...] with average = True for
            powder-averaging. If one wants a calculation for a list of
            particular directions the list has to follow the format:
            [[direction_x, direction_y, direction_z],...]. Custom grids will be
            automatically normalized.
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
        average : bool, optional
            Turns on powder-averaging using a list of directions and weights in
            the form of ArrayLike structure: [[direction_x, direction_y,
            direction_z, weight],...].
        slt : str, optional
            If given the results will be saved in a group of this name to .slt
            file with suffix: _zeeman_splitting., by default None
        autotune : bool, optional
            If True the program will automatically try to choose the best
            number of threads (and therefore parallel processes), for the given
            number of CPUs, to be used during the calculation. Note that this
            process can take a significant amount of time, so start to use it
            with medium-sized calculations (e.g. for states_cutoff > 300 with
            dense grids or a higher number of field values) where it becomes
            a necessity., by default Falsee

        Returns
        -------
        ndarray[float64]
            The resulting array gives Zeeman splitting of number_of_states
            energy levels in cm-1 for each direction (or average) in the form
            [orientations, fields, energies] - the first dimension
            runs over different orientations, the second over field values, and
            the last gives energy of number_of_states states.

        Raises
        ------
        SltSaveError
            If the name of the group already exists in the .slt file.
        SltInputError
            If input ArrayLike data cannot be converted to numpy.NDArrays.
        SltInputError
            If fields are not a one-diemsional array.
        SltInputError
            If number of states is not a positive integer less or equal to the
            states cutoff.
        SltCompError
            If autotuning a number of processes and threads is unsuccessful.
        SltCompError
            If the calculation of Zeeman splitting is unsuccessful.
        SltFileError
            If the program is unable to correctly save results to .slt file.

        See Also
        --------
        slothpy.Compound.plot_zeeman,
        slothpy.lebedev_laikov_grid : For the description of the prescribed
                                      Lebedev-Laikov grids.

        Note
        -----
        Here, (number_cpu // number_threads) parallel processes are used to
        distribute the workload over the provided field values.
        """
        if slt is not None:
            slt_group_name = f"{slt}_zeeman_splitting"
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
        except Exception as exc:
            raise SltInputError(exc) from None

        if fields.ndim != 1:
            raise SltInputError(
                ValueError("The list of fields has to be a 1D array.")
            ) from None

        try:
            max_states = self[f"{group}", "SOC"].shape[0]
        except Exception as exc1:
            try:
                max_states = self[f"{group}", "SOC_energies"].shape[0]
            except Exception as exc2:
                raise SltFileError(
                    self._hdf5,
                    exc2,
                    YELLOW
                    + f" {type(exc1).__name__}"
                    + RESET
                    + f": {str(exc1)} \nFailed to get SOC states from "
                    + BLUE
                    + "Group "
                    + RESET
                    + '"'
                    + BLUE
                    + group
                    + RESET
                    + '".',
                ) from None

        if (
            not isinstance(number_of_states, int)
            or number_of_states <= 0
            or number_of_states > max_states
        ):
            raise SltInputError(
                ValueError(
                    "The number of states has to be an integer less or equal"
                    " to the states cutoff."
                )
            ) from None

        if isinstance(grid, int):
            grid = lebedev_laikov_grid(grid)
            average = True
        elif average:
            grid = _normalize_grid_vectors(grid)
        else:
            grid = _normalize_orientations(grid)

        if autotune:
            try:
                number_cpu, number_threads = _auto_tune(
                    self._hdf5,
                    group,
                    fields.size,
                    states_cutoff,
                    grid.shape[0],
                    1,
                    number_cpu,
                    _autotune_size,
                    True,
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
            zeeman_array = _zeeman_splitting(
                self._hdf5,
                group,
                number_of_states,
                fields,
                grid,
                states_cutoff,
                number_cpu,
                number_threads,
                average,
            )
        except Exception as exc:
            raise SltCompError(
                self._hdf5,
                exc,
                "Failed to compute Zeeman splitting from "
                + BLUE
                + "Group "
                + RESET
                + '"'
                + BLUE
                + f"{group}"
                + RESET
                + '".',
            ) from None

        if average:
            name = "average "
        else:
            name = ""

        if slt is not None:
            try:
                self[
                    slt_group_name,
                    f"{slt}_zeeman",
                    f"Dataset containing {name}Zeeman splitting over grid"
                    " of directions with shape: (orientations, field,"
                    f" energy) calculated from group: {group}.",
                    f"Group({slt}) containing {name}Zeeman splitting"
                    f" calculated from group: {group}.",
                ] = zeeman_array[:, :, :]
                self[
                    slt_group_name,
                    f"{slt}_fields",
                    "Dataset containing magnetic field H values used in"
                    f" simulation of {name}Zeeman splitting from group:"
                    f" {group}.",
                ] = fields[:]
                if average:
                    self[
                        slt_group_name,
                        f"{slt}_orientations",
                        "Dataset containing magnetic field orientation"
                        " grid with weights used in simulation of"
                        f" {name}Zeeman splitting from group: {group}.",
                    ] = grid[:, :]
                else:
                    self[
                        slt_group_name,
                        f"{slt}_orientations",
                        "Dataset containing magnetic field orientations"
                        " used in simulation of"
                        f" {name}Zeeman splitting from group: {group}.",
                    ] = grid[:, :3]

            except Exception as exc:
                raise SltFileError(
                    self._hdf5,
                    exc,
                    "Failed to save Zeeman splitting to "
                    + BLUE
                    + "Group "
                    + RESET
                    + '"'
                    + BLUE
                    + slt_group_name
                    + RESET
                    + '".',
                ) from None

        return zeeman_array

    def zeeman_matrix(
        self,
        group: str,
        states_cutoff: int,
        fields: ndarray[float64],
        orientations: ndarray[float64],
        slt: str = None,
    ) -> ndarray[complex128]:
        """
        Calculates Zeeman matrices for a given list of magnetic fields and
        their orientations.

        Parameters
        ----------
        group : str
            Name of a group containing results of relativistic ab initio
            calculations used for the computation of the Zeeman matrices.
        states_cutoff : int
            Number of states that will be taken into account for construction
            of Zeeman Hamiltonian. If set to zero, all available states from
            the file will be used., by default 0
        fields : ndarray[float64]
            ArrayLike structure (can be converted to numpy.NDArray) of field
            values (T) for which Zeeman matrices will be computed.
        orientations : ndarray[float64]
            List (ArrayLike structure) of particular magnetic field directions
            for which Zeeman matrices will be constructed. The list has to
            follow the format: [[direction_x, direction_y, direction_z],...].
            The vectors will be automatically normalized.
        slt : str, optional
            If given the results will be saved in a group of this name to .slt
            file with suffix: _zeeman_matrix., by default None

        Returns
        -------
        ndarray[complex128]
            The resulting array gives Zeeman matrices for each field value and
            orientation in the form [fields, orientations, matrix, matrix] in
            atomic units a.u. (Hartree).

        Raises
        ------
        SltSaveError
            If the name of the group already exists in the .slt file.
        SltInputError
            If input ArrayLike data cannot be converted to numpy.NDArrays.
        SltInputError
            If fields are not a one-diemsional array.
        SltCompError
            If the calculation of Zeeman matrices is unsuccessful.
        SltFileError
            If the program is unable to correctly save results to .slt file.
        """
        if slt is not None:
            slt_group_name = f"{slt}_zeeman_matrix"
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
        except Exception as exc:
            raise SltInputError(exc) from None

        if fields.ndim != 1:
            raise SltInputError(
                ValueError("The list of fields has to be a 1D array.")
            ) from None

        orientations = _normalize_orientations(orientations)

        try:
            zeeman_matrix_array = _get_zeeman_matrix(
                self._hdf5, group, states_cutoff, fields, orientations
            )
        except Exception as exc:
            raise SltCompError(
                self._hdf5,
                exc,
                "Failed to calculate Zeeman matrix from "
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
                    f"{slt}_matrix",
                    "Dataset containing Zeeman matrices calculated from"
                    f" group: {group} in the form [fields, orientations,"
                    " matrix, matrix].",
                    f"Group({slt}) containing Zeeman matrices calculated"
                    f" from group: {group}.",
                ] = zeeman_matrix_array[:, :, :, :]
                self[
                    slt_group_name,
                    f"{slt}_fields",
                    "Dataset containing magnetic field H values used in"
                    " simulation of Zeeman matrices from group:"
                    f" {group}.",
                ] = fields[:]
                self[
                    slt_group_name,
                    f"{slt}_orientations",
                    "Dataset containing magnetic field orientations"
                    " used in simulation of"
                    f" Zeeman matrices from group: {group}.",
                ] = orientations[:, :]

            except Exception as exc:
                raise SltFileError(
                    self._hdf5,
                    exc,
                    "Failed to save Zeeman matrix to "
                    + BLUE
                    + "Group "
                    + RESET
                    + '"'
                    + BLUE
                    + slt_group_name
                    + RESET
                    + '".',
                ) from None

        return zeeman_matrix_array

    def soc_energies_cm_1(
        self, group: str, number_of_states: int = 0, slt: str = None
    ) -> ndarray[float64]:
        """
        Returns energies for the given number of first spin-orbit
        states in cm-1.

        Parameters
        ----------
        group : str
            Name of a group containing results of relativistic ab initio
            calculations.
        number_of_states : int, optional
            Number of states whose energy will be returned. If set to zero, all
            available states will be inculded., by default 0
        slt : str, optional
            If given the results will be saved in a group of this name to .slt
            file with suffix: _soc_energies., by default None

        Returns
        -------
        ndarray[float64]
            The resulting array is one-dimensional and contains the energy of
            first number_of_states states in cm-1.

        Raises
        ------
        SltSaveError
            If the name of the group already exists in the .slt file.
        SltReadError
            If the program is unable to get SOC energies from the .slt file.
        SltFileError
            If the program is unable to correctly save results to .slt file.

        See Also
        --------
        slothpy.exporting.table_energy_and_g
        """
        if slt is not None:
            slt_group_name = f"{slt}_soc_energies"
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
            soc_energies_array = _get_soc_energies_cm_1(
                self._hdf5, group, number_of_states
            )
        except Exception as exc:
            raise SltReadError(
                self._hdf5,
                exc,
                "Failed to read SOC energies from "
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
                    f"{slt}_energies",
                    "Dataset containing SOC (Spin-Orbit Coupling) energies"
                    f" calculated from group: {group}.",
                    f"Group({slt}) containing SOC (Spin-Orbit Coupling)"
                    f" energies calculated from group: {group}.",
                ] = soc_energies_array
            except Exception as exc:
                raise SltFileError(
                    self._hdf5,
                    exc,
                    "Failed to save SOC energies to "
                    + BLUE
                    + "Group "
                    + RESET
                    + '"'
                    + BLUE
                    + slt_group_name
                    + RESET
                    + '".',
                ) from None

        return soc_energies_array

    def states_magnetic_momenta(
        self,
        group: str,
        states: Union[int, ndarray[int]] = 0,
        rotation: ndarray[float64] = None,
        slt: str = None,
    ) -> ndarray[float64]:
        """
        Calculates magnetic momenta of a given list (or number) of SOC states.

        Parameters
        ----------
        group : str
            Name of a group containing results of relativistic ab initio
            calculations used for the computation of the magnetic momenta.
        states : Union[int, ndarray[int]], optional
            ArrayLike structure (can be converted to numpy.NDArray) of
            states indexes for which magnetic momenta will be calculated. If
            set to an integer it acts as a states cutoff (first n states will
            be given). For all available states set it to zero., by default 0
        rotation : ndarray[float64], optional
            A (3,3) orthogonal rotation matrix used to rotate momenta matrices.
            Note that the inverse matrix has to be given to rotate the
            reference frame instead., by default None
        slt : str, optional
            If given the results will be saved in a group of this name to .slt
            file with suffix: _states_magnetic_momenta., by default None

        Returns
        -------
        ndarray[float64]
            The resulting array is one-dimensional and contains the magnetic
            momenta corresponding to the given states indexes in atomic units.

        Raises
        ------
        SltSaveError
            If the name of the group already exists in the .slt file.
        SltInputError
            If input ArrayLike data cannot be converted to numpy.NDArrays.
        SltCompError
            If the calculation of magnetic momenta is unsuccessful.
        SltFileError
            If the program is unable to correctly save results to .slt file.
        """
        if slt is not None:
            slt_group_name = f"{slt}_states_magnetic_momenta"
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

        if not isinstance(states, int):
            try:
                states = array(states, dtype=int64)
            except Exception as exc:
                raise SltInputError(exc) from None

        try:
            magnetic_momenta_array = _get_states_magnetic_momenta(
                self._hdf5, group, states, rotation
            )
        except Exception as exc:
            raise SltCompError(
                self._hdf5,
                exc,
                "Failed to compute states magnetic momenta from "
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
                    f"{slt}_magnetic_momenta",
                    "Dataset containing states magnetic momenta"
                    f" (0-x,1-y,2-z) calculated from group: {group}.",
                    f"Group({slt}) containing states magnetic momenta"
                    f" calculated from group: {group}.",
                ] = magnetic_momenta_array
                self[
                    slt_group_name,
                    f"{slt}_states",
                    "Dataset containing indexes of states (or states"
                    " cutoff) used in simulation of magnetic momenta from"
                    f" group: {group}.",
                ] = array(states)

            except Exception as exc:
                raise SltFileError(
                    self._hdf5,
                    exc,
                    "Failed to save states magnetic momenta to "
                    + BLUE
                    + "Group "
                    + RESET
                    + '"'
                    + BLUE
                    + slt_group_name
                    + RESET
                    + '".',
                ) from None

        return magnetic_momenta_array

    def states_total_angular_momenta(
        self,
        group: str,
        states: Union[int, ndarray[int]] = 0,
        rotation: ndarray[float64] = None,
        slt: str = None,
    ) -> ndarray[float64]:
        """
        Calculates total angular momenta of a given list (or number) of SOC
        states.

        Parameters
        ----------
        group : str
            Name of a group containing results of relativistic ab initio
            calculations used for the computation of the magnetic momenta.
        states : Union[int, ndarray[int]], optional
            ArrayLike structure (can be converted to numpy.NDArray) of
            states indexes for which total angular momenta will be calculated.
            If set to an integer it acts as a states cutoff (first n states
            will be given). For all available states set it to zero.
            , by default 0
        rotation : ndarray[float64], optional
            A (3,3) orthogonal rotation matrix used to rotate momenta matrices.
            Note that the inverse matrix has to be given to rotate the
            reference frame instead., by default None
        slt : str, optional
            If given the results will be saved in a group of this name to .slt
            file with suffix: _states_total_angular_momenta., by default None

        Returns
        -------
        ndarray[float64]
            The resulting array is one-dimensional and contains the total
            angular momenta corresponding to the given states indexes in atomic
            units.

        Raises
        ------
        SltSaveError
            If the name of the group already exists in the .slt file.
        SltInputError
            If input ArrayLike data cannot be converted to numpy.NDArrays.
        SltCompError
            If the calculation of total angular momenta is unsuccessful.
        SltFileError
            If the program is unable to correctly save results to .slt file.
        """
        if slt is not None:
            slt_group_name = f"{slt}_states_total_angular_momenta"
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

        if not isinstance(states, int):
            try:
                states = array(states, dtype=int64)
            except Exception as exc:
                raise SltInputError(exc) from None

        try:
            total_angular_momenta_array = _get_states_total_angular_momenta(
                self._hdf5, group, states, rotation
            )
        except Exception as exc:
            raise SltCompError(
                self._hdf5,
                exc,
                "Failed to compute states total angular momenta from "
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
                    f"{slt}_total_angular_momenta",
                    "Dataset containing states total angular momenta"
                    f" (0-x,1-y,2-z) calculated from group: {group}.",
                    f"Group({slt}) containing states total angular momenta"
                    f" calculated from group: {group}.",
                ] = total_angular_momenta_array
                self[
                    slt_group_name,
                    f"{slt}_states",
                    "Dataset containing indexes of states (or states"
                    " cutoff) used in simulation of total angular momenta"
                    f" from group: {group}.",
                ] = array(states)

            except Exception as exc:
                raise SltFileError(
                    self._hdf5,
                    exc,
                    "Failed to save states total angular momenta to "
                    + BLUE
                    + "Group "
                    + RESET
                    + '"'
                    + BLUE
                    + slt_group_name
                    + RESET
                    + '".',
                ) from None

        return total_angular_momenta_array

    def magnetic_momenta_matrix(
        self,
        group: str,
        states_cutoff: ndarray = 0,
        rotation: ndarray[float64] = None,
        slt: str = None,
    ) -> ndarray[complex128]:
        """
        Calculates magnetic momenta matrix for a given number of SOC states.

        Parameters
        ----------
        group : str
            Name of a group containing results of relativistic ab initio
            calculations used for the computation of the magnetic momenta
            matrix.
        states_cutoff : ndarray, optional
            Number of states that will be taken into account for construction
            of the magnetic momenta matrix. If set to zero, all available
            states from the file will be included., by default 0
        rotation : ndarray[float64], optional
            A (3,3) orthogonal rotation matrix used to rotate momenta matrices.
            Note that the inverse matrix has to be given to rotate the
            reference frame instead., by default None
        slt : str, optional
            If given the results will be saved in a group of this name to .slt
            file with suffix: _magnetic_momenta_matrix., by default None

        Returns
        -------
        ndarray[complex128]
            The resulting magnetic_momenta_matrix_array gives magnetic momenta
            in atomic units and is in the form [coordinates, matrix, matrix]
            - the first dimension runs over coordinates (0-x, 1-y, 2-z).

        Raises
        ------
        SltSaveError
            If the name of the group already exists in the .slt file.
        SltCompError
            If the calculation of magetic momenta matrix is unsuccessful.
        SltFileError
            If the program is unable to correctly save results to .slt file.
        """
        if slt is not None:
            slt_group_name = f"{slt}_magnetic_momenta_matrix"
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
            magnetic_momenta_matrix_array = _get_magnetic_momenta_matrix(
                self._hdf5, group, states_cutoff, rotation
            )
        except Exception as exc:
            raise SltCompError(
                self._hdf5,
                exc,
                "Failed to compute magnetic momenta matrix from "
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
                    f"{slt}_magnetic_momenta_matrix",
                    "Dataset containing magnetic momenta matrix"
                    f" (0-x, 1-y, 2-z) calculated from group: {group}.",
                    f"Group {group} containing magnetic momenta"
                    f" matrix calculated from group: {group}.",
                ] = magnetic_momenta_matrix_array[:, :]
            except Exception as exc:
                raise SltFileError(
                    self._hdf5,
                    exc,
                    "Failed to save states magnetic momenta to "
                    + BLUE
                    + "Group "
                    + RESET
                    + '"'
                    + BLUE
                    + slt_group_name
                    + RESET
                    + '".',
                ) from None

        return magnetic_momenta_matrix_array

    def total_angular_momenta_matrix(
        self,
        group: str,
        states_cutoff: int = 0,
        rotation: ndarray[float64] = None,
        slt: str = None,
    ) -> ndarray[complex128]:
        """
        Calculates total angular momenta matrix for a given number of SOC
        states.

        Parameters
        ----------
        group : str
            Name of a group containing results of relativistic ab initio
            calculations used for the computation of the total angular momenta
            matrix.
        states_cutoff : ndarray, optional
            Number of states that will be taken into account for construction
            of the total angular momenta matrix. If set to zero, all available
            states from the file will be included., by default 0
        rotation : ndarray[float64], optional
            A (3,3) orthogonal rotation matrix used to rotate momenta matrices.
            Note that the inverse matrix has to be given to rotate the
            reference frame instead., by default None
        slt : str, optional
            If given the results will be saved in a group of this name to .slt
            file with suffix: _total angular_momenta_matrix., by default None

        Returns
        -------
        ndarray[complex128]
            The resulting total_angular_momenta_matrix_array gives total
            angular momenta in atomic units and is in the form [coordinates,
            matrix, matrix] - the first dimension runs over coordinates
            (0-x, 1-y, 2-z).

        Raises
        ------
        SltSaveError
            If the name of the group already exists in the .slt file.
        SltCompError
            If the calculation of total angular momenta matrix is unsuccessful.
        SltFileError
            If the program is unable to correctly save results to .slt file.
        """
        if slt is not None:
            slt_group_name = f"{slt}_total_angular_momenta_matrix"
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
            total_angular_momenta_matrix_array = (
                _get_total_angular_momneta_matrix(
                    self._hdf5, group, states_cutoff, rotation
                )
            )
        except Exception as exc:
            raise SltCompError(
                self._hdf5,
                exc,
                "Failed to compute total angular momenta matrix from "
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
                    f"{slt}_total_angular_momenta_matrix",
                    "Dataset containing total angular momenta matrix"
                    f" (0-x, 1-y, 2-z) calculated from group: {group}.",
                    f"Group {group} containing total angular momenta"
                    f" matrix calculated from group: {group}.",
                ] = total_angular_momenta_matrix_array[:, :]
            except Exception as exc:
                raise SltFileError(
                    self._hdf5,
                    exc,
                    "Failed to save states total angular momenta to "
                    + BLUE
                    + "Group "
                    + RESET
                    + '"'
                    + BLUE
                    + slt_group_name
                    + RESET
                    + '".',
                ) from None

        return total_angular_momenta_matrix_array

    def matrix_decomposition_in_z_pseudo_spin_basis(
        self,
        group: str,
        matrix: Union["soc", "zeeman"],
        pseudo_kind: Union["magnetic", "total_angular"],
        start_state: int = 0,
        stop_state: int = 0,
        rotation: ndarray[float64] = None,
        field: float64 = None,
        orientation: ndarray[float64] = None,
        slt: str = None,
    ) -> ndarray[float64]:
        """
        Calculates decomposition of a given matrix in "z" pseudo-spin basis.

        Parameters
        ----------
        group : str
            Name of a group containing results of relativistic ab initio
            calculations used for the construction of the matrix.
        matrix : Union["soc", "zeeman"]
            Type of a matrix to be decomposed. Two options available: "soc" or
            "zeeman".
        pseudo_kind : Union["magnetic", "total_angular"]
            Kind of a pseudo-spin basis. Two options available: "magnetic" or
            "total_angular" for the decomposition in a particular basis.
        start_state : int, optional
            Number of the first SOC state to be included., by default 0
        stop_state : int, optional
            Number of the last SOC state to be included. If both start and stop
            are set to zero all available states from the file will be used.
            , by default 0
        rotation : ndarray[float64], optional
            A (3,3) orthogonal rotation matrix used to rotate momenta matrices.
            Note that the inverse matrix has to be given to rotate the
            reference frame instead., by default None
        field : float64, optional
            If matrix type = "zeeman" it controls a magnetic field value at
            which Zeeman matrix will be computed., by default None
        orientation : ndarray[float64], optional
            If matrix type = "zeeman" it controls the orientation of the
            magnetic field and has to be in the form [direction_x, direction_y,
            direction_z] and be an ArrayLike structure (can be converted to
            numpy.NDArray)., by default None
        slt : str, optional
            If given the results will be saved in a group of this name to .slt
            file with suffix: _magnetic/total_angular_decomposition.
            , by default None

        Returns
        -------
        ndarray[float64]
            The resulting array gives decomposition in % where rows are
            SOC/Zeeman states and columns are associated with pseudo spin basis
            (from -Sz to Sz).

        Raises
        ------
        SltSaveError
            If the name of the group already exists in the .slt file.
        SltCompError
            If the decomposition of the matrix is unsuccessful.
        SltFileError
            If the program is unable to correctly save results to .slt file.
        """
        if slt is not None:
            slt_group_name = f"{slt}_{pseudo_kind}_decomposition"
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
            if orientation is not None:
                orientation = _normalize_orientation(orientation)
            decomposition = _get_decomposition_in_z_pseudo_spin_basis(
                self._hdf5,
                group,
                matrix,
                pseudo_kind,
                start_state,
                stop_state,
                rotation,
                field,
                orientation,
            )
        except Exception as exc:
            raise SltCompError(
                self._hdf5,
                exc,
                f"Failed to decompose {matrix} matrix from "
                + BLUE
                + "Group "
                + RESET
                + '"'
                + BLUE
                + f"{group}"
                + RESET
                + f'". in {pseudo_kind} basis.',
            ) from None

        if slt is not None:
            dim = (decomposition.shape[1] - 1) / 2
            try:
                self[
                    slt_group_name,
                    f"{slt}_{pseudo_kind}_decomposition",
                    "Dataset containing decomposition (rows - SO-states,"
                    f' columns - basis) in "z" {pseudo_kind} momentum'
                    f" basis of {matrix} matrix from group: {group}.",
                    f'Group({slt}) containing decomposition in "z"'
                    f" {pseudo_kind} basis of {matrix} matrix calculated"
                    f" from group: {group}.",
                ] = decomposition[:, :]
                self[
                    slt_group_name,
                    f"{slt}_pseudo_spin_states",
                    "Dataset containing Sz pseudo-spin states"
                    " corresponding to the decomposition of"
                    f" {matrix} matrix from group: {group}.",
                ] = arange(-dim, dim + 1, step=1, dtype=float64)
            except Exception as exc:
                raise SltFileError(
                    self._hdf5,
                    exc,
                    f"Failed to save {pseudo_kind} decomposition of"
                    f" {matrix} matrix "
                    + BLUE
                    + "Group "
                    + RESET
                    + '"'
                    + BLUE
                    + slt_group_name
                    + RESET
                    + '".',
                ) from None

        return decomposition

    def soc_crystal_field_parameters(
        self,
        group: str,
        start_state: int,
        stop_state: int,
        order: int,
        pseudo_kind: Union["magnetic", "total_angular"],
        even_order: bool = True,
        complex: bool = False,
        rotation: ndarray[float64] = None,
        slt: str = None,
    ) -> list:
        """
        Calculates ITO decomposition (CFPs) of SOC matrix.

        Parameters
        ----------
        group : str
            Name of a group containing results of relativistic ab initio
            calculations used for obtaining the SOC matrix.
        start_state : int
            Number of the first SOC state to be included.
        stop_state : int
            Number of the last SOC state to be included. If both start and stop
            are set to zero all available states from the file will be used.
        order : int
            Order of the highest ITO (CFP) to be included in the decomposition.
        pseudo_kind : Union["magnetic", "total_angular"]
            Kind of a pseudo-spin basis. Two options available: "magnetic" or
            "total_angular" for the decomposition in a particular basis.
        even_order : bool, optional
            If True, only even order ITOs (CFPs) will be included in the
            decomposition., by default True
        complex : bool, optional
            If True, instead of real ITOs (CFPs) complex ones will be given.,
            by default False
        rotation : ndarray[float64], optional
            A (3,3) orthogonal rotation matrix used to rotate momenta matrices.
            Note that the inverse matrix has to be given to rotate the
            reference frame instead., by default None
        slt : str, optional
            If given the results will be saved in a group of this name to .slt
            file with suffix: _soc_ito_decomposition., by default None

        Returns
        -------
        list
            The resulting list gives CFP - B_k_q (ITO) in the form [k,q,B_k_q].

        Raises
        ------
        SltSaveError
            If the name of the group already exists in the .slt file.
        SltReadError
            If the program is unable to read SOC matrix from the file.
        SltInputError
            If the order exceeds 2S pseudo-spin value.
        SltCompError
            If the ITO decomposition of the matrix is unsuccessful.
        SltFileError
            If the program is unable to correctly save results to .slt file.

        Note
        ----
        The decomposition is obtained using a projection method described in
        [1] (eq. 41) employing ITOs defined in [2] (eq. 29) with
        a normalization factor from eq. 17.

        References
        ----------
        .. [1] L. F. Chibotaru and L. Ungur
            "Ab initio calculation of anisotropic magnetic properties of
            complexes. I. Unique definition of pseudospin Hamiltonians and
            their derivation"
            J. Chem. Phys. 137, 064112 (2012).
        .. [2] I. D. Ryabov
            "On the Generation of Operator Equivalents and the Calculation
            of Their Matrix Elements"
            J. Magn. Reson. 140, 141–145 (1999).
        """
        if slt is not None:
            slt_group_name = f"{slt}_soc_ito_decomposition"
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
            soc_matrix = _get_soc_matrix_in_z_pseudo_spin_basis(
                self._hdf5,
                group,
                start_state,
                stop_state,
                pseudo_kind,
                rotation,
            )
        except Exception as exc:
            raise SltReadError(
                self._hdf5,
                exc,
                "Failed to read SOC matrix from "
                + BLUE
                + "Group "
                + RESET
                + '"'
                + BLUE
                + f"{group}"
                + RESET
                + f'". in {pseudo_kind} basis.',
            ) from None

        dim = (soc_matrix.shape[1] - 1) / 2

        if not isinstance(order, int) or order < 0 or order > 2 * dim:
            raise SltInputError(
                ValueError(
                    "Order of ITO parameters has to be a positive integer or"
                    " it exceeds 2S. Set it less or equal."
                )
            )

        try:
            if complex:
                cfp = _ito_complex_decomp_matrix(soc_matrix, order, even_order)
            else:
                cfp = _ito_real_decomp_matrix(soc_matrix, order, even_order)
        except Exception as exc:
            raise SltCompError(
                self._hdf5,
                exc,
                "Failed to ITO decompose SOC matrix from "
                + BLUE
                + "Group "
                + RESET
                + '"'
                + BLUE
                + f"{group}"
                + RESET
                + f'". in {pseudo_kind} basis.',
            ) from None

        cfp_return = cfp

        if slt is not None:
            cfp = array(cfp)

            try:
                self[
                    slt_group_name,
                    f"{slt}_ito_parameters",
                    'Dataset containing ITO decomposition in "z"'
                    " pseudo-spin basis of SOC matrix from group:"
                    f" {group}.",
                    f'Group({slt}) containing ITO decomposition in "z"'
                    " pseudo-spin basis of SOC matrix calculated from"
                    f" group: {group}.",
                ] = cfp[:, :]
                self[
                    slt_group_name,
                    f"{slt}_pseudo_spin_states",
                    "Dataset containing S pseudo-spin number"
                    " corresponding to the decomposition of SOC matrix"
                    f" from group: {group}.",
                ] = array([dim])
            except Exception as exc:
                raise SltFileError(
                    self._hdf5,
                    exc,
                    "Failed to save ITO decomposition of SOC matrix to "
                    + BLUE
                    + "Group "
                    + RESET
                    + '"'
                    + BLUE
                    + slt_group_name
                    + RESET
                    + '".',
                ) from None

        return cfp_return

    def zeeman_matrix_ito_decpomosition(
        self,
        group: str,
        start_state: int,
        stop_state: int,
        field: float64,
        orientation: ndarray[float64],
        order: int,
        pseudo_kind: Union["magnetic", "total_angular"],
        complex: bool = False,
        rotation: ndarray[float64] = None,
        slt: str = None,
    ) -> list:
        """
        Calculates ITO decomposition of Zeeman matrix.

        Parameters
        ----------
        group : str
            Name of a group containing results of relativistic ab initio
            calculations used for obtaining the Zeeman matrix.
        start_state : int
            Number of the first Zeeman state to be included.
        stop_state : int
            Number of the last Zeeman state to be included. If both start and
            stop are set to zero all available states from the file will be
            used.
        field : float64
            Magnetic field value at which Zeeman matrix will be computed.
        orientation : ndarray[float64]
            Orientation of the magnetic field in the form of an ArrayLike
            structure (can be converted to numpy.NDArray) [direction_x,
            direction_y, direction_z].
        order : int
            Order of the highest ITO (CFP) to be included in the decomposition.
        pseudo_kind : Union["magnetic", "total_angular"]
            Kind of a pseudo-spin basis. Two options available: "magnetic" or
            "total_angular" for the decomposition in a particular basis.
        complex : bool, optional
            If True, instead of real ITOs (CFPs) complex ones will be given.,
            by default False
        rotation : ndarray[float64], optional
            A (3,3) orthogonal rotation matrix used to rotate momenta matrices.
            Note that the inverse matrix has to be given to rotate the
            reference frame instead., by default None
        slt : str, optional
            If given the results will be saved in a group of this name to .slt
            file with suffix: _zeeman_ito_decomposition., by default None

        Returns
        -------
        list
            The resulting list gives ITOs - B_k_q in the form [k,q,B_k_q]

        Raises
        ------
        SltSaveError
            If the name of the group already exists in the .slt file.
        SltCompError
            If the program is unable to calculate Zeeman matrix from the file.
        SltInputError
            If the order exceeds 2S pseudo-spin value
        SltCompError
            If the ITO decomposition of the matrix is unsuccessful
        SltFileError
            If the program is unable to correctly save results to .slt file.

        Note
        ----
        The decomposition is obtained using a projection method described in
        [1] (eq. 41) employing ITOs defined in [2] (eq. 29) with
        a normalization factor from eq. 17.

        References
        ----------
        .. [1] L. F. Chibotaru and L. Ungur
            "Ab initio calculation of anisotropic magnetic properties of
            complexes. I. Unique definition of pseudospin Hamiltonians and
            their derivation"
            J. Chem. Phys. 137, 064112 (2012).
        .. [2] I. D. Ryabov
            "On the Generation of Operator Equivalents and the Calculation
            of Their Matrix Elements"
            J. Magn. Reson. 140, 141–145 (1999).
        """
        if slt is not None:
            slt_group_name = f"{slt}_zeeman_ito_decomposition"
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

        orientation = _normalize_orientation(orientation)

        try:
            zeeman_matrix = _get_zeeman_matrix_in_z_pseudo_spin_basis(
                self._hdf5,
                group,
                field,
                orientation,
                start_state,
                stop_state,
                pseudo_kind,
                rotation,
            )
        except Exception as exc:
            raise SltCompError(
                self._hdf5,
                exc,
                "Failed to calculate Zeeman matrix from "
                + BLUE
                + "Group "
                + RESET
                + '"'
                + BLUE
                + f"{group}"
                + RESET
                + f'". in {pseudo_kind} basis.',
            ) from None

        dim = (zeeman_matrix.shape[1] - 1) / 2

        if not isinstance(order, int) or order < 0 or order > 2 * dim:
            raise SltInputError(
                ValueError(
                    "Order of ITO parameters has to be a positive integer or"
                    " it exceeds 2S. Set it less or equal."
                )
            )

        try:
            if complex:
                ito = _ito_complex_decomp_matrix(zeeman_matrix, order)
            else:
                ito = _ito_real_decomp_matrix(zeeman_matrix, order)
        except Exception as exc:
            raise SltCompError(
                self._hdf5,
                exc,
                "Failed to ITO decompose Zeeman matrix from "
                + BLUE
                + "Group "
                + RESET
                + '"'
                + BLUE
                + f"{group}"
                + RESET
                + f'". in {pseudo_kind} basis.',
            ) from None

        ito_return = ito

        if slt is not None:
            ito = array(ito)

            try:
                self[
                    slt_group_name,
                    f"{slt}_ito_parameters",
                    'Dataset containing ITO decomposition in "z"'
                    " pseudo-spin basis of Zeeman matrix from group:"
                    f" {group}.",
                    f'Group({slt}) containing ITO decomposition in "z"'
                    " pseudo-spin basis of Zeeman matrix calculated from"
                    f" group: {group}.",
                ] = ito[:, :]
                self[
                    slt_group_name,
                    f"{slt}_pseudo_spin_states",
                    "Dataset containing S pseudo-spin number"
                    " corresponding to the decomposition of Zeeman matrix"
                    f" from group: {group}.",
                ] = array([dim])
            except Exception as exc:
                raise SltFileError(
                    self._hdf5,
                    exc,
                    "Failed to save ITO decomposition of Zeeman matrix to "
                    + BLUE
                    + "Group "
                    + RESET
                    + '"'
                    + BLUE
                    + slt_group_name
                    + RESET
                    + '".',
                ) from None

        return ito_return

    def matrix_from_ito(
        self,
        full_group_name: str,
        complex: bool,
        dataset_name: str = None,
        pseudo_spin: float64 = None,
        slt: str = None,
    ) -> ndarray[complex128]:
        """
        Calculates matrix from a given ITO decomposition.

        Parameters
        ----------
        full_group_name : str
            Full name of a group containing ITO decomposition.
        complex : bool
            Determines the type of ITOs in the dataset. If True, instead of
            real ITOs complex ones will be used., by default False
        dataset_name : str, optional
            A custom name for a user-created dataset within the group that
            contains list of B_k_q parameters in the form [k,q,B_k_q].,
            by default None
        pseudo_spin : float64, optional
            Pseudo spin S value for the user-defined dataset., by default None
        slt : str, optional
            If given the results will be saved in a group of this name to .slt
            file with suffix: _matrix_from_ito., by default None

        Returns
        -------
        ndarray[complex128]
            Matrix reconstructed from a given ITO list.

        Raises
        ------
        SltSaveError
            If the name of the group already exists in the .slt file.
        SltCompError
            If the calculation of the matrix from ITOs is unsuccessful.
        SltFileError
            If the program is unable to correctly save results to .slt file.

        Note
        ----
        ITOs defined in [2] (eq. 29) with a normalization factor from eq. 17
        are used.

        References
        ----------
        .. [1] I. D. Ryabov
            "On the Generation of Operator Equivalents and the Calculation
            of Their Matrix Elements"
            J. Magn. Reson. 140, 141–145 (1999).
        """
        if slt is not None:
            slt_group_name = f"{slt}_matrix_from_ito"
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
            if (
                (dataset_name is not None)
                and (pseudo_spin is not None)
                and isinstance(pseudo_spin, int)
                and pseudo_spin > 0
            ):
                J = pseudo_spin
                coefficients = self[f"{full_group_name}", f"{dataset_name}"]
                if complex:
                    matrix = _matrix_from_ito_complex(J, coefficients)
                else:
                    matrix = _matrix_from_ito_real(J, coefficients)

            else:
                if full_group_name.endswith("_zeeman_ito_decomposition"):
                    dataset_name = full_group_name[
                        : -len("_zeeman_ito_decomposition")
                    ]
                elif full_group_name.endswith("_soc_ito_decomposition"):
                    dataset_name = full_group_name[
                        : -len("_soc_ito_decomposition")
                    ]
                else:
                    raise NameError(
                        f"Invalid group name: {full_group_name}. It must end"
                        " with _soc_ito_decomposition or"
                        " _zeeman_ito_decomposition."
                    )
                J = self[
                    f"{full_group_name}",
                    f"{dataset_name}_pseudo_spin_states",
                ]
                coefficients = self[
                    f"{full_group_name}",
                    f"{dataset_name}_ito_parameters",
                ]
                if complex:
                    matrix = _matrix_from_ito_complex(J[0], coefficients)
                else:
                    matrix = _matrix_from_ito_real(J[0], coefficients)
        except Exception as exc:
            raise SltCompError(
                self._hdf5,
                exc,
                "Failed to compute matrix from ITOs from "
                + BLUE
                + "Group "
                + RESET
                + '"'
                + BLUE
                + f"{full_group_name}"
                + RESET
                + '".',
            ) from None

        if slt is not None:
            try:
                self[
                    slt_group_name,
                    f"{slt}_matrix",
                    "Dataset containing matrix from ITOs calculated from"
                    f" group: {full_group_name}.",
                    f"Group({slt}) containing matrix from ITO calculated"
                    f" from group: {full_group_name}.",
                ] = matrix[:, :]
            except Exception as exc:
                raise SltFileError(
                    self._hdf5,
                    exc,
                    "Failed to save matrix from ITOs to "
                    + BLUE
                    + "Group "
                    + RESET
                    + '"'
                    + BLUE
                    + slt_group_name
                    + RESET
                    + '".',
                ) from None

        return matrix

    def soc_zeem_in_z_angular_magnetic_momentum_basis(
        self,
        group: str,
        start_state: int,
        stop_state: int,
        matrix_type: Union["soc", "zeeman"],
        basis_kind: Union["magnetic", "total_angular"],
        rotation: ndarray[float64] = None,
        field: float64 = None,
        orientation: ndarray[float64] = None,
        slt: str = None,
    ) -> ndarray[complex128]:
        """
        Calculates SOC or Zeeman matrix in "z" magnetic or total angular
        momentum basis.

        Parameters
        ----------
        group : str
            Name of a group containing results of relativistic ab initio
            calculations used for obtaining the SOC or Zeeman matrix.
        start_state : int
            Number of the first SOC state to be included.
        stop_state : int
            Number of the last SOC state to be included. If both start and stop
            are set to zero all available states from the file will be used
        matrix_type : Union["soc", "zeeman"]
            Type of a matrix to be decomposed. Two options available: "soc" or
            "zeeman".
        basis_kind : Union["magnetic", "total_angular"]
            Kind of a basis. Two options available: "magnetic" or
            "total_angular" for the decomposition in a particular basis
        rotation : ndarray[float64], optional
            A (3,3) orthogonal rotation matrix used to rotate momenta matrices.
            Note that the inverse matrix has to be given to rotate the
            reference frame instead., by default None
        field : float64, optional
            _description_, by default None
        orientation : ndarray[float64], optional
            Orientation of the magnetic field in the form of an ArrayLike
            structure (can be converted to numpy.NDArray) [direction_x,
            direction_y, direction_z]., by default None
        slt : str, optional
            If given the results will be saved in a group of this name to .slt
            file with suffix: _{matrix_type}_matrix_in_{basis_kind}_basis.,
            by default None

        Returns
        -------
        ndarray[complex128]
            Matrix in a given kind of basis.

        Raises
        ------
        SltInputError
            If an unsuported type of matrix or basis is provided.
        SltInputError
            If there is no field value or orientation provided for Zeeman
            matrix.
        SltSaveError
            If the name of the group already exists in the .slt file.
        SltCompError
            If the calculation of a matrix in "z" basis is unsuccessful.
        SltFileError
            If the program is unable to correctly save results to .slt file.
        """
        if (matrix_type not in ["zeeman", "soc"]) or (
            basis_kind not in ["total_angular", "magnetic"]
        ):
            raise SltInputError(
                NotImplementedError(
                    "The only valid matrix types and pseudo spin kinds are"
                    ' "soc" or "zeeman" and "magnetic" or "total_angular"'
                    " respectively."
                )
            )

        if matrix_type == "zeeman" and (
            (field is None) or (orientation is None)
        ):
            raise SltInputError(
                ValueError(
                    "For Zeeman matrix provide field value and orientation."
                )
            )

        if slt is not None:
            slt_group_name = (
                f"{slt}_{matrix_type}_matrix_in_{basis_kind}_basis"
            )
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
            if matrix_type == "zeeman":
                orientation = _normalize_orientation(orientation)
                matrix = _get_zeeman_matrix_in_z_pseudo_spin_basis(
                    self._hdf5,
                    group,
                    field,
                    orientation,
                    start_state,
                    stop_state,
                    basis_kind,
                    rotation,
                )
            elif matrix_type == "soc":
                matrix = _get_soc_matrix_in_z_pseudo_spin_basis(
                    self._hdf5,
                    group,
                    start_state,
                    stop_state,
                    basis_kind,
                    rotation,
                )
        except Exception as exc:
            raise SltCompError(
                self._hdf5,
                exc,
                "Failed to compute matrix from ITOs from "
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
                    f"{slt}_matrix",
                    f"Dataset containing {matrix_type} matrix in"
                    f' {basis_kind} momentum "z" basis calculated from'
                    f" group: {group}.",
                    f"Group({slt}) containing {matrix_type} matrix in"
                    f' {basis_kind} momentum "z" basis calculated from'
                    f" group: {group}.",
                ] = matrix[:, :]
            except Exception as exc:
                raise SltFileError(
                    self._hdf5,
                    exc,
                    f'Failed to save {matrix} matrix in "z"'
                    f" {basis_kind} basis to "
                    + BLUE
                    + "Group "
                    + RESET
                    + '"'
                    + BLUE
                    + slt_group_name
                    + RESET
                    + '".',
                ) from None

        return matrix

    def plot_mth(
        self,
        group: str,
        show_fig: bool = True,
        save: bool = False,
        save_path: str = ".",
        save_name: str = "",
        colour_map_name: str or list[str] = "rainbow",
        xlim: tuple[int or float] = (),
        ylim: tuple[int or float] = (),
        xticks: int or float = 1,
        yticks: int or float = 0,
        field: Union["B", "H"] = "B",
    ):
        """
        Creates graphs of M(H,T) given name of the group in .slt file, graphs
        can be optionally shown, saved, colour palettes can be changed.

        Parameters
        ----------
        group: str
            Name of a group from .slt file for which a plot will be created.
        show_fig: bool = True
            Determines if plot is shown.
            Possible use: saving many plots automatically without preview.
        save: bool = False
            Determines if the plot is saved.
        save_path: str = "."
            Determines a path where the file will be saved if save = True.
        save_name: str = ""
            Determines name of the file that would be created if save = True,
            if left empty it will use following format: f"mgh_{group}.tiff".
        colour_map_name: str or list[str] = "rainbow"
            Input of the colour_map function.
        xlim: tuple(optional: float, optional: float) = ()
            Determines the lower and upper limit of the x-axis if two floats
            are passed, or just the upper limit if one is passed.
        ylim: tuple(optional: float, optional: float) = ()
            Determines the lower and upper limit of the y-axis if two floats
            are passed, or just the upper limit if one is passed.
        xticks: int or float = 1
            Determines the frequency of x major ticks.
        yticks: int or float = 0
            Determines the frequency of y major ticks.
        field: Union['B','H'] = 'B'
            Determines the field unit - B[T] or H[kOe].

        Returns
        -------
        Nothing

        Raises
        ------
        SltFileError
            If unable to load the data file. Most likely encountered if the
            group name is incorrect.
        SltPlotError
            If unable to create the plot.
        SltSaveError
            If unable to save the plot as an image.

        See Also
        --------
        slothpy.Compound.calculate_mth
        """
        try:
            # Getting data from .slt or sloth file
            mth = self[f"{group}_magnetisation", f"{group}_mth"]
            fields = self[f"{group}_magnetisation", f"{group}_fields"]
            if field == "H":
                fields *= 10
                xticks *= 10
            temps = self[f"{group}_magnetisation", f"{group}_temperatures"]
        except Exception as exc:
            raise SltFileError(
                self._hdf5,
                exc,
                "Failed to load magnetisation file"
                + BLUE
                + "Group "
                + RESET
                + '"'
                + BLUE
                + f"{group}"
                + RESET
                + '".'
                + RED
                + "Check if the group exists.",
            ) from None

        try:
            # Plotting in matplotlib
            fig, ax = subplots()
            # Defining colour maps for graphs
            colour = iter(
                colour_map(colour_map_name)(linspace(0, 1, len(temps)))
            )
            # Creating a plot
            for i, mh in enumerate(mth):
                c = next(colour)
                ax.plot(fields, mh, linewidth=2, c=c, label=f"{temps[i]} K")

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
            tight_layout()
            if show_fig:
                _display_plot(fig, partial(close, "all"))
        except Exception as exc:
            close("all")
            raise SltPlotError(
                self._hdf5,
                exc,
                "Failed to plot magnetisation data"
                + BLUE
                + "Group "
                + RESET
                + '"'
                + BLUE
                + f"{group}"
                + RESET
                + '".',
            ) from None
        if save:
            try:
                # Saving plot figure
                if not save_name:
                    fig.savefig(
                        path.join(save_path, f"mgh_{group}.tiff"), dpi=600
                    )
                else:
                    fig.savefig(path.join(save_path, save_name), dpi=600)

            except Exception as exc:
                close("all")
                raise SltSaveError(
                    self._hdf5,
                    exc,
                    "Failed to save magnetisation data plot "
                    + BLUE
                    + "Group "
                    + RESET
                    + '"'
                    + BLUE
                    + f"{group}"
                    + RESET
                    + '", filename: '
                    + PURPLE
                    + f"mgh_{group}.tiff",
                ) from None
        close("all")

    def plot_chitht(
        self,
        group: str,
        show_fig: bool = True,
        save: bool = False,
        save_path: str = ".",
        save_name: str = "",
        colour_map_name: str or list[str] = "funmat",
        xlim: tuple[int or float] = (),
        ylim: tuple[int or float] = (),
        xticks: int or float = 100,
        yticks: int or float = 0,
        field: Union["B", "H"] = "B",
    ):
        """
        Creates graphs of chiT(H,T) or chi(H,T) depending on the content of
        .slt file, given a name of the group in .slt file, graphs can be
        optionally saved, colour palettes can be changed.

        Parameters
        ----------
        group: str
            Name of a group from .slt file for which a plot will be created.
        show_fig: bool = True
            Determines if plot is shown.
            Possible use: saving many plots automatically without preview.
        save: bool = False
            Determines if the plot is saved.
        save_path: str = "."
            Determines a path where the file will be saved if save = True.
        save_name: str = ""
            Determines name of the file that would be created if save = True,
            if left empty it will use following format: f"chitht_{group}.tiff".
        colour_map_name: str or list[str] = 'funmat'
            Input of colour_map function.
        xlim: tuple(optional: float, optional: float) = ()
            Determines the lower and upper limit of the x-axis if two floats
            are passed, or just the upper limit if one is passed.
        ylim: tuple(optional: float, optional: float) = ()
            Determines the lower and upper limit of the y-axis if two floats
            are passed, or just the upper limit if one is passed.
        xticks: int or float = 100
            Determines the frequency of x major ticks.
        yticks: int or float = 0
            Determines the frequency of y major ticks.
        field: Union['B','H'] = 'B'
            Determines the field unit - B[T] or H[kOe].

        Returns
        -------
        Nothing

        Raises
        ------
        SltFileError
            If unable to load the data file. Most likely encountered if the
            group name is incorrect.
        SltPlotError
            If unable to create the plot.
        SltSaveError
            If unable to save the plot as an image.

        See Also
        --------
        slothpy.Compound.calculate_chitht
        """
        try:
            # Getting data from .slt or sloth file
            try:
                chi = self[f"{group}_susceptibility", f"{group}_chiht"]
                T = False
            except Exception as exc:
                chi = self[f"{group}_susceptibility", f"{group}_chitht"]
                T = True
            fields = self[f"{group}_susceptibility", f"{group}_fields"]
            if field == "H":
                fields *= 10
            temps = self[f"{group}_susceptibility", f"{group}_temperatures"]
        except Exception as exc:
            raise SltFileError(
                self._hdf5,
                exc,
                "Failed to load susceptibility file"
                + BLUE
                + "Group "
                + RESET
                + '"'
                + BLUE
                + f"{group}"
                + RESET
                + '".'
                + RED
                + "Check if the group exists.",
            ) from None

        try:
            # Plotting in matplotlib
            fig, ax = subplots()
            # Defining colour maps for graphs
            colour = iter(
                colour_map(colour_map_name)(linspace(0, 1, len(fields)))
            )
            # Creating a plot
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
            tight_layout()
            if show_fig:
                _display_plot(fig, partial(close, "all"))
        except Exception as exc:
            close("all")
            raise SltPlotError(
                self._hdf5,
                exc,
                "Failed to plot susceptibility data"
                + BLUE
                + "Group "
                + RESET
                + '"'
                + BLUE
                + f"{group}"
                + RESET
                + '".',
            ) from None
        if save:
            try:
                # Saving plot figure
                if not save_name:
                    fig.savefig(
                        path.join(save_path, f"chitht_{group}.tiff"), dpi=300
                    )
                else:
                    fig.savefig(path.join(save_path, save_name), dpi=300)
            except Exception as exc:
                close("all")
                raise SltSaveError(
                    self._hdf5,
                    exc,
                    "Failed to save susceptibility data plot "
                    + BLUE
                    + "Group "
                    + RESET
                    + '"'
                    + BLUE
                    + f"{group}"
                    + RESET
                    + '", filename: '
                    + PURPLE
                    + f"chitht_{group}.tiff",
                ) from None
        close("all")

    def plot_helmholtz_energyth(
        self,
        group: str,
        internal_energy: bool = False,
        show_fig: bool = True,
        save: bool = False,
        save_path: str = ".",
        save_name: str = "",
        colour_map_name: str or list[str] = "PrOr",
        xlim: tuple[int or float] = (),
        ylim: tuple[int or float] = (),
        xticks: int or float = 1,
        yticks: int or float = 0,
        field: Union["B", "H"] = "B",
    ):
        """
        Creates graphs of Helmholtz energy(T,H) or internal energy(T,H) given a
        name of the group in .slt file, graphs can be optionally saved, colour
        palettes can be changed.

        Parameters
        ----------
        group: str
            Name of a group from .slt file for which a plot will be created.
        internal_energy: bool = False
            Changes the plot from the Helmholtz to internal energy.
        show_fig: bool = True
            Determines if plot is shown.
            Possible use: saving many plots automatically without preview.
        save: bool = False
            Determines if the plot is saved.
        save_path: str = "."
            Determines a path where the file will be saved if save = True.
        save_name: str = ""
            Determines name of the file that would be created if save = True,
            if left empty it will use following format: f"energyth_{group}.tiff".
        colour_map_name: str or list[str] = 'PrOr'
            Input of the colour_map function.
        xlim: tuple(optional: float, optional: float) = ()
            Determines the lower and upper limit of the x-axis if two floats
            are passed, or just the upper limit if one is passed.
        ylim: tuple(optional: float, optional: float) = ()
            Determines the lower and upper limit of the y-axis if two floats
            are passed, or just the upper limit if one is passed.
        xticks: int or float = 100
            Determines the frequency of x major ticks.
        yticks: int = 0
            Determines the freqency of y major ticks.
        field: Union['B','H'] = 'B'
            Determines the field unit - B[T] or H[kOe].

        Returns
        -------
        Nothing

        Raises
        ------
        SltFileError
            If unable to load the data file. Most likely encountered if the
            group name is incorrect.
        SltPlotError
            If unable to create plot.
        SltSaveError
            If unable to save plot as image.

        See Also
        --------
        slothpy.Compound.calculate_helmholtz_energyth
        """
        if internal_energy:
            name = "internal"
        else:
            name = "helmholtz"
        try:
            # Getting data from .slt or sloth file
            eth = self[f"{group}_{name}_energy", f"{group}_eth"]
            fields = self[f"{group}_{name}_energy", f"{group}_fields"]
            if field == "H":
                fields *= 10
                xticks *= 10
            temps = self[f"{group}_{name}_energy", f"{group}_temperatures"]
        except Exception as exc:
            raise SltFileError(
                self._hdf5,
                exc,
                "Failed to load Helmholtz energy file"
                + BLUE
                + "Group "
                + RESET
                + '"'
                + BLUE
                + f"{group}"
                + RESET
                + '".'
                + RED
                + "Check if the group exists.",
            ) from None

        try:
            # Plotting in matplotlib
            fig, ax = subplots()
            # Defining colour maps for graphs
            colour = iter(
                colour_map(colour_map_name)(linspace(0, 1, len(temps)))
            )
            # Creating a plot
            for i, eh in enumerate(eth):
                c = next(colour)
                ax.plot(fields, eh, linewidth=2, c=c, label=f"{temps[i]} K")

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
            ax.set_ylabel(r"$E\ /\ \mathrm{cm^{-1}}$")
            if xlim:
                if len(xlim) == 2:
                    ax.set_ylim(xlim[0], xlim[1])
                else:
                    ax.set_ylim(xlim[0])
            else:
                if len(temps) > 17:
                    ax.set_xlim(0, fields[-1])
                    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

                else:
                    ax.set_xlim(0, fields[-1] + 0.3 * fields[-1])
                    ax.legend()
            if ylim:
                if len(ylim) == 2:
                    ax.set_ylim(ylim[0], ylim[1])
                else:
                    ax.set_ylim(ylim[0])
            tight_layout()
            if show_fig:
                _display_plot(fig, partial(close, "all"))
        except Exception as exc:
            close("all")
            raise SltPlotError(
                self._hdf5,
                exc,
                "Failed to plot energy data"
                + BLUE
                + "Group "
                + RESET
                + '"'
                + BLUE
                + f"{group}"
                + RESET
                + '".',
            ) from None
        if save:
            try:
                # Saving plot figure
                if not save_name:
                    fig.savefig(
                        path.join(save_path, f"energyth_{group}.tiff"), dpi=600
                    )
                else:
                    fig.savefig(path.join(save_path, save_name), dpi=600)
            except Exception as exc:
                close("all")
                raise SltSaveError(
                    self._hdf5,
                    exc,
                    "Failed to save energy data plot "
                    + BLUE
                    + "Group "
                    + RESET
                    + '"'
                    + BLUE
                    + f"{group}"
                    + RESET
                    + '", filename: '
                    + PURPLE
                    + f"energyth_{group}.tiff",
                ) from None
        close("all")

    def plot_zeeman(
        self,
        group: str,
        show_fig: bool = True,
        save: bool = False,
        save_path: str = ".",
        save_name: str = "",
        colour_map_name1: str or list[str] = "BuPi",
        colour_map_name2: str or list[str] = "BuPi_r",
        single: bool = False,
        xlim: tuple[int or float] = (),
        ylim: tuple[int or float] = (),
        xticks: int or float = 1,
        yticks: int or float = 0,
        field: Union["H", "B"] = "B",
    ):
        """
        Function that creates graphs of E(H,orientation) given name of
        the group in .slt file, graphs can be optionally saved, colour palettes
        can be changed.

        Parameters
        ----------
        group: str
            Name of a group from .slt file for which plot will be created.
        show_fig: bool = True
            Determines if plot is shown.
            Possible use: saving many plots automatically without preview.
        save: bool = False
            Determines if the plot is saved.
        save_path: str = "."
            Determines a path where the file will be saved if save = True.
        save_name: str = ""
            Determines name of the file that would be created if save = True,
            if left empty it will use following format: f"zeeman_{group}.tiff"
            or f"zeeman_{group}_{orientation[i]}.tiff".
        colour_map_name1: str or list[str] = 'BuPi'
            Input of the colour_map function, determines a colour of the lower
            set of split lines.
        colour_map_name2: str or list[str] = 'BuPi_r'
            Input of the colour_map function, determines a colour of the higher
            set of split lines.
        single: bool = False
            Determines if all orientations are plotted together if plot is not
            a result of averaging.
        xlim: tuple of 1-2 floats = ()
            Determines the lower and upper limit of x-axis if two floats are
            passed, or just the upper limit if one is passed.
        ylim: tuple of 1-2 floats = ()
            Determines the lower and upper limit of y-axis if two floats are
            passed, or just the upper limit if one is passed.
        xticks: int or float = 1
            Determines the frequency of x major ticks.
        yticks: int or float = 0
            Determines the frequency of y major ticks.
        field: Union['B','H'] = 'B'
            Determines the field unit - B[T] or H[kOe].

        Returns
        -------
        Nothing

        Raises
        ------
        SltFileError
            If unable to load data file. Most likely encountered if the
            group name is incorrect.
        SltPlotError
            If unable to create the plot.
        SltSaveError
            If unable to save the plot as an image.

        See Also
        --------
        slothpy.Compound.calculate_zeeman_splitting
        """
        try:
            # Getting data from .slt
            zeeman = self[f"{group}_zeeman_splitting", f"{group}_zeeman"]
            fields = self[f"{group}_zeeman_splitting", f"{group}_fields"]
            if field == "H":
                fields *= 10
                xticks *= 10
            orientations = self[
                f"{group}_zeeman_splitting", f"{group}_orientations"
            ]

        except Exception as exc:
            raise SltFileError(
                self._hdf5,
                exc,
                f"Failed to load Zeeman splitting file"
                + BLUE
                + "Group "
                + RESET
                + '"'
                + BLUE
                + f"{group}"
                + RESET
                + '".'
                + RED
                + "Check if group exist.",
            ) from None

        try:
            if orientations.shape[1] != 3:
                single = True
            # Plotting in matplotlib
            if not single:
                number_of_plots = orientations.shape[0]
                if number_of_plots % 5 == 0:
                    fig = figure(figsize=(16, 3.2 * (number_of_plots / 5)))
                    gs = GridSpec(int(number_of_plots / 5), 5)
                    divisor = 5
                elif number_of_plots % 3 == 0:
                    fig = figure(figsize=(9.6, 3.2 * (number_of_plots / 3)))
                    gs = GridSpec(int(number_of_plots / 3), 3)
                    divisor = 3
                elif number_of_plots % 2 == 0:
                    fig = figure(figsize=(6.4, 3.2 * (number_of_plots / 2)))
                    gs = GridSpec(int(number_of_plots / 2), 2)
                    divisor = 2
                else:
                    fig = figure(figsize=(6.4, 3.2 * number_of_plots))
                    gs = GridSpec(1, number_of_plots)
                    divisor = 1
                # Creating a plot
                for i, zee in enumerate(zeeman):
                    if i % divisor != 0:
                        rc(
                            "axes",
                            prop_cycle=_custom_colour_cycler(
                                len(zeeman[0][0]),
                                colour_map_name1,
                                colour_map_name2,
                            ),
                        )
                        multiple_plots = fig.add_subplot(
                            gs[i // divisor, i % divisor]
                        )
                        plot(fields, zee, linewidth=0.75)
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
                        if orientations.shape[1] != 3:
                            title("Averaged Splitting")
                        else:
                            title(
                                f"Orientation [{round(orientations[i][0], 3)} "
                                + f"{round(orientations[i][1], 3)} {round(orientations[i][2], 3)}]"
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
                        if (i // divisor) == 0:
                            rc(
                                "axes",
                                prop_cycle=_custom_colour_cycler(
                                    len(zeeman[0][0]),
                                    colour_map_name1,
                                    colour_map_name2,
                                ),
                            )
                            multiple_plots = fig.add_subplot(
                                gs[i // divisor, i % divisor]
                            )
                            plot(fields, zee, linewidth=0.75)
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
                            multiple_plots.tick_params(which="major", length=7)
                            multiple_plots.tick_params(
                                which="minor", length=3.5
                            )
                            multiple_plots.yaxis.set_minor_locator(
                                AutoMinorLocator(2)
                            )
                            if orientations.shape[1] != 3:
                                title("Averaged Splitting")
                            else:
                                title(
                                    "Orientation"
                                    f" [{round(orientations[i][0], 3)} "
                                    f"{round(orientations[i][1], 3)} {round(orientations[i][2], 3)}]"
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
                            rc(
                                "axes",
                                prop_cycle=_custom_colour_cycler(
                                    len(zeeman[0][0]),
                                    colour_map_name1,
                                    colour_map_name2,
                                ),
                            )
                            multiple_plots = fig.add_subplot(
                                gs[i // divisor, i % divisor]
                            )
                            plot(fields, zee, linewidth=0.75)
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
                            multiple_plots.tick_params(which="major", length=7)
                            multiple_plots.tick_params(
                                which="minor", length=3.5
                            )
                            multiple_plots.yaxis.set_minor_locator(
                                AutoMinorLocator(2)
                            )
                            if orientations.shape[1] == 3:
                                title("Averaged Splitting")
                            else:
                                title(
                                    "Orientation"
                                    f" [{round(orientations[i][0], 3)} "
                                    f"{round(orientations[i][1], 3)} {round(orientations[i][2], 3)}]"
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
                if field == "B":
                    fig.supxlabel(r"$B\ /\ \mathrm{T}$")
                if field == "H":
                    fig.supxlabel(r"$H\ /\ \mathrm{kOe}$")
                fig.supylabel(r"$\mathrm{Energy\ /\ cm^{-1}}$")
                tight_layout()
                if show_fig:
                    _display_plot(fig, partial(close, "all"))
            elif single:
                for i, zee in enumerate(zeeman):
                    rc(
                        "axes",
                        prop_cycle=_custom_colour_cycler(
                            len(zeeman[0][0]),
                            colour_map_name1,
                            colour_map_name2,
                        ),
                    )
                    fig, ax = subplots()
                    ax.plot(fields, zee, linewidth=0.75)
                    if orientations.shape[1] != 3:
                        title("Averaged Splitting")
                    else:
                        title(
                            "Orientation"
                            f" [{round(orientations[i][0], 3)} "
                            f"{round(orientations[i][1], 3)} {round(orientations[i][2], 3)}]"
                        )
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
                    tight_layout()
                    if show_fig:
                        _display_plot(fig, partial(close, "all"))
                    if save:
                        try:
                            # Saving plot figure
                            if not save_name:
                                fig.savefig(
                                    path.join(
                                        save_path,
                                        f"zeeman_{group}_Orientation"
                                        f" {orientations[i]}.tiff",
                                    ),
                                    dpi=600,
                                )
                            else:
                                fig.savefig(
                                    path.join(save_path, save_name),
                                    dpi=600,
                                )
                        except Exception as exc:
                            close("all")
                            raise SltSaveError(
                                self._hdf5,
                                exc,
                                f"Failed to save zeeman splitting data plot"
                                + BLUE
                                + "Group "
                                + RESET
                                + '"'
                                + BLUE
                                + f"{group}"
                                + RESET
                                + '", filename: '
                                + PURPLE
                                + f"zeeman_{group}_Orientation"
                                + f" {orientations[i]}.tiff",
                            ) from None

        except Exception as exc:
            close("all")
            raise SltPlotError(
                self._hdf5,
                exc,
                f"Failed to plot zeeman splitting data"
                + BLUE
                + "Group "
                + RESET
                + '"'
                + BLUE
                + f"{group}"
                + RESET
                + '".',
            ) from None
        if save and not single:
            try:
                # Saving plot figure
                if not save_name:
                    fig.savefig(
                        path.join(save_path, f"zeeman_{group}.tiff"),
                        dpi=600,
                    )
                else:
                    fig.savefig(
                        path.join(save_path, save_name),
                        dpi=600,
                    )
            except Exception as exc:
                close("all")
                raise SltSaveError(
                    self._hdf5,
                    exc,
                    f"Failed to save zeeman splitting data plot "
                    + BLUE
                    + "Group "
                    + RESET
                    + '"'
                    + BLUE
                    + f"{group}"
                    + RESET
                    + '", filename: '
                    + PURPLE
                    + f"zeeman_{group}.tiff",
                ) from None
        close("all")

    def plot_3d(
        self,
        group: str,
        data_type: Union[
            "chit",
            "chi",
            "helmholtz_energy",
            "internal_energy",
            "magnetisation",
        ],
        field_i: int,
        temp_i: int,
        show_fig: bool = True,
        save: bool = False,
        save_path: str = ".",
        save_name: str = "",
        colour_map_name: str or list[str] = "dark_rainbow_r_l",
        round_temp: int = 3,
        round_field: int = 3,
        lim_scalar: float = 1.0,
        ticks: float = 1.0,
        r_density: int = 0,
        c_density: int = 0,
        elev: int = 30,
        azim: int = -60,
        roll: int = 0,
        axis_off: bool = False,
        add_g_tensor_axes: bool = False,
        axes_group: str = "",
        axes_colors: list[str] = ["r", "g", "b"],
        doublet_number: int = None,
        axes_scale_factor: float64 = 1.0,
        rotation: ndarray[float64] = None,
    ):
        """
        Creates 3d plots of data dependent on field B[T] and temperature T[K].

        Parameters
        ----------
        group: str
            Name of a group from .slt file for which a plot will be created.
        data_type: Union["chit", "chi", "helmholtz_energy", "internal_energy",
          "magnetisation"]
            Type of the data that will be used to create plot.
        field_i: int
            Index of the field from dataset that will be used for the plot.
        temp_i: int
            Index of the temperature from the dataset that will be used for the
            plot.
        show_fig: bool = True
            Determines if plot is shown.
            Possible use: saving many plots automatically without preview.
        save: bool = False
            Determines if the plot is saved.
        save_path: str = "."
            Determines path where file will be saved if save = True.
        save_name: str = ""
            Determines name of a file that would be created if save = True,
            if left empty it will use following format:
            f"{group}_3d_{data_type}.tiff".
        colour_map_name: str or list[str] = 'dark_rainbow_r_l'
            Input of the colour_map function.
        round_temp: int = 3
            Determines how many digits will be rounded in the graph's title
            for temperature.
        round_field: int = 3
            Determines how many digits will be rounded in the graph's title
            for field.
        lim_scalar: float = 1.
            Scalar used to set limits of the axes, smaller values magnify the
            plotted figure.
        ticks: float = 1.
            Frequency of the ticks on all axes.
        r_density: int = 0
            Determines the rcount of a 3D plot.
        c_density: int = 0
            Determines the ccount of a 3D plot.
        elev: int = 30
            Determines an angle between a viewing position and the xy plane.
        azim: int = -60
            Determines a rotation of a viewing position in ralation to z axis.
        roll: int = 0
            Determines a rotation of camera around the viewing (position) axis.
        axis_off: bool = False
            Determines if the axes are turned off.
        add_g_tensor_axes: bool = False
            Determines if add to the plot main magnetic axes scaled by the
            correcponding pseudo-g-tensor values.
        axes_group: str = ""
            Name of a group from calculate_g_tensor_axes method from .slt file.
        axes_colors: list[str] = ['r','g','b']
            Determines the colors of the magnetic axes in order of x, y, z.
            Accepts matplotlib colors inputs, for example HTML colour codes.
        doublet_number: int = None
            Number of a doublet for which axes will be added to the plot.
        axes_scale_factor: float64 = 1.0
            Scale factor determining the length of the longest (main) magnetic
            axis concerning the maximal value of the loaded data and setting
            a maximal limit of the plot's xyz axes. It should be set > 1
            otherwise, some data will end up missing from the plot! The limit
            is max(loaded_data) * axes_scale_factor.
        rotation: ndarray[float64] = None
            Has to be given if 3d data was calculated with optional rotation of
            the coordinate frame and add_g_tensor_axes option is turned on.
            One must provide the same rotation as that used for the simulation
            to apply it to the magnetic axes.

        Returns
        -------
        Nothing

        Raises
        ------
        SltFileError
            If unable to load data file. Most likely encountered if the group
            name is incorrect.
        SltPlotError
            If unable to create the plot.
        SltSaveError
            If unable to save the plot as an image.

        See Also
        --------
        slothpy.Compound.calculate_mag_3d, slothpy.Compound.calculate_chit_3d,
        slothpy.Compound.calculate_helmholtz_energy_3d,
        slothpy.Compound.calculate_g_tensor_axes
        """
        if (not isinstance(axes_scale_factor, float)) or (
            axes_scale_factor < 1
        ):
            raise SltInputError(
                ValueError(
                    "Axes scale factor has to be a float greater than 1."
                )
            )

        try:
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
                    f" B={round(self[f'{group}_3d_susceptibility', f'{group}_fields'][field_i], round_field)} T,"
                    f"T={round(self[f'{group}_3d_susceptibility', f'{group}_temperatures'][temp_i], round_temp)} K"
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
                    f" B={round(self[f'{group}_3d_susceptibility', f'{group}_fields'][field_i], round_field)} T,"
                    f"T={round(self[f'{group}_3d_susceptibility', f'{group}_temperatures'][temp_i], round_temp)} K"
                )
            elif data_type == "helmholtz_energy":
                x = self[f"{group}_3d_helmholtz_energy", f"{group}_energy_3d"][
                    0, field_i, temp_i, :, :
                ]
                y = self[f"{group}_3d_helmholtz_energy", f"{group}_energy_3d"][
                    1, field_i, temp_i, :, :
                ]
                z = self[f"{group}_3d_helmholtz_energy", f"{group}_energy_3d"][
                    2, field_i, temp_i, :, :
                ]
                description = (
                    "Hemholtz energy dependence on direction,"
                    f" B={round(self[f'{group}_3d_helmholtz_energy', f'{group}_fields'][field_i], round_field)} T,"
                    f"T={round(self[f'{group}_3d_helmholtz_energy', f'{group}_temperatures'][temp_i], round_temp)} K"
                )
            elif data_type == "internal_energy":
                x = self[f"{group}_3d_internal_energy", f"{group}_energy_3d"][
                    0
                ]
                y = self[f"{group}_3d_internal_energy", f"{group}_energy_3d"][
                    1
                ]
                z = self[f"{group}_3d_internal_energy", f"{group}_energy_3d"][
                    2
                ]
                description = (
                    "Internal energy dependence on distance,"
                    f" B={round(self[f'{group}_3d_internal_energy', f'{group}_fields'][field_i], round_field)} T,"
                    f"T={round(self[f'{group}_3d_internal_energy', f'{group}_temperatures'][temp_i], round_temp)} K"
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
                    f" B={round(self[f'{group}_3d_magnetisation', f'{group}_fields'][field_i], round_field)} T,"
                    f"T={round(self[f'{group}_3d_magnetisation', f'{group}_temperatures'][temp_i], round_temp)} K"
                )
            else:
                raise ValueError
        except Exception as exc:
            raise SltFileError(
                self._hdf5,
                exc,
                "Failed to load the 3D data file "
                + BLUE
                + "Group "
                + RESET
                + '"'
                + BLUE
                + f"{group}"
                + RESET
                + '". '
                + f"Check if the group exists and data type {data_type} "
                + f"matches one from the list "
                + f'["chit", "chi", "helmholtz_energy", "internal_energy",'
                f' "magnetisation"].',
            ) from None
        try:
            fig = figure()
            ax = fig.add_subplot(projection="3d")
            max_array = array([max(x), max(y), max(z)])
            lim = max(max_array) * axes_scale_factor
            norm = Normalize(z.min(), z.max())
            colors = colour_map(colour_map_name)(norm(z))
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
            elif data_type == "helmholtz_energy":
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
                for axis_i in [ax.xaxis, ax.yaxis, ax.zaxis]:
                    axis_i.set_ticklabels([])
                    axis_i._axinfo["axisline"]["linewidth"] = 1
                    axis_i._axinfo["axisline"]["color"] = (0, 0, 0)
                    axis_i._axinfo["grid"]["linewidth"] = 0.5
                    axis_i._axinfo["grid"]["linestyle"] = "-"
                    axis_i._axinfo["grid"]["color"] = (0, 0, 0)
                    axis_i._axinfo["tick"]["inward_factor"] = 0.0
                    axis_i._axinfo["tick"]["outward_factor"] = 0.0
                    axis_i.set_pane_color((0.95, 0.95, 0.95))
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
            ax.azim = azim
            ax.elev = elev
            ax.roll = roll
            title(description)
            if add_g_tensor_axes:
                doublets = axes_matrix = self[
                    f"{axes_group}_g_tensors_axes", f"{axes_group}_axes"
                ].shape[0]
                if (
                    (not isinstance(doublet_number, int))
                    or (doublet_number < 0)
                    or (doublet_number > doublets - 1)
                ):
                    raise SltInputError(
                        ValueError(
                            "Doublet number must be a nonnegative integer less"
                            " or equal to the number of doublets in the axes"
                            " group."
                        )
                    ) from None
                axes_matrix = self[
                    f"{axes_group}_g_tensors_axes", f"{axes_group}_axes"
                ][doublet_number]
                g_tensor = self[
                    f"{axes_group}_g_tensors_axes", f"{axes_group}_g_tensors"
                ][doublet_number]

                vec = axes_matrix * g_tensor[newaxis, 1:]
                if rotation is not None:
                    if rotation.shape != (3, 3):
                        raise SltInputError(
                            ValueError(
                                "Input rotation matrix must be a 3x3 matrix."
                            )
                        ) from None
                    product = rotation.T @ rotation
                    if not allclose(product, identity(3), atol=1e-2, rtol=0):
                        raise SltInputError(
                            ValueError(
                                "Input rotation matrix must be orthogonal."
                            )
                        ) from None
                    vec = rotation @ vec
                max_vec = max(vec)
                vec = vec * lim / max_vec
                for i in range(3):
                    ax.plot(
                        [vec[0, i], -vec[0, i]],
                        [vec[1, i], -vec[1, i]],
                        [vec[2, i], -vec[2, i]],
                        axes_colors[i],
                        linewidth=3,
                    )
            if axis_off:
                ax.set_axis_off()
            if show_fig:
                _display_plot(fig, partial(close, "all"))
        except Exception as exc:
            close("all")
            raise SltPlotError(
                self._hdf5,
                exc,
                "Failed to plot 3D data "
                + BLUE
                + "Group "
                + RESET
                + '"'
                + BLUE
                + f"{group}"
                + RESET
                + '".',
            ) from None
        if save:
            try:
                if axis_off:
                    if not save_name:
                        fig.savefig(
                            path.join(
                                save_path, f"{group}_3d_{data_type}.tiff"
                            ),
                            transparent=True,
                            dpi=600,
                        )
                    else:
                        fig.savefig(
                            path.join(save_path, save_name),
                            transparent=True,
                            dpi=600,
                        )
                else:
                    if not save_name:
                        fig.savefig(
                            path.join(
                                save_path, f"{group}_3d_{data_type}.tiff"
                            ),
                            dpi=600,
                        )
                    else:
                        fig.savefig(
                            path.join(save_path, save_name),
                            dpi=600,
                        )
            except Exception as exc:
                close("all")
                raise SltSaveError(
                    self._hdf5,
                    exc,
                    "Failed to save the 3D data plot "
                    + BLUE
                    + "Group "
                    + RESET
                    + '"'
                    + BLUE
                    + f"{group}"
                    + RESET
                    + '", filename: '
                    + PURPLE
                    + f"{group}_3d_{data_type}.tiff",
                ) from None
        close("all")

    def animate_3d(
        self,
        group: str,
        data_type: Union[
            "chit",
            "chi",
            "helmholtz_energy",
            "internal_energy",
            "magnetisation",
        ],
        animation_variable: Union["temperature", "field"],
        filename: str,
        i_start: int = 0,
        i_end: int = -1,
        i_constant: int = 0,
        colour_map_name: str or list[str] = "dark_rainbow_r_l",
        lim_scalar: float = 1.0,
        ticks: int or float = 1,
        r_density: int = 0,
        c_density: int = 0,
        axis_off: bool = False,
        fps: int = 15,
        dpi: int = 100,
        bar: bool = True,
        bar_scale: bool = False,
        bar_colour_map_name: str or list[str] = "dark_rainbow_r",
        temp_rounding: int = 0,
        field_rounding: int = 0,
        elev: int = 30,
        azim: int = -60,
        roll: int = 0,
        add_g_tensor_axes: bool = False,
        axes_group: str = "",
        axes_colors: list[str] = ["r", "g", "b"],
        doublet_number: int = None,
        axes_scale_factor: float64 = 1.0,
        rotation: ndarray[float64] = None,
    ):
        """
        Creates animations of 3d plots dependent on field B[T]
        and temperature T[K].

        Parameters
        ----------
        group: str
            Name of a group from .slt file for which plot will be created.
        data_type: Union["chit", "chi", "helmholtz_energy", "internal_energy",
          "magnetisation"]
            Type of data that will be used to create plot.
        animation_variable: Union["temperature", "field"]
            Variable changing during animation, can take one of two values:
            temperature or field.
        filename: str
            Name of the output .gif file.
        i_start: int = 0
            Index of first frame's field/temperature.
        i_end: int = -1
            Index of last frame's field/temperature.
        i_constant: int
            Index of constant temperature/field.
        colour_map_name: str or list = 'dark_rainbow_r_l'
            Input of colour_map function, determines colour of main figure.
        lim_scalar: float = 1.
            Scalar used to set limits of axes, smaller values magnify plotted
            figure.
        ticks: float = 1
            Determines the ticks spacing.
        r_density: int = 0
            Determines rcount of 3D plot.
        c_density: int = 0
            Determines ccount of 3D plot.
        axis_off: bool = False
            Determines if axes are turned off.
        fps: int
            Number of frames per second in animation.
        dpi: int
            Dots per inch resolution of frames.
        bar: bool = True
            Determines if bar representing animation variable is shown.
        bar_scale: bool = False
            Determines if a scale should be shown for bar.
        bar_colour_map_name: str or list = 'dark_rainbow_r_l'
            Input of the colour_map function, determines the colour of the bar.
        temp_rounding: int = 0
            Determines how many decimal places are shown in bar/plot labels
            for temperatures.
        field_rounding: int = 0
            Determines how many decimal places are shown in bar/plot labels
            for fields.
        elev: int = 30
            Determines an angle between a viewing position and the xy plane.
        azim: int = -60
            Determines a rotation of a viewing position in ralation to z axis.
        roll: int = 0
            Determines a rotation of camera around the viewing (position) axis.
        add_g_tensor_axes: bool = False
            Determines if add to the plot main magnetic axes scaled by the
            correcponding pseudo-g-tensor values.
        axes_group: str = ""
            Name of a group from calculate_g_tensor_axes method from .slt file.
        axes_colors: list[str] = ['r','g','b']
            Determines the colors of the magnetic axes in order of x, y, z.
            Accepts matplotlib colors inputs, for example HTML colour codes.
        doublet_number: int = None
            Number of a doublet for which axes will be added to the plot.
        axes_scale_factor: float64 = 1.0
            Scale factor determining the length of the longest (main) magnetic
            axis concerning the maximal value of the loaded data and setting
            a maximal limit of the plot's xyz axes. It should be set > 1
            otherwise, some data will end up missing from the plot! The limit
            is max(loaded_data) * axes_scale_factor.
        rotation: ndarray[float64] = None
            Has to be given if 3d data was calculated with optional rotation of
            the coordinate frame and add_g_tensor_axes option is turned on.
            One must provide the same rotation as that used for the simulation
            to apply it to the magnetic axes.

        Returns
        -------
        Nothing

        Raises
        ------
        SltFileError
            If unable to load data file. Most likely encountered if the
            group name is incorrect.
        SltPlotError
            If unable to create the plot.

        See Also
        --------
        slothpy.Compound.calculate_mag_3d, slothpy.Compound.calculate_chit_3d,
        slothpy.Compound.calculate_helmholtz_energy_3d,
        slothpy.Compound.calculate_g_tensor_axes
        """
        if (not isinstance(axes_scale_factor, float)) or (
            axes_scale_factor < 1
        ):
            raise SltInputError(
                ValueError(
                    "Axes scale factor has to be a float greater than 1."
                )
            )

        try:
            T = False
            if data_type == "chit":
                x0 = self[f"{group}_3d_susceptibility", f"{group}_chit_3d"][0]
                y0 = self[f"{group}_3d_susceptibility", f"{group}_chit_3d"][1]
                z0 = self[f"{group}_3d_susceptibility", f"{group}_chit_3d"][2]
                fields = self[f"{group}_3d_susceptibility", f"{group}_fields"]
                temps = self[
                    f"{group}_3d_susceptibility", f"{group}_temperatures"
                ]
                T = True
            elif data_type == "chi":
                x0 = self[f"{group}_3d_susceptibility", f"{group}_chi_3d"][0]
                y0 = self[f"{group}_3d_susceptibility", f"{group}_chi_3d"][1]
                z0 = self[f"{group}_3d_susceptibility", f"{group}_chi_3d"][2]
                fields = self[f"{group}_3d_susceptibility", f"{group}_fields"]
                temps = self[
                    f"{group}_3d_susceptibility", f"{group}_temperatures"
                ]
            elif data_type == "helmholtz_energy":
                x0 = self[
                    f"{group}_3d_helmholtz_energy", f"{group}_energy_3d"
                ][0]
                y0 = self[
                    f"{group}_3d_helmholtz_energy", f"{group}_energy_3d"
                ][1]
                z0 = self[
                    f"{group}_3d_helmholtz_energy", f"{group}_energy_3d"
                ][2]
                fields = self[
                    f"{group}_3d_helmholtz_energy", f"{group}_fields"
                ]
                temps = self[
                    f"{group}_3d_helmholtz_energy", f"{group}_temperatures"
                ]
            elif data_type == "internal_energy":
                x0 = self[f"{group}_3d_internal_energy", f"{group}_energy_3d"][
                    0
                ]
                y0 = self[f"{group}_3d_internal_energy", f"{group}_energy_3d"][
                    1
                ]
                z0 = self[f"{group}_3d_internal_energy", f"{group}_energy_3d"][
                    2
                ]
                fields = self[f"{group}_3d_internal_energy", f"{group}_fields"]
                temps = self[
                    f"{group}_3d_internal_energy", f"{group}_temperatures"
                ]
            elif data_type == "magnetisation":
                x0 = self[f"{group}_3d_magnetisation", f"{group}_mag_3d"][0]
                y0 = self[f"{group}_3d_magnetisation", f"{group}_mag_3d"][1]
                z0 = self[f"{group}_3d_magnetisation", f"{group}_mag_3d"][2]
                fields = self[f"{group}_3d_magnetisation", f"{group}_fields"]
                temps = self[
                    f"{group}_3d_magnetisation", f"{group}_temperatures"
                ]
            else:
                raise ValueError(
                    "Acceptable data types: chit, chi, helmholtz_energy and"
                    " magnetisation"
                )
            if add_g_tensor_axes:
                doublets = axes_matrix = self[
                    f"{axes_group}_g_tensors_axes", f"{axes_group}_axes"
                ].shape[0]
                if (
                    (not isinstance(doublet_number, int))
                    or (doublet_number < 0)
                    or (doublet_number > doublets - 1)
                ):
                    raise SltInputError(
                        ValueError(
                            "Doublet number must be a nonnegative integer less"
                            " or equal to the number of doublets in the axes"
                            " group."
                        )
                    ) from None
                axes_matrix = self[
                    f"{axes_group}_g_tensors_axes", f"{axes_group}_axes"
                ][doublet_number]
                g_tensor = self[
                    f"{axes_group}_g_tensors_axes", f"{axes_group}_g_tensors"
                ][doublet_number]

        except Exception as exc:
            raise SltFileError(
                self._hdf5,
                exc,
                "Failed to load the 3D data file "
                + BLUE
                + "Group "
                + RESET
                + '"'
                + BLUE
                + f"{group}"
                + RESET
                + '". '
                + f"Check if the group exists and data type {data_type} "
                + f"matches one from the list "
                + f'["chit", "chi", "helmholtz_energy", "internal_energy",'
                f' "magnetisation"].',
            ) from None
        if animation_variable == "temperature":
            description = (
                f"{data_type[0].upper() + data_type[1:].replace('_', ' ')} dependence"
                f" on direction, B={fields[i_constant]:.4f} T"
            )
        elif animation_variable == "field":
            description = (
                f"{data_type[0].upper() + data_type[1:].replace('_', ' ')} dependence"
                f" on direction, T={temps[i_constant]:.4f} K"
            )
        else:
            raise ValueError(
                "There exist only two animation variables: field and"
                " temperature"
            )
        title = description

        try:
            fig = figure()
            ax = fig.add_subplot(projection="3d")
            if bar:
                colour = iter(
                    colour_map(bar_colour_map_name)(
                        linspace(0, 1, i_end - i_start)
                    )
                )
                indicator = linspace(0, 1, i_end - i_start)
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
                        elif (
                            data_type == "helmholtz_energy"
                            or "internal_energy"
                        ):
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
                        else:
                            raise ValueError(
                                "Acceptable data types: chit, chi,"
                                " helmholtz_energy and magnetisation"
                            )
                        max_array = array([max(x), max(y), max(z)])
                        lim = max(max_array) * axes_scale_factor
                        norm = Normalize(z.min(), z.max())
                        colors = colour_map(colour_map_name)(norm(z))
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
                        if add_g_tensor_axes:
                            vec = axes_matrix * g_tensor[newaxis, 1:]
                            if rotation is not None:
                                if rotation.shape != (3, 3):
                                    raise SltInputError(
                                        ValueError(
                                            "Input rotation matrix must be a"
                                            " 3x3 matrix."
                                        )
                                    ) from None
                                product = rotation.T @ rotation
                                if not allclose(
                                    product, identity(3), atol=1e-2, rtol=0
                                ):
                                    raise SltInputError(
                                        ValueError(
                                            "Input rotation matrix must be"
                                            " orthogonal."
                                        )
                                    ) from None
                                vec = rotation @ vec
                            max_vec = max(vec)
                            vec = vec * lim / max_vec
                            for i in range(3):
                                ax.plot(
                                    [vec[0, i], -vec[0, i]],
                                    [vec[1, i], -vec[1, i]],
                                    [vec[2, i], -vec[2, i]],
                                    axes_colors[i],
                                )
                        surface.set_facecolor((0, 0, 0, 0))
                        ax.set_xlim(-lim * lim_scalar, lim * lim_scalar)
                        ax.set_ylim(-lim * lim_scalar, lim * lim_scalar)
                        ax.set_zlim(-lim * lim_scalar, lim * lim_scalar)
                        # Important order of operations!

                        if ticks == 0:
                            for axis_i in [ax.xaxis, ax.yaxis, ax.zaxis]:
                                axis_i.set_ticklabels([])
                                axis_i._axinfo["axisline"]["linewidth"] = 1
                                axis_i._axinfo["axisline"]["color"] = (0, 0, 0)
                                axis_i._axinfo["grid"]["linewidth"] = 0.5
                                axis_i._axinfo["grid"]["linestyle"] = "-"
                                axis_i._axinfo["grid"]["color"] = (0, 0, 0)
                                axis_i._axinfo["tick"]["inward_factor"] = 0.0
                                axis_i._axinfo["tick"]["outward_factor"] = 0.0
                                axis_i.set_pane_color((0.95, 0.95, 0.95))
                        else:
                            ax.xaxis.set_minor_locator(AutoMinorLocator(2))
                            ax.yaxis.set_minor_locator(AutoMinorLocator(2))
                            ax.zaxis.set_minor_locator(AutoMinorLocator(2))
                            if not (not T and ticks == 1):
                                ax.xaxis.set_major_locator(
                                    MultipleLocator(ticks)
                                )
                                ax.yaxis.set_major_locator(
                                    MultipleLocator(ticks)
                                )
                                ax.zaxis.set_major_locator(
                                    MultipleLocator(ticks)
                                )
                        ax.grid(False)

                        ax.set_box_aspect([1, 1, 1])
                        ax.azim = azim
                        ax.elev = elev
                        ax.roll = roll
                        ax.set_title(title)
                        if axis_off:
                            ax.set_axis_off()

                        if bar:
                            c = next(colour)
                            axins = ax.inset_axes([0, 0.6, 0.098, 0.2])
                            axins.bar(
                                1,
                                indicator[i_temp - i_start],
                                width=0.2,
                                color=c,
                            )
                            axins.set_ylim(0, 1)
                            if not bar_scale:
                                axins.text(
                                    1,
                                    1,
                                    s=(
                                        f"{round(temps[i_end], temp_rounding)} K"
                                    ),
                                    verticalalignment="bottom",
                                    horizontalalignment="center",
                                )
                                axins.text(
                                    1,
                                    -0.03,
                                    s=(
                                        f"{round(temps[i_start], temp_rounding)} K"
                                    ),
                                    verticalalignment="top",
                                    horizontalalignment="center",
                                )
                                axins.axison = False
                            if bar_scale:
                                axins.get_xaxis().set_visible(False)
                                axins.xaxis.set_tick_params(labelbottom=False)
                                axins.yaxis.set_major_formatter(
                                    FuncFormatter(my_ticks)
                                )
                                axins.yaxis.set_minor_locator(
                                    AutoMinorLocator(2)
                                )

                        writer.grab_frame()
                        cla()

                elif animation_variable == "field":
                    for i_field in range(i_start, i_end):
                        x = x0[i_field, i_constant, :, :]
                        y = y0[i_field, i_constant, :, :]
                        z = z0[i_field, i_constant, :, :]
                        if data_type == "chit":
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
                        elif data_type == "helmholtz_energy":
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
                        else:
                            raise ValueError(
                                "Acceptable data types: chit, chi,"
                                " helmholtz_energy and magnetisation"
                            )
                        # title = description
                        max_array = array([max(x), max(y), max(z)])
                        lim = max(max_array) * axes_scale_factor
                        norm = Normalize(z.min(), z.max())
                        colors = colour_map(colour_map_name)(norm(z))
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
                        if add_g_tensor_axes:
                            vec = axes_matrix * g_tensor[newaxis, 1:]
                            if rotation is not None:
                                if rotation.shape != (3, 3):
                                    raise SltInputError(
                                        ValueError(
                                            "Input rotation matrix must be a"
                                            " 3x3 matrix."
                                        )
                                    ) from None
                                product = rotation.T @ rotation
                                if not allclose(
                                    product, identity(3), atol=1e-2, rtol=0
                                ):
                                    raise SltInputError(
                                        ValueError(
                                            "Input rotation matrix must be"
                                            " orthogonal."
                                        )
                                    ) from None
                                vec = rotation @ vec
                            max_vec = max(vec)
                            vec = vec * lim / max_vec
                            for i in range(3):
                                ax.plot(
                                    [vec[0, i], -vec[0, i]],
                                    [vec[1, i], -vec[1, i]],
                                    [vec[2, i], -vec[2, i]],
                                    axes_colors[i],
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
                                ax.xaxis.set_major_locator(
                                    MultipleLocator(ticks)
                                )
                                ax.yaxis.set_major_locator(
                                    MultipleLocator(ticks)
                                )
                                ax.zaxis.set_major_locator(
                                    MultipleLocator(ticks)
                                )
                        ax.grid(False)

                        ax.set_box_aspect([1, 1, 1])
                        ax.azim = azim
                        ax.elev = elev
                        ax.roll = roll

                        if axis_off:
                            axis("off")

                        if bar:
                            c = next(colour)
                            axins = ax.inset_axes([0, 0.6, 0.098, 0.2])
                            axins.bar(
                                1,
                                indicator[i_field - i_start],
                                width=0.2,
                                color=c,
                            )
                            axins.set_ylim(0, 1)

                            if not bar_scale:
                                axins.text(
                                    1,
                                    1,
                                    s=(
                                        f"{round(fields[i_end], field_rounding)} T"
                                    ),
                                    verticalalignment="bottom",
                                    horizontalalignment="center",
                                )
                                axins.text(
                                    1,
                                    -0.03,
                                    s=(
                                        f"{round(fields[i_start], field_rounding)} T"
                                    ),
                                    verticalalignment="top",
                                    horizontalalignment="center",
                                )
                                axins.axison = False
                            if bar_scale:
                                axins.get_xaxis().set_visible(False)
                                axins.xaxis.set_tick_params(labelbottom=False)
                                axins.yaxis.set_major_formatter(
                                    FuncFormatter(my_ticks)
                                )
                        writer.grab_frame()
                        cla()
                else:
                    raise ValueError(
                        "There exist only two animation variables: field and"
                        " temperature"
                    )
        except Exception as exc:
            close("all")
            raise SltPlotError(
                self._hdf5,
                exc,
                f"Failed to plot and save 3d data"
                + BLUE
                + " Group "
                + RESET
                + '"'
                + BLUE
                + f"{group}"
                + RESET
                + '".',
            ) from None

        close("all")

    def interactive_plot_3d(
        self,
        group: str,
        data_type: Union[
            "chit",
            "chi",
            "helmholtz_energy",
            "internal_energy",
            "magnetisation",
        ],
        colour_map_name: str or list[str] = "dark_rainbow_r",
        T_slider_colour: str = "#77f285",
        B_slider_colour: str = "#794285",
        temp_bar_colour_map_name: str or list[str] = "BuRd",
        field_bar_colour_map_name: str or list[str] = "BuPi",
        lim_scalar: float = 1.0,
        ticks: int or float = 1,
        bar: bool = True,
        bar_scale: bool = False,
        temp_rounding: int = 2,
        field_rounding: int = 2,
        axis_off: int = False,
        add_g_tensor_axes: bool = False,
        axes_group: str = "",
        axes_colors: list[str] = ["r", "g", "b"],
        doublet_number: int = None,
        axes_scale_factor: float64 = 1.0,
        rotation: ndarray[float64] = None,
    ):
        """
        Creates interactive widget plot dependent on field and temperature
        values.

        Parameters
        ----------
        group: str
            Name of a group from .slt file for which plot will be created.
        data_type: Union["chit", "chi", "helmholtz_energy", "internal_energy",
          "magnetisation"]
            Type of the data that will be used to create the plot.
        colour_map_name: str or list = 'dark_rainbow_r_l'
            Input of the colour_map function, determines a colour of the main
            figure.
        T_slider_colour: str
            Determines a colour of the temperature slider.
        B_slider_colour: str
            Determines a colour of the field slider.
        temp_bar_colour_map_name: str or list[str] = 'BuRd'
            Input of the colour_map function, determines a colour map of the
            temperature bar.
        field_bar_colour_map_name: str or list[str] = 'BuPi'
            Input of the colour_map function, determines a colour map of the
            field bar.
        lim_scalar: float = 1.
            Scalar used to set limits of the axes, smaller values magnify the
            plotted figure.
        ticks: float = 1
            Determines the ticks spacing.
        bar: bool = True
            Determines if the bar is shown.
        bar_scale: bool = False
            Determines if the bar scale is shown.
        temp_rounding: int = 2
            Determines how many significant digits are shown relative to the
            int(value) for temperature.
        temp_rounding: int = 2
            Determines how many significant digits are shown relative to the
            int(value) for field.
        axis_off: bool = False
            Determines if the axes are turned off.
        add_g_tensor_axes: bool = False
            Determines if add to the plot main magnetic axes scaled by the
            correcponding pseudo-g-tensor values.
        axes_group: str = ""
            Name of a group from calculate_g_tensor_axes method from .slt file.
        axes_colors: list[str] = ['r','g','b']
            Determines the colors of the magnetic axes in order of x, y, z.
            Accepts matplotlib colors inputs, for example HTML colour codes.
        doublet_number: int = None
            Number of a doublet for which axes will be added to the plot.
        axes_scale_factor: float64 = 1.0
            Scale factor determining lenght of the longest (main) magnetic axis
            with respect to the maximal value of the loaded data and setting
            maximal limit of the plot's xyz axes. It should be set > 1
            otherwise some data will end up missing on the plot! The limit is
            max(loaded_data) * axes_scale_factor.
        axes_group: str = ""
            Name of a group from calculate_g_tensor_axes method from .slt file.
        doublet_number: int = None
            Number of a doublet for which axes will be added to the plot.
        axes_scale_factor: float64 = 1.0
            Scale factor determining the length of the longest (main) magnetic
            axis concerning the maximal value of the loaded data and setting
            a maximal limit of the plot's xyz axes. It should be set > 1
            otherwise, some data will end up missing from the plot! The limit is
            max(loaded_data) * axes_scale_factor.
        rotation: ndarray[float64] = None
            Has to be given if 3d data was calculated with optional rotation of
            the coordinate frame and add_g_tensor_axes option is turned on.
            One must provide the same rotation as that used for the simulation
            to apply it to the magnetic axes.

        Returns
        -------
        Nothing

        Raises
        ------
        SltFileError
            If unable to load the data file. Most likely encountered if the
            group name is incorrect.
        SltPlotError
            If unable to create the plot.

        See Also
        --------
        slothpy.Compound.calculate_mag_3d, slothpy.Compound.calculate_chit_3d,
        slothpy.Compound.calculate_helmholtz_energy_3d,
        slothpy.Compound.calculate_g_tensor_axes
        """
        if (not isinstance(axes_scale_factor, float)) or (
            axes_scale_factor < 1
        ):
            raise SltInputError(
                ValueError(
                    "Axes scale factor has to be a float greater than 1."
                )
            )

        field_i, temp_i = 0, 0
        try:
            T = False
            if data_type == "chit":
                x0 = self[f"{group}_3d_susceptibility", f"{group}_chit_3d"][0]
                y0 = self[f"{group}_3d_susceptibility", f"{group}_chit_3d"][1]
                z0 = self[f"{group}_3d_susceptibility", f"{group}_chit_3d"][2]
                fields = self[f"{group}_3d_susceptibility", f"{group}_fields"]
                temps = self[
                    f"{group}_3d_susceptibility", f"{group}_temperatures"
                ]
                T = True
            if data_type == "chi":
                x0 = self[f"{group}_3d_susceptibility", f"{group}_chi_3d"][0]
                y0 = self[f"{group}_3d_susceptibility", f"{group}_chi_3d"][1]
                z0 = self[f"{group}_3d_susceptibility", f"{group}_chi_3d"][2]
                fields = self[f"{group}_3d_susceptibility", f"{group}_fields"]
                temps = self[
                    f"{group}_3d_susceptibility", f"{group}_temperatures"
                ]

            elif data_type == "helmholtz_energy":
                x0 = self[
                    f"{group}_3d_helmholtz_energy", f"{group}_energy_3d"
                ][0]
                y0 = self[
                    f"{group}_3d_helmholtz_energy", f"{group}_energy_3d"
                ][1]
                z0 = self[
                    f"{group}_3d_helmholtz_energy", f"{group}_energy_3d"
                ][2]
                fields = self[
                    f"{group}_3d_helmholtz_energy", f"{group}_fields"
                ]
                temps = self[
                    f"{group}_3d_helmholtz_energy", f"{group}_temperatures"
                ]

            elif data_type == "internal_energy":
                x0 = self[f"{group}_3d_internal_energy", f"{group}_energy_3d"][
                    0
                ]
                y0 = self[f"{group}_3d_internal_energy", f"{group}_energy_3d"][
                    1
                ]
                z0 = self[f"{group}_3d_internal_energy", f"{group}_energy_3d"][
                    2
                ]
                fields = self[f"{group}_3d_internal_energy", f"{group}_fields"]
                temps = self[
                    f"{group}_3d_internal_energy", f"{group}_temperatures"
                ]

            elif data_type == "magnetisation":
                x0 = self[f"{group}_3d_magnetisation", f"{group}_mag_3d"][0]
                y0 = self[f"{group}_3d_magnetisation", f"{group}_mag_3d"][1]
                z0 = self[f"{group}_3d_magnetisation", f"{group}_mag_3d"][2]
                fields = self[f"{group}_3d_magnetisation", f"{group}_fields"]
                temps = self[
                    f"{group}_3d_magnetisation", f"{group}_temperatures"
                ]
            if add_g_tensor_axes:
                doublets = axes_matrix = self[
                    f"{axes_group}_g_tensors_axes", f"{axes_group}_axes"
                ].shape[0]
                if (
                    (not isinstance(doublet_number, int))
                    or (doublet_number < 0)
                    or (doublet_number > doublets - 1)
                ):
                    raise SltInputError(
                        ValueError(
                            "Doublet number must be a nonnegative integer less"
                            " or equal to the number of doublets in the axes"
                            " group."
                        )
                    ) from None
                axes_matrix = self[
                    f"{axes_group}_g_tensors_axes", f"{axes_group}_axes"
                ][doublet_number]
                g_tensor = self[
                    f"{axes_group}_g_tensors_axes", f"{axes_group}_g_tensors"
                ][doublet_number]

        except Exception as exc:
            raise SltFileError(
                self._hdf5,
                exc,
                "Failed to load the 3D data file "
                + BLUE
                + "Group "
                + RESET
                + '"'
                + BLUE
                + f"{group}"
                + RESET
                + '".'
                + f"Check if the group exists and data type {data_type} "
                + f"matches one from the list"
                + f' ["chit", "chi", "helmholtz_energy", "internal_energy",'
                f' "magnetisation"].',
            ) from None
        try:
            fig = figure()
            global ax
            ax = fig.add_subplot(projection="3d")
            if bar:
                colour1 = colour_map(temp_bar_colour_map_name)(
                    linspace(0, 1, len(temps))
                )
                colour2 = colour_map(field_bar_colour_map_name)(
                    linspace(0, 1, len(fields))
                )

                indicator1 = linspace(0, 1, len(temps))
                indicator2 = linspace(0, 1, len(fields))
                if bar_scale:

                    def my_ticks(x, pos):
                        return f"{round(x * temps[-1], temp_rounding)} K"

                    def my_ticks2(x, pos):
                        return f"{round(x * fields[-1], field_rounding)} T"

            x = x0[field_i, temp_i, :, :]
            y = y0[field_i, temp_i, :, :]
            z = z0[field_i, temp_i, :, :]

            max_array = array([max(x), max(y), max(z)])
            lim = max(max_array) * axes_scale_factor
            norm = Normalize(z.min(), z.max())
            colors = colour_map(colour_map_name)(norm(z))
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
            if add_g_tensor_axes:
                vec = axes_matrix * g_tensor[newaxis, 1:]
                if rotation is not None:
                    if rotation.shape != (3, 3):
                        raise SltInputError(
                            ValueError(
                                "Input rotation matrix must be a 3x3 matrix."
                            )
                        ) from None
                    product = rotation.T @ rotation
                    if not allclose(product, identity(3), atol=1e-2, rtol=0):
                        raise SltInputError(
                            ValueError(
                                "Input rotation matrix must be orthogonal."
                            )
                        ) from None
                    vec = rotation @ vec
                max_vec = max(vec)
                vec = vec * lim / max_vec
                for i in range(3):
                    ax.plot(
                        [vec[0, i], -vec[0, i]],
                        [vec[1, i], -vec[1, i]],
                        [vec[2, i], -vec[2, i]],
                        axes_colors[i],
                    )
            surface.set_facecolor((0, 0, 0, 0))
            ax.set_xlim(-lim * lim_scalar, lim * lim_scalar)
            ax.set_ylim(-lim * lim_scalar, lim * lim_scalar)
            ax.set_zlim(-lim * lim_scalar, lim * lim_scalar)
            # Important order of operations!
            labelpad = 20 * len(str(ticks)) / 4
            if data_type in "chit":
                if T:
                    ax_label = (
                        r"$\chi_{\mathrm{M}}T\ /\ \mathrm{cm^{3}mol^{-1}K}$"
                    )
                else:
                    ax_label = (
                        r"$\chi_{\mathrm{M}}\ /\ \mathrm{cm^{3}mol^{-1}}$"
                    )
            elif data_type == "helmholtz_energy" or "internal_energy":
                ax_label = r"$E\ /\ \mathrm{cm^{-1}}$"
            elif data_type == "magnetisation":
                ax_label = r"$M\ /\ \mathrm{\mu_{B}}$"
            ax.set_xlabel(ax_label, labelpad=labelpad)
            ax.set_ylabel(ax_label, labelpad=labelpad)
            ax.set_zlabel(ax_label, labelpad=labelpad)
            if ticks == 0:
                for axis_i in [ax.xaxis, ax.yaxis, ax.zaxis]:
                    axis_i.set_ticklabels([])
                    axis_i._axinfo["axisline"]["linewidth"] = 1
                    axis_i._axinfo["axisline"]["color"] = (0, 0, 0)
                    axis_i._axinfo["grid"]["linewidth"] = 0.5
                    axis_i._axinfo["grid"]["linestyle"] = "-"
                    axis_i._axinfo["grid"]["color"] = (0, 0, 0)
                    axis_i._axinfo["tick"]["inward_factor"] = 0.0
                    axis_i._axinfo["tick"]["outward_factor"] = 0.0
                    axis_i.set_pane_color((0.95, 0.95, 0.95))
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
            fig.subplots_adjust(left=0.1)
            if bar:
                c = colour1[temp_i]
                axins = ax.inset_axes([-0.05, 0.7, 0.098, 0.2])
                axins.bar(1, indicator1[temp_i], width=0.2, color=c)
                axins.set_ylim(0, 1)
                c = colour2[field_i]
                axins2 = ax.inset_axes([-0.05, 0.2, 0.098, 0.2])
                axins2.bar(1, indicator2[field_i], width=0.2, color=c)
                axins2.set_ylim(0, 1)

                if not bar_scale:
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
                    axins.axison = False
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
                if bar_scale:
                    axins.get_xaxis().set_visible(False)
                    axins.xaxis.set_tick_params(labelbottom=False)
                    axins.yaxis.set_major_formatter(FuncFormatter(my_ticks))
                    axins.yaxis.set_minor_locator(AutoMinorLocator(2))
                    axins2.get_xaxis().set_visible(False)
                    axins2.xaxis.set_tick_params(labelbottom=False)
                    axins2.yaxis.set_major_formatter(FuncFormatter(my_ticks2))
                    axins2.yaxis.set_minor_locator(AutoMinorLocator(2))

            if bar:
                c = colour1[temp_i]
                axins = ax.inset_axes([-0.05, 0.7, 0.098, 0.2])
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

                max_array = array([max(x), max(y), max(z)])
                lim = max(max_array) * axes_scale_factor
                norm = Normalize(z.min(), z.max())
                colors = colour_map(colour_map_name)(norm(z))
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
                if add_g_tensor_axes:
                    vec = axes_matrix * g_tensor[newaxis, 1:]
                    if rotation is not None:
                        if rotation.shape != (3, 3):
                            raise SltInputError(
                                ValueError(
                                    "Input rotation matrix must be a 3x3"
                                    " matrix."
                                )
                            ) from None
                        product = rotation.T @ rotation
                        if not allclose(
                            product, identity(3), atol=1e-2, rtol=0
                        ):
                            raise SltInputError(
                                ValueError(
                                    "Input rotation matrix must be orthogonal."
                                )
                            ) from None
                        vec = rotation @ vec
                    max_vec = max(vec)
                    vec = vec * lim / max_vec
                    for i in range(3):
                        ax.plot(
                            [vec[0, i], -vec[0, i]],
                            [vec[1, i], -vec[1, i]],
                            [vec[2, i], -vec[2, i]],
                            axes_colors[i],
                        )
                surface.set_facecolor((0, 0, 0, 0))
                ax.set_xlim(-lim * lim_scalar, lim * lim_scalar)
                ax.set_ylim(-lim * lim_scalar, lim * lim_scalar)
                ax.set_zlim(-lim * lim_scalar, lim * lim_scalar)
                ax.set_title(
                    f"B={fields[field_i]:.4f} T, T={temps[temp_i]:.4f} K"
                )
                ax.set_xlabel(ax_label, labelpad=labelpad)
                ax.set_ylabel(ax_label, labelpad=labelpad)
                ax.set_zlabel(ax_label, labelpad=labelpad)

                if ticks == 0:
                    for axis_i in [ax.xaxis, ax.yaxis, ax.zaxis]:
                        axis_i.set_ticklabels([])
                        axis_i._axinfo["axisline"]["linewidth"] = 1
                        axis_i._axinfo["axisline"]["color"] = (0, 0, 0)
                        axis_i._axinfo["grid"]["linewidth"] = 0.5
                        axis_i._axinfo["grid"]["linestyle"] = "-"
                        axis_i._axinfo["grid"]["color"] = (0, 0, 0)
                        axis_i._axinfo["tick"]["inward_factor"] = 0.0
                        axis_i._axinfo["tick"]["outward_factor"] = 0.0
                        axis_i.set_pane_color((0.95, 0.95, 0.95))
                else:
                    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
                    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
                    ax.zaxis.set_minor_locator(AutoMinorLocator(2))
                    if not (not T and ticks == 1):
                        ax.xaxis.set_major_locator(MultipleLocator(ticks))
                        ax.yaxis.set_major_locator(MultipleLocator(ticks))
                        ax.zaxis.set_major_locator(MultipleLocator(ticks))
                ax.grid(False)
                fig.subplots_adjust(left=0.1)
                if bar:
                    c = colour1[temp_i]
                    axins = ax.inset_axes([-0.05, 0.7, 0.098, 0.2])
                    axins.bar(1, indicator1[temp_i], width=0.2, color=c)
                    axins.set_ylim(0, 1)
                    c = colour2[field_i]
                    axins2 = ax.inset_axes([-0.05, 0.2, 0.098, 0.2])
                    axins2.bar(1, indicator2[field_i], width=0.2, color=c)
                    axins2.set_ylim(0, 1)

                    if not bar_scale:
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
                        axins.axison = False
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
                    if bar_scale:
                        axins.get_xaxis().set_visible(False)
                        axins.xaxis.set_tick_params(labelbottom=False)
                        axins.yaxis.set_major_formatter(
                            FuncFormatter(my_ticks)
                        )
                        axins.yaxis.set_minor_locator(AutoMinorLocator(2))
                        axins2.get_xaxis().set_visible(False)
                        axins2.xaxis.set_tick_params(labelbottom=False)
                        axins2.yaxis.set_major_formatter(
                            FuncFormatter(my_ticks2)
                        )
                        axins2.yaxis.set_minor_locator(AutoMinorLocator(2))

                fig.canvas.draw()

            slider_temp.on_changed(slider_update)
            slider_field.on_changed(slider_update)
            if axis_off:
                ax.set_axis_off()
            _display_plot(fig, partial(close, "all"))
        except Exception as exc:
            raise SltPlotError(
                self._hdf5,
                exc,
                "Failed to plot 3D data"
                + BLUE
                + " Group "
                + RESET
                + '"'
                + BLUE
                + f"{group}"
                + RESET
                + '".',
            ) from None
