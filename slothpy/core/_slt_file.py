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

from __future__ import annotations

from typing import Union, Optional, Iterator, List
import warnings
from ast import literal_eval
from functools import wraps
from os import makedirs
from os.path import join

from h5py import File, Group, Dataset, string_dtype
from numpy import ndarray, array, empty, int32, int64, float32, float64, tensordot, abs, diag, prod
from numpy.exceptions import ComplexWarning
from numpy.linalg import norm
warnings.filterwarnings("ignore", category=ComplexWarning)
from scipy.linalg import eigvalsh
from ase import Atoms
from ase.io import write
from ase.data import atomic_numbers
from ase.cell import Cell

from slothpy.core._registry import MethodTypeMeta, type_registry
from slothpy.core._config import settings
from slothpy.core._slothpy_exceptions import slothpy_exc, KeyError, SltFileError, SltReadError, SltInputError
from slothpy.core._input_parser import validate_input
from slothpy._general_utilities._constants import RED, GREEN, BLUE, PURPLE, YELLOW, RESET
from slothpy._general_utilities._math_expresions import _magnetic_dipole_momenta_from_spins_angular_momenta, _total_angular_momenta_from_spins_angular_momenta
from slothpy._general_utilities._io import _get_dataset_slt_dtype, _group_exists, _xyz_to_slt, _supercell_to_slt, _hessian_to_slt, _read_hessian_born_charges_from_dir
from slothpy._general_utilities._constants import U_PI_A_AU, E_PI_A_AU
from slothpy._general_utilities._direct_product_space import _kron_mult
from slothpy._general_utilities._math_expresions import _subtract_const_diagonal
from slothpy._general_utilities._utils import _check_n
from slothpy.core._delayed_methods import *


class SltAttributes:
    def __init__(self, hdf5_file, item_path):
        self._hdf5 = hdf5_file
        self._item_path = item_path

    @slothpy_exc("SltFileError")
    def __getitem__(self, attr_name):
        with File(self._hdf5, 'r') as file:
            item = file[self._item_path]
            return item.attrs[attr_name]

    @slothpy_exc("SltFileError")
    def __setitem__(self, attr_name, value):
        with File(self._hdf5, 'r+') as file:
            item = file[self._item_path]
            item.attrs[attr_name] = value

    @slothpy_exc("SltFileError")
    def __repr__(self):
        with File(self._hdf5, 'r+') as file:
            file[self._item_path]
            return f"<{YELLOW}SltAttributes{RESET} for {BLUE}Group{RESET}/{PURPLE}Dataset{RESET} '{self._item_path}' in {GREEN}File{RESET} '{self._hdf5}'.>"

    @slothpy_exc("SltFileError")
    def __str__(self):
        with File(self._hdf5, 'a') as file:
            item = file[self._item_path]
            dict_str = f"{RED}Attributes{RESET}: " + ', '.join([f"{YELLOW}{key}{RESET}: {value}" for key, value in item.attrs.items()])
            formatted_str = f"{{{dict_str}}}".rstrip()
            return formatted_str
    
    @slothpy_exc("SltFileError")
    def __delitem__(self, attr_name):
        with File(self._hdf5, 'r+') as file:
            item = file[self._item_path]
            del item.attrs[attr_name]

    @slothpy_exc("SltFileError")
    def __contains__(self, item):
        with File(self._hdf5, 'r') as file:
            file_item = file[self._item_path]
            return item in file_item.attrs

class SltGroup:
    def __init__(self, hdf5_file, group_name):
        self._hdf5 = hdf5_file
        self._group_name = group_name
        self._exists = _group_exists(hdf5_file, group_name)
        self.attributes = SltAttributes(hdf5_file, group_name)

    @slothpy_exc("SltFileError")
    def __getitem__(self, key):
        full_path = f"{self._group_name}/{key}"
        with File(self._hdf5, 'r') as file:
            if full_path in file:
                item = file[full_path]
                if isinstance(item, Dataset):
                    return SltDataset(self._hdf5, full_path)
                elif isinstance(item, Group):
                    raise KeyError(f"Hierarchy only up to {BLUE}Group{RESET}/{PURPLE}Dataset{RESET} or standalone {PURPLE}Datasets{RESET} are supported in .slt files.")
            else:
                raise KeyError(f"{PURPLE}Dataset{RESET} '{key}' doesn't exist in the {BLUE}Group{RESET} '{self._group_name}'.")

    @slothpy_exc("SltSaveError")        
    def __setitem__(self, key, value):
        with File(self._hdf5, 'r+') as file:
            group = file.require_group(self._group_name)
            self._exists = True
            if key in group:
                raise KeyError(f"{PURPLE}Dataset{RESET} '{key}' already exists within the {BLUE}Group{RESET} '{self._group_name}'. Delete it manually to ensure your data safety.")
            group.create_dataset(key, data=array(value))

    @slothpy_exc("SltFileError")
    def __repr__(self): 
        if self._exists:
            return f"<{BLUE}SltGroup{RESET} '{self._group_name}' in {GREEN}File{RESET} '{self._hdf5}'.>"
        else:
            raise RuntimeError(f"This is a {BLUE}Proxy Group{RESET} and it does not exist in the .slt file yet. Initialize it by setting dataset within it - group['new_dataset'] = value.")

    @slothpy_exc("SltFileError")
    def __str__(self):
        if self._exists:
            with File(self._hdf5, 'r+') as file:
                item = file[self._group_name]
                representation = f"{RED}Group{RESET}: {BLUE}{self._group_name}{RESET} from File: {GREEN}{self._hdf5}{RESET}"
                for attribute_name, attribute_text in item.attrs.items():
                    representation += f" | {YELLOW}{attribute_name}{RESET}: {attribute_text}"
                representation += "\nDatasets: \n"
                for dataset_name, dataset in item.items():
                    representation += f"{PURPLE}{dataset_name}{RESET}"
                    for attribute_name, attribute_text in dataset.attrs.items():
                        representation += f" | {YELLOW}{attribute_name}{RESET}: {attribute_text}"
                    representation += "\n"
                return representation.rstrip()
        else:
            raise RuntimeError("This is a {BLUE}Proxy Group{RESET} and it does not exist in the .slt file yet. Initialize it by setting dataset within it - group['new_dataset'] = value.")

    @slothpy_exc("SltFileError")
    def __delitem__(self, key):
        with File(self._hdf5, 'r+') as file:
            group = file[self._group_name]
            if key not in group:
                raise KeyError(f"{PURPLE}Dataset{RESET} '{key}' does not exist in the {BLUE}Group{RESET} '{self._group_name}'.")
            del group[key]

    def delegate_method_to_slt_group(method):
        @wraps(method)
        def wrapper(self, *args, **kwargs):
            if not self._exists:
                raise SltFileError(self._hdf5, IOError(f"{BLUE}Group{RESET} '{self._group_name}' does not exist in the .slt file.")) from None
            obj = type_registry.get(self.type)
            if hasattr(obj, '_from_slt_file'):
                obj = obj._from_slt_file(self)
            else:
                obj = obj(self)
            delegated_method = getattr(obj, method.__name__, None)
            if delegated_method is None:
                raise SltInputError(AttributeError(f"'{obj.__class__.__name__}' object has no method '{method.__name__}'.")) from None
            return delegated_method(*args, **kwargs)
        
        return wrapper

    @property
    @delegate_method_to_slt_group
    def atoms_object(self) -> Atoms: 
        pass

    @property
    @delegate_method_to_slt_group
    def cell_object(self) -> Cell: 
        pass
    
    @property
    @delegate_method_to_slt_group
    def charge(self) -> int:
        pass
    
    @property
    @delegate_method_to_slt_group
    def multiplicity(self) -> int:
        pass

    @property
    @delegate_method_to_slt_group
    def hessian(self) -> ndarray:
        pass
        
    @property
    def attrs(self):
        """
        Property to mimic h5py's attribute access convention.
        """
        return self.attributes
    
    @property
    def type(self):
        try:
            return self.attributes["Type"]
        except SltFileError as exc:
            raise SltReadError(self._hdf5, None, f"{BLUE}Group{RESET}: '{self._group_name}' is not a valid SlothPy group and has no type.") from None
    
    @property
    @validate_input("HAMILTONIAN", True)
    def energies(self):
        return self["STATES_ENERGIES"]
    
    e = energies
    
    @property
    @validate_input("HAMILTONIAN", True)
    def spins(self):
        return SltDatasetSLP(self._hdf5, f"{self._group_name}/SPINS", "S")
    
    s = spins
    
    @property
    @validate_input("HAMILTONIAN", True)
    def sx(self):
        return SltDatasetSLP(self._hdf5, f"{self._group_name}/SPINS", "S", 0)
    
    @property
    @validate_input("HAMILTONIAN", True)
    def sy(self):
        return SltDatasetSLP(self._hdf5, f"{self._group_name}/SPINS", "S", 1)
    
    @property
    @validate_input("HAMILTONIAN", True)
    def sz(self):
        return SltDatasetSLP(self._hdf5, f"{self._group_name}/SPINS", "S", 2)
    
    @property
    @validate_input("HAMILTONIAN", True)
    def angular_momenta(self):
        return SltDatasetSLP(self._hdf5, f"{self._group_name}/ANGULAR_MOMENTA", "L")
    
    l = angular_momenta
    
    @property
    @validate_input("HAMILTONIAN", True)
    def lx(self):
        return SltDatasetSLP(self._hdf5, f"{self._group_name}/ANGULAR_MOMENTA", "L", 0)
    
    @property
    @validate_input("HAMILTONIAN", True)
    def ly(self):
        return SltDatasetSLP(self._hdf5, f"{self._group_name}/ANGULAR_MOMENTA", "L", 1)
    
    @property
    @validate_input("HAMILTONIAN", True)
    def lz(self):
        return SltDatasetSLP(self._hdf5, f"{self._group_name}/ANGULAR_MOMENTA", "L", 2)
    
    @property
    @validate_input("HAMILTONIAN", True)
    def electric_dipole_momenta(self):
        return SltDatasetSLP(self._hdf5, f"{self._group_name}/ELECTRIC_DIPOLE_MOMENTA", "P")
    
    p = electric_dipole_momenta
    
    @property
    @validate_input("HAMILTONIAN", True)
    def px(self):
        return SltDatasetSLP(self._hdf5, f"{self._group_name}/ELECTRIC_DIPOLE_MOMENTA", "P", 0)
    
    @property
    @validate_input("HAMILTONIAN", True)
    def py(self):
        return SltDatasetSLP(self._hdf5, f"{self._group_name}/ELECTRIC_DIPOLE_MOMENTA", "P", 1)
    
    @property
    @validate_input("HAMILTONIAN", True)
    def pz(self):
        return SltDatasetSLP(self._hdf5, f"{self._group_name}/ELECTRIC_DIPOLE_MOMENTA", "P", 2)
    
    @property
    @validate_input("HAMILTONIAN", True)
    def total_angular_momenta(self):
        return SltDatasetJM(self._hdf5, f"{self._group_name}", "J")
    
    j = total_angular_momenta
    
    @property
    @validate_input("HAMILTONIAN", True)
    def jx(self):
        return SltDatasetJM(self._hdf5, f"{self._group_name}", "J", 0)
    
    @property
    @validate_input("HAMILTONIAN", True)
    def jy(self):
        return SltDatasetJM(self._hdf5, f"{self._group_name}", "J", 1)
    
    @property
    @validate_input("HAMILTONIAN", True)
    def jz(self):
        return SltDatasetJM(self._hdf5, f"{self._group_name}", "J", 2)
    
    @property
    @validate_input("HAMILTONIAN", True)
    def magnetic_dipole_momenta(self):
        return SltDatasetJM(self._hdf5, f"{self._group_name}", "M")
    
    m = magnetic_dipole_momenta
    
    @property
    @validate_input("HAMILTONIAN", True)
    def mx(self):
        return SltDatasetJM(self._hdf5, f"{self._group_name}", "M", 0)
    
    @property
    @validate_input("HAMILTONIAN", True)
    def my(self):
        return SltDatasetJM(self._hdf5, f"{self._group_name}", "M", 1)
    
    @property
    @validate_input("HAMILTONIAN", True)
    def mz(self):
        return SltDatasetJM(self._hdf5, f"{self._group_name}", "M", 2)
    
    @delegate_method_to_slt_group
    def plot(self, *args, **kwargs):
        pass
    
    @delegate_method_to_slt_group
    def to_numpy_array(self, *args, **kwargs):
        pass
    
    @delegate_method_to_slt_group
    def to_data_frame(self, *args, **kwargs):
        pass
    
    @delegate_method_to_slt_group
    def to_csv(self, csv_filepath: str, *args, separator: str = ",", **kwargs):
        pass

    @delegate_method_to_slt_group
    def to_xyz(self, xyz_filepath: str, *args, **kwargs):
        pass

    @delegate_method_to_slt_group
    def replace_atoms(self, atom_indices: List[int], new_symbols: List[str]) -> None:
        """
        Replaces atoms at specified indices with new element symbols and updates
        the HDF5 group.
        
        Parameters:
        ----------
        atom_indices : List[int]
            List of 0-based atom indices to be replaced.
        new_symbols : List[str]
            List of new element symbols corresponding to each atom index.
        
        Raises:
        ------
        ValueError:
            If the lengths of atom_indices and new_symbols do not match.
            If any new symbol is invalid or not recognized.
        IndexError:
            If any atom index is out of bounds.
        """
        pass

    @delegate_method_to_slt_group
    def generate_finite_stencil_displacements(self, displacement_number: int, step: float, output_option: Literal["xyz", "iterator", "slt"] = "xyz", custom_directory: Optional[str] = None, slt_group_name: Optional[str] = None) -> Optional[Iterator[Atoms]]:
        """
        Generates finite stencil displacements for derivative calculations.

        Displaces each atom along x, y, and z axes in both negative and
        positive directions by a specified number of steps and step size.

        If this is used for supercells it only displaces each atom in the first
        unit cell.

        Parameters:
        ----------
        displacement_number : int
            Number of displacement steps in each direction (negative and positive).
        step : float
            Magnitude of each displacement step in Angstroms.
        output_option : str, optional
            Specifies the output mode. Options:
            - 'xyz': Write displaced structures as .xyz files.
            - 'iterator': Return an iterator yielding tuple of dislaced ASE
            Atoms objects, dofs numbers and displacements numbers
            - 'slt': Dump all displaced structures into the .slt file.
            Default is 'xyz'.
        custom_directory : str, optional
            Directory path to save .xyz files. Required if output_option is 'xyz'.
        slt_group_name : str, optional
            Name of the SltGroup to store displaced structures. Required if
            output_option is 'slt'.

        Returns:
        -------
        Iterator[Atoms] or None
            Returns an iterator of ASE Atoms objects if output_option is 'iterator'.
            Otherwise, returns None.

        Raises:
        ------
        ValueError:
            If invalid output_option is specified or required parameters are missing.
        IOError:
            If writing to files or HDF5 fails.
        """
        pass

    @delegate_method_to_slt_group
    def supercell(self, nx: int, ny: int, nz: int, out, output_option: Literal["xyz", "slt"] = "xyz", xyz_filepath: Optional[str] = None, slt_group_name: Optional[str] = None) -> SltGroup:
        """
        Generates a supercell by repeating the unit cell along x, y, and z axes.

        Repeats the unit cell `nx`, `ny`, and `nz` times along the x, y, and z
        axes respectively to create a supercell where cordinates are such that
        they start with unit cell for nx = ny = nz = 0 and then the slowest
        varying index is nx while the fastest is nz.

        Parameters:
        ----------
        nx, ny, nz : int
            Number of repetitions along the x, y, and z-axis.
        output_option : str, optional
            Specifies the output mode. Options:
            - 'xyz': Write the supercell structure as a `.xyz` file.
            - 'slt': Save the supercell structure in the `.slt` file.
            Default is 'xyz'.
        xyz_filepath : str, optional
            File path to save the `.xyz` file. Required if `output_option` is 'xyz'.
        slt_group_name : str, optional
            Name of the `SltGroup` to store the supercell structure. Required if
            `output_option` is 'slt'.

        Returns:
        -------
        None

        Raises:
        ------
        ValueError:
            If an invalid `output_option` is specified or required parameters are missing.
        IOError:
            If writing to files or HDF5 fails.

        Notes:
        -----
        - If the object is already a supercell, a warning will be issued indicating
        that a mega-cell will be created by multiplying the current parameters.
        """
        pass

    @delegate_method_to_slt_group
    def generate_supercell_finite_stencil_displacements(self, nx: int, ny: int, nz: int, displacement_number: int, step: float, output_option: Literal["xyz", "iterator", "slt"] = "xyz", custom_directory: Optional[str] = None, slt_group_name: Optional[str] = None, save_supercell_to_slt: Optional[str] = None) -> Optional[Iterator[Atoms]]:
        """
        Generates a new supercell and finite stencil displacements for it by
        displacing atoms in the first unit cell.

        Displaces each atom in the first unit cell of a supercell along x, y,
        and z axes in both negative and positive directions by a specified
        number of steps and step size.

        Parameters:
        ----------
        nx, ny, nz : int
            Number of repetitions along the x, y, and z axes to create the supercell.
        displacement_number : int
            Number of displacement steps in each direction (negative and positive).
        step : float
            Magnitude of each displacement step in Angstroms.
        output_option : str, optional
            Specifies the output mode. Options:
            - 'xyz': Write displaced structures as .xyz files.
            - 'iterator': Return an iterator yielding tuple of dislaced ASE
            Atoms objects, dofs numbers, displacements numbers, nx, ny, and nz.
            - 'slt': Dump all displaced structures into the .slt file.
            Default is 'xyz'.
        custom_directory : str, optional
            Directory path to save .xyz files. Required if output_option is 'xyz'.
        slt_group_name : str, optional
            Name of the SltGroup to store displaced structures. Required if
            output_option is 'slt'.
        save_supercell_to_slt: str, optional
            When provided, the created supercell is saved to the group of this
            name in the .slt file and can be used for further processing, e.g.
            creating Hessian after finite displacement calculations

        Returns:
        -------
        Iterator[Atoms] or None
            Returns an iterator of ASE Atoms objects if output_option is 'iterator'.
            Otherwise, returns None.

        Raises:
        ------
        ValueError:
            If invalid output_option is specified or required parameters are missing.
        IOError:
            If writing to files or HDF5 fails.

        Note:
        -----
        For a supercell, use generate_finite_stencil_displacements if you do
        not wish to repeat it further with new nx, ny, and nz.
        """
        pass

    @delegate_method_to_slt_group
    def hessian_from_finite_displacements(self, dirpath: str, format: Literal["CP2K"], slt_group_name: str, displacement_number: int, step: float, accoustic_sum_rule: Literal["symmetric", "self_term", "without"] = "symmetric", born_charges: bool = False, force_files_suffix: Optional[str] = None, dipole_momenta_files_suffix: Optional[str] = None) -> SltGroup:
        pass

    @validate_input("HAMILTONIAN", True)
    def states_energies_cm_1(self, start_state=0, stop_state=0, slt_save=None) -> SltStatesEnergiesCm1:
        return SltStatesEnergiesCm1(self, start_state, stop_state, slt_save)
    
    @validate_input("HAMILTONIAN", True)
    def states_energies_au(self, start_state=0, stop_state=0, slt_save=None) -> SltStatesEnergiesAu:
        return SltStatesEnergiesAu(self, start_state, stop_state, slt_save)
    
    @validate_input("HAMILTONIAN", True)
    def spin_matrices(self, xyz='xyz', start_state=0, stop_state=0, rotation=None, slt_save=None) -> SltSpinMatrices:
        return SltSpinMatrices(self, xyz, start_state, stop_state, rotation, slt_save)
    
    @validate_input("HAMILTONIAN", True)
    def states_spins(self, xyz='xyz', start_state=0, stop_state=0, rotation=None, slt_save=None) -> SltStatesSpins:
        return SltStatesSpins(self, xyz, start_state, stop_state, rotation, slt_save)
    
    @validate_input("HAMILTONIAN", True)
    def angular_momentum_matrices(self, xyz='xyz', start_state=0, stop_state=0, rotation=None, slt_save=None) -> SltAngularMomentumMatrices:
        return SltAngularMomentumMatrices(self, xyz, start_state, stop_state, rotation, slt_save)
    
    @validate_input("HAMILTONIAN", True)
    def states_angular_momenta(self, xyz='xyz', start_state=0, stop_state=0, rotation=None, slt_save=None) -> SltStatesAngularMomenta:
        return SltStatesAngularMomenta(self, xyz, start_state, stop_state, rotation, slt_save)

    @validate_input("HAMILTONIAN", True)
    def electric_dipole_momentum_matrices(self, xyz='xyz', start_state=0, stop_state=0, rotation=None, slt_save=None) -> SltElectricDipoleMomentumMatrices:
        return SltElectricDipoleMomentumMatrices(self, xyz, start_state, stop_state, rotation, slt_save)
    
    @validate_input("HAMILTONIAN", True)
    def states_electric_dipole_momenta(self, xyz='xyz', start_state=0, stop_state=0, rotation=None, slt_save=None) -> SltStatesElectricDipoleMomenta:
        return SltStatesElectricDipoleMomenta(self, xyz, start_state, stop_state, rotation, slt_save)

    @validate_input("HAMILTONIAN", True)
    def total_angular_momentum_matrices(self, xyz='xyz', start_state=0, stop_state=0, rotation=None, slt_save=None) -> SltTotalAngularMomentumMatrices:
        return SltTotalAngularMomentumMatrices(self, xyz, start_state, stop_state, rotation, slt_save)

    @validate_input("HAMILTONIAN", True)
    def states_total_angular_momenta(self, xyz='xyz', start_state=0, stop_state=0, rotation=None, slt_save=None) -> SltStatesTotalAngularMomenta:
        return SltStatesTotalAngularMomenta(self, xyz, start_state, stop_state, rotation, slt_save)

    @validate_input("HAMILTONIAN", True)
    def magnetic_dipole_momentum_matrices(self, xyz='xyz', start_state=0, stop_state=0, rotation=None, slt_save=None) -> SltMagneticDipoleMomentumMatrices:
        return SltMagneticDipoleMomentumMatrices(self, xyz, start_state, stop_state, rotation, slt_save)

    @validate_input("HAMILTONIAN", True)
    def states_magnetic_dipole_momenta(self, xyz='xyz', start_state=0, stop_state=0, rotation=None, slt_save=None) -> SltStatesMagneticDipoleMomenta:
        return SltStatesMagneticDipoleMomenta(self, xyz, start_state, stop_state, rotation, slt_save)
    
    def _retrieve_hamiltonian_dict(self, states_cutoff=[0,0], rotation=None, hyperfine=None, coordinates=None, local_states=True):
        states = self.attributes["States"]
        electric_dipole = False
        magnetic_interactions = False
        electric_interactions = False
        try:
            if self.attributes["Additional"] == "ELECTRIC_DIPOLE_MOMENTA":
                electric_dipole = True
        except SltFileError:
            pass
        try:
            if "m" in self.attributes["Interactions"]:
                magnetic_interactions = True
            if "p" in self.attributes["Interactions"]:
                electric_interactions = True
        except SltFileError:
            pass
        if self.attributes["Kind"] == "SLOTHPY":
            with File(self._hdf5, 'r') as file:
                group = file[self._group_name]
                
                def load_dict_from_group(group, subgroup_name):
                    data_dict = {}
                    subgroup = group[subgroup_name]
                    for key in subgroup.keys():
                        value = subgroup[key][()]
                        original_key = literal_eval(key.rsplit('_', 1)[0])
                        if original_key not in data_dict.keys():
                            data_dict[original_key] = []
                        if isinstance(value, bytes):
                            value = value.decode('utf-8')
                            if value == 'None':
                                value = None
                        elif isinstance(value, ndarray) and value.shape == ():
                            value = value.item()
                        data_dict[original_key].append(value)
                    return data_dict
                
                magnetic_centers = load_dict_from_group(group, "MAGNETIC_CENTERS")
                if not local_states:
                    for center in magnetic_centers.values():
                        center[1][0] = center[1][1]
                exchange_interactions = load_dict_from_group(group, "EXCHANGE_INTERACTIONS")
        else:
            magnetic_centers = {0:(self._group_name, (states_cutoff[0],0,states_cutoff[1]), rotation, coordinates, hyperfine)}
            exchange_interactions = None
        
        return magnetic_centers, exchange_interactions, states, electric_dipole, magnetic_interactions, electric_interactions, local_states
    
    @validate_input("HAMILTONIAN", only_group_check=True)
    def _hamiltonian_from_slt_group(self, states_cutoff=[0,0], rotation=None, hyperfine=None, local_states=True):
            return SltHamiltonian(self, states_cutoff, rotation, hyperfine, local_states)
    
    @validate_input("HAMILTONIAN")
    def zeeman_splitting(
        self,
        magnetic_fields: ndarray[Union[float32, float64]],
        orientations: ndarray[Union[float32, float64]],
        number_of_states: int = 0,
        states_cutoff: int = [0, "auto"],
        rotation: ndarray = None,
        electric_field_vector: ndarray = None,
        hyperfine: dict = None,
        number_cpu: int = None,
        number_threads: int = None,
        slt_save: str = None,
        autotune: bool = False,
    ) -> SltZeemanSplitting:
        return SltZeemanSplitting(self, magnetic_fields, orientations, number_of_states, states_cutoff, rotation, electric_field_vector, hyperfine, number_cpu, number_threads, autotune, slt_save)
    
    @validate_input("HAMILTONIAN")
    def magnetisation(
        self,
        magnetic_fields: ndarray[Union[float32, float64]],
        orientations: ndarray[Union[float32, float64]],
        temperatures: ndarray[Union[float32, float64]],
        states_cutoff: int = [0, "auto"],
        rotation: ndarray = None,
        electric_field_vector: ndarray = None,
        hyperfine: dict = None,
        number_cpu: int = None,
        number_threads: int = None,
        slt_save: str = None,
        autotune: bool = False,
    ) -> SltMagnetisation:
        return SltMagnetisation(self, magnetic_fields, orientations, temperatures, states_cutoff, rotation, electric_field_vector, hyperfine, number_cpu, number_threads, autotune, slt_save)


class SltDataset:
    def __init__(self, hdf5_file_path, dataset_path):
        self._hdf5 = hdf5_file_path
        self._dataset_path = dataset_path
        self.attributes = SltAttributes(hdf5_file_path, dataset_path)

    @slothpy_exc("SltReadError")
    def __getitem__(self, slice_):
        with File(self._hdf5, 'r') as file:
            dataset = file[self._dataset_path]
            dtype = dataset.dtype
            match str(dtype)[0]:
                case "c":
                    return dataset.astype(settings.complex)[slice_]
                case "f":
                    return dataset.astype(settings.float)[slice_]
                case "i":
                    return dataset.astype(settings.int)[slice_]
                case _:
                    return dataset[slice_]
        
    @slothpy_exc("SltSaveError")
    def __setitem__(self, slice_, value):
        with File(self._hdf5, 'r+') as file:
            dataset = file[self._dataset_path]
            dataset[slice_] = array(value)

    @slothpy_exc("SltFileError")
    def __repr__(self):
        with File(self._hdf5, 'r+') as file:
            file[self._dataset_path]
            return f"<{PURPLE}SltDataset{RESET} '{self._dataset_path}' in {GREEN}File{RESET} '{self._hdf5}'.>"
    
    @slothpy_exc("SltFileError")
    def __str__(self):
        with File(self._hdf5, 'r+') as file:
            item = file[self._dataset_path]
            representation = f"{RED}Dataset{RESET}: {PURPLE}{self._dataset_path}{RESET} from File: {GREEN}{self._hdf5}{RESET}"
            for attribute_name, attribute_text in item.attrs.items():
                representation += f" | {YELLOW}{attribute_name}{RESET}: {attribute_text}"
            return representation.rstrip()
        
    @property
    def attrs(self):
        """
        Property to mimic h5py's attribute access convention.
        """
        return self.attributes
    
    @property
    def shape(self):
        """
        Property to mimic h5py's shape access convention.
        """
        with File(self._hdf5, 'r') as file:
            dataset = file[self._dataset_path]
            return dataset.shape
        
    @property
    def dtype(self):
        """
        Property to mimic h5py's shape access convention.
        """
        return _get_dataset_slt_dtype(self._hdf5, self._dataset_path)


class SltDatasetSLP():
    def __init__(self, hdf5_file_path, dataset_path, slp, xyz: int = None):
        self._hdf5 = hdf5_file_path
        self._dataset_path = dataset_path
        self._slp = slp
        self._xyz = xyz
        self._xyz_dict = {0: "x", 1: "y", 2: "z"}

    @slothpy_exc("SltFileError")
    def __getitem__(self, slice_):
        with File(self._hdf5, 'r') as file:
            dataset = file[self._dataset_path].astype(settings.complex)
            if self._xyz is None:
                return dataset[*(slice_,) if isinstance(slice_, slice) else slice_]
            else:
                return dataset[self._xyz, *(slice_,) if isinstance(slice_, slice) else slice_]
        
    @slothpy_exc("SltFileError")
    def __repr__(self):
        return f"<{PURPLE}SltDataset{self._slp}{self._xyz_dict[self._xyz] if self._xyz is not None else ''}{RESET} from '{self._dataset_path}' in {GREEN}File{RESET} '{self._hdf5}'.>"

    @property
    def shape(self):
        """
        Property to mimic h5py's shape access convention.
        """
        with File(self._hdf5, 'r') as file:
            dataset = file[self._dataset_path]
            if self._xyz in [0,1,2]:
                return dataset.shape[1:]
            else:
                return dataset.shape
        
    @property
    def dtype(self):
        """
        Property to mimic h5py's shape access convention.
        """
        return _get_dataset_slt_dtype(self._hdf5, self._dataset_path)
    
    def _get_diagonal(self, start, stop):
        with File(self._hdf5, 'r') as file:
            dataset = file[self._dataset_path]
            size = stop - start
            if self._xyz is None:
                diag = empty((3,size), dtype = dataset.dtype)
                for i in range(start, stop):
                    diag[:,i] = dataset[:,i,i]
            else:
                diag = empty(size, dtype = dataset.dtype)
                for i in range(start, stop):
                    diag[i] = dataset[self._xyz,i,i]
            return diag.astype(settings.float)


class SltDatasetJM():
    def __init__(self, hdf5_file_path, group_path, jm, xyz: int = None):
        self._hdf5 = hdf5_file_path
        self._group_name = group_path
        self._jm = jm
        self._xyz = xyz
        self._xyz_dict = {0: "x", 1: "y", 2: "z"}

    @slothpy_exc("SltFileError")
    def __getitem__(self, slice_):
        with File(self._hdf5, 'r') as file:
            group = file[self._group_name]
            if self._xyz is not None:
                dataset_s = group["SPINS"].astype(settings.complex)[self._xyz, *(slice_,) if isinstance(slice_, slice) else slice_]
                dataset_l = group["ANGULAR_MOMENTA"].astype(settings.complex)[self._xyz, *(slice_,) if isinstance(slice_, slice) else slice_]
            else:
                dataset_s = group["SPINS"].astype(settings.complex)[slice_]
                dataset_l = group["ANGULAR_MOMENTA"].astype(settings.complex)[slice_]
            if self._jm == "J":
                return _total_angular_momenta_from_spins_angular_momenta(dataset_s, dataset_l)
            elif self._jm == "M":
                return  _magnetic_dipole_momenta_from_spins_angular_momenta(dataset_s, dataset_l)
            else:
                raise ValueError("The only supported options are 'J' for total angular momenta or 'M' for magnetic dipole momenta.")
        
    @slothpy_exc("SltFileError")
    def __repr__(self):
        return f"<{PURPLE}SltDataset{self._jm}{self._xyz_dict[self._xyz] if self._xyz is not None else ''}{RESET} from {BLUE}Group{RESET} '{self._group_name}' in {GREEN}File{RESET} '{self._hdf5}'.>"
    
    @property
    def shape(self):
        """
        Property to mimic h5py's shape access convention.
        """
        with File(self._hdf5, 'r') as file:
            dataset = file[self._group_name]["SPINS"]
            if self._xyz in [0,1,2]:
                return dataset.shape[1:]
            else:
                return dataset.shape
        
    @property
    def dtype(self):
        """
        Property to mimic h5py's shape access convention.
        """
        return _get_dataset_slt_dtype(self._hdf5, f"{self._group_name}/SPINS")
    
    def _get_diagonal(self, start, stop):
        with File(self._hdf5, 'r') as file:
            group = file[self._group_name]
            size = stop - start
            dataset_s = group["SPINS"]
            dataset_l = group["ANGULAR_MOMENTA"]
            if self._xyz is None:
                diag_s = empty((3,size), dtype = dataset_s.dtype)
                diag_l = empty((3,size), dtype = dataset_l.dtype)
                for i in range(start, stop):
                    diag_s[:,i] = dataset_s[:,i,i]
                    diag_l[:,i] = dataset_l[:,i,i]
            else:
                diag_s = empty(size, dtype = dataset_s.dtype)
                diag_l = empty(size, dtype = dataset_l.dtype)
                for i in range(start, stop):
                    diag_s[i] = dataset_s[self._xyz,i,i]
                    diag_l[i] = dataset_l[self._xyz,i,i]
            if self._jm == "J":
                return _total_angular_momenta_from_spins_angular_momenta(diag_s.astype(settings.float), diag_l.astype(settings.float))
            elif self._jm == "M":
                return  _magnetic_dipole_momenta_from_spins_angular_momenta(diag_s.astype(settings.float), diag_l.astype(settings.float))
            else:
                raise ValueError("The only supported options are 'J' for total angular momenta or 'M' for magnetic dipole momenta.")
            

class SltHamiltonian(metaclass=MethodTypeMeta):
    _method_type = "HAMILTONIAN"

    __slots__ = ["_hdf5", "_magnetic_centers", "_exchange_interactions", "_states", "_electric_dipole", "_magnetic_interactions", "_electric_interactions", "_mode", "_local_states"]

    def __init__(self, slt_group: SltGroup, states_cutoff=[0,0], rotation=None, hyperfine=None, local_states=True) -> None:
        self._hdf5: str = slt_group._hdf5
        self._magnetic_centers, self._exchange_interactions, self._states, self._electric_dipole, self._magnetic_interactions, self._electric_interactions, self._local_states = slt_group._retrieve_hamiltonian_dict(states_cutoff, rotation, hyperfine, local_states)
        self._mode: str = None # "eslpjm"
    
    @property
    def e(self):
        data = []
        for center in self._magnetic_centers.values():
            data.append(SltStatesEnergiesAu(SltGroup(self._hdf5, center[0]), stop_state=center[1][0]).eval())
        return data

    @property
    def s(self):
        data = []
        for center in self._magnetic_centers.values():
            arr = SltSpinMatrices(SltGroup(self._hdf5, center[0]), stop_state=center[1][0], rotation=center[2]).eval()
            arr.conj(out=arr)
            data.append(arr)
        return data
        
    @property
    def l(self):
        data = []
        for center in self._magnetic_centers.values():
            arr = SltAngularMomentumMatrices(SltGroup(self._hdf5, center[0]), stop_state=center[1][0], rotation=center[2]).eval()
            arr.conj(out=arr)
            data.append(arr)
        return data
    
    @property
    def p(self):
        data = []
        for center in self._magnetic_centers.values():
            arr = SltElectricDipoleMomentumMatrices(SltGroup(self._hdf5, center[0]), stop_state=center[1][0], rotation=center[2]).eval()
            arr.conj(out=arr)
            data.append(arr)
        return data

    @property
    def j(self):
        data = []
        for center in self._magnetic_centers.values():
            arr = SltTotalAngularMomentumMatrices(SltGroup(self._hdf5, center[0]), stop_state=center[1][0], rotation=center[2]).eval()
            arr.conj(out=arr)
            data.append(arr)
        return data

    @property
    def m(self):
        data = []
        for center in self._magnetic_centers.values():
            arr = SltMagneticDipoleMomentumMatrices(SltGroup(self._hdf5, center[0]), stop_state=center[1][0], rotation=center[2]).eval()
            arr.conj(out=arr)
            data.append(arr)
        return data

    @property
    def interaction_matrix(self):
        result = zeros((self._states, self._states), dtype=settings.complex)
        n = len(self._magnetic_centers.keys())
        if not any(value[3] is None for value in self._magnetic_centers.values()) and self._magnetic_interactions:
            dipole_magnetic_momenta_dict = {key: SltGroup(self._hdf5, self._magnetic_centers[key][0]).magnetic_dipole_momentum_matrices(stop_state=self._magnetic_centers[key][1][1], rotation=self._magnetic_centers[key][2]).eval().conj() for key in self._magnetic_centers.keys()}
            result = self._add_dipole_interaction(dipole_magnetic_momenta_dict, n, U_PI_A_AU, result)
            if self._electric_dipole and self._electric_interactions:
                dipole_electric_momenta_dict = {key: SltGroup(self._hdf5, self._magnetic_centers[key][0]).electric_dipole_momentum_matrices(stop_state=self._magnetic_centers[key][1][1], rotation=self._magnetic_centers[key][2]).eval().conj() for key in self._magnetic_centers.keys()}
                result = self._add_dipole_interaction(dipole_electric_momenta_dict, n, E_PI_A_AU, result)

        for (key1, key2), J in self._exchange_interactions.items():
            spin_dict = {key: SltGroup(self._hdf5, self._magnetic_centers[key][0]).spin_matrices(stop_state=self._magnetic_centers[key][1][1], rotation=self._magnetic_centers[key][2]).eval().conj() for key in self._magnetic_centers.keys()}
            for l in range(3):
                for m in range(3):
                    coeff = - J[0][l, m]
                    if abs(coeff) < 1e-13:
                        continue
                    op1 = coeff * spin_dict[key1][l]
                    op2 = spin_dict[key2][m]
                    ops = [op1 if k == key1 else op2 if k == key2 else spin_dict[k].shape[1] for k in range(n)]
                    result += _kron_mult(ops)
        
        result_tmp = result.copy()
        energy_dict = {key: SltGroup(self._hdf5, self._magnetic_centers[key][0]).states_energies_au(stop_state=self._magnetic_centers[key][1][1]).eval().astype(result.dtype) for key in self._magnetic_centers.keys()}
        for i in range(n):
            ops = [diag(energy_dict[k]) if k == i else energy_dict[k].shape[0] for k in range(n)]
            result_tmp += _kron_mult(ops)
        eigenvalues = eigvalsh(result_tmp, driver="evr", check_finite=False, overwrite_a=True, overwrite_b=True)
        _subtract_const_diagonal(result, eigenvalues[0])

        #TODO: hyperfine interactions and different types of interactions (J,L???)
        return result
    
    def _add_dipole_interaction(self, dipole_momenta_dict, n, coeff, result):
        for key1 in self._magnetic_centers.keys():
            for key2 in range(key1+1, n):
                r_vec = self._magnetic_centers[key1][3] - self._magnetic_centers[key2][3]
                r_norm = norm(r_vec)
                if r_norm <= 1e-2:
                    raise ValueError("Magnetic centers are closer than 0.01 Angstrom. Please double-check the SlothPy Hamiltonian dictionary. Quitting here.")
                coeff = coeff / r_norm ** 3
                r_vec = r_vec / r_norm
                op1 = tensordot(dipole_momenta_dict[key1], - 3. * coeff * r_vec ,axes=(0, 0))
                op2 = tensordot(dipole_momenta_dict[key2], r_vec, axes=(0, 0))
                ops = [op1 if k == key1 else op2 if k == key2 else dipole_momenta_dict[k].shape[1] for k in range(n)]
                result += _kron_mult(ops)
                for i in range(3):
                    ops[key1] = coeff * dipole_momenta_dict[key1][i]
                    ops[key2] = dipole_momenta_dict[key2][i]
                    result += _kron_mult(ops)
        
        return result

    @property
    def arrays_to_shared_memory(self):
        arrays = [item for property in self._mode for item in getattr(self, property)]
        if len(self._magnetic_centers.keys()) > 1:
            arrays.append(self.interaction_matrix)
        return arrays
    
    @property
    def info(self):
        info_list = []
        for i in range(len(self._magnetic_centers.keys())):
            info_list.append(self._magnetic_centers[i][1])
        return (self._mode, info_list, self._local_states)


#TODO From here consider using the input parser on top of the methods and move parts with validation to it


class SltXyz(metaclass=MethodTypeMeta):
    _method_type = "XYZ"

    __slots__ = ["_hdf5", "_slt_group", "_atoms", "_charge", "_multiplicity"]

    def __init__(self, slt_group: SltGroup) -> None:
        self._slt_group = slt_group
        elements = slt_group["ELEMENTS"][:]
        if isinstance(elements[0], bytes):
            elements = [elem.decode('utf-8') for elem in elements]
        self._atoms = Atoms(elements, slt_group["COORDINATES"][:])
        self._charge = None
        self._multiplicity = None
        if "Charge" in slt_group.attributes:
            self._charge = slt_group.attributes["Charge"]
        if "Multiplicity" in slt_group.attributes:
            self._multiplicity = slt_group.attributes["Multiplicity"]

    def atoms_object(self):
        return self._atoms
    
    def charge(self):
        return self._charge
    
    def multiplicity(self):
        return self._multiplicity

    def to_xyz(self, xyz_filepath: str):
        if not xyz_filepath.endswith(".xyz"):
            xyz_filepath += ".xyz"
        additional_info = ""
        if self._charge is not None:
            additional_info += f"Charge: {self._charge} "
        if self._multiplicity is not None:
            additional_info += f"Multiplicity: {self._multiplicity} "
        if self._method_type in ["UNIT_CELL", "SUPERCELL"]:
            additional_info += f"{"Cell" if self._method_type == "UNIT_CELL" else "Supercell"} parameters [a, b, c, alpha, beta, gamma]: {self._atoms.get_cell_lengths_and_angles()} "
        if self._method_type == "SUPERCELL":
            additional_info += f"Supercell_Repetitions [nx, ny, nz] = {self._nxnynz.tolist()} "
        
        write(xyz_filepath, self._atoms, comment=f"{additional_info}Created by SlothPy from File/Group '{self._slt_group._hdf5}/{self._slt_group._group_name}'")

    def replace_atoms(self, atom_indices: List[int], new_symbols: List[str]) -> None:
        if len(atom_indices) != len(new_symbols):
            raise SltInputError(ValueError("The lists 'atom_indices' and 'new_symbols' must have the same length.")) from None
        
        num_atoms = len(self._atoms)
        
        for idx in atom_indices:
            if not (0 <= idx < num_atoms):
                raise SltInputError(IndexError(f"Atom index {idx} is out of bounds for a structure with {num_atoms} atoms.")) from None
        
        for symbol in new_symbols:
            if symbol not in atomic_numbers:
                raise SltInputError(ValueError(f"Invalid element symbol: '{symbol}'. Please provide valid chemical element symbols.")) from None

        current_symbols = self._atoms.get_chemical_symbols()
        for idx, new_sym in zip(atom_indices, new_symbols):
            print(f"Replacing atom at index {idx} ({current_symbols[idx]}) with '{new_sym}'.")
            current_symbols[idx] = new_sym
        self._atoms.set_chemical_symbols(current_symbols)
        
        try:
            elements_ds = self._slt_group["ELEMENTS"]
            elements = elements_ds[:]

            if isinstance(elements[0], bytes):
                elements = [elem.decode('utf-8') for elem in elements]
            else:
                elements = list(elements)

            for idx, new_sym in zip(atom_indices, new_symbols):
                elements[idx] = new_sym
            
            if isinstance(elements_ds.dtype, type(string_dtype(encoding='utf-8'))):
                elements_encoded = array(elements, dtype='S')
            else:
                elements_encoded = array(elements, dtype=elements_ds.dtype)

            elements_ds[:] = elements_encoded
            print(f"'ELEMENTS' dataset successfully updated in group '{self._slt_group._group_name}'.")
        except Exception as exc:
            raise SltFileError(self._slt_group._hdf5, exc, f"Failed to update 'ELEMENTS' dataset in the .slt group") from None
        
        return self._slt_group
        
    def generate_finite_stencil_displacements(self, displacement_number: int, step: float, output_option: Literal["xyz", "iterator", "slt"] = "xyz", custom_directory: Optional[str] = None, slt_group_name: Optional[str] = None, _supercell: bool = False, _nx: Optional[int] = None, _ny: Optional[int] = None, _nz: Optional[int] = None) -> Optional[Iterator[Atoms]]:
        if output_option not in ['xyz', 'iterator', 'slt']:
            raise SltInputError(ValueError("Invalid output_option. Choose from 'xyz', 'iterator', or 'slt'.")) from None
        if output_option == 'xyz' and not custom_directory:
            raise SltInputError(ValueError("The custom_directory must be specified when output_option is 'xyz'.")) from None
        if output_option == 'slt' and not slt_group_name:
            raise SltInputError(ValueError("The slt_group_name must be specified when output_option is 'slt'.")) from None
        if not isinstance(displacement_number, (int, int32, int64)) or displacement_number < 0:
            raise SltInputError(ValueError("The displacement_number must be a nonnegative integer.")) from None
        try:
            float(step)
        except Exception as exc:
            raise SltInputError(exc, "Invalid step provided.") from None
        
        if output_option == 'xyz':
            makedirs(custom_directory, exist_ok=True)

        if self._method_type == "SUPERCELL":
            num_atoms = len(self._atoms) // self._nxnynz.prod()
        else:
            num_atoms = len(self._atoms)

        total_dofs = 3 * num_atoms
        n_checked = False
        if _supercell:
            n_checked = _check_n(_nx, _ny, _nz)
        if n_checked:
            atoms_tmp = self._atoms.repeat((_nx, _ny, _nz))
        else:
            atoms_tmp = self._atoms

        if self._method_type == "SUPERCELL":
            _nx, _ny, _nz = self._nxnynz
            _nx, _ny, _nz = int(_nx), int(_ny), int(_nz)
            n_checked = True

        def displacement_generator() -> Iterator[Atoms]:
            zero_geometry_flag = False
            for dof in range(total_dofs):
                axis = dof % 3
                atom_idx = dof // 3
                for multiplier in range(-displacement_number, displacement_number + 1):
                    if multiplier == 0 and zero_geometry_flag:
                        continue
                    elif multiplier == 0:
                        zero_geometry_flag = True
                        yield (atoms_tmp, dof, multiplier, _nx, _ny, _nz) if n_checked else (atoms_tmp, dof, multiplier)
                        continue
                    displaced_atoms = atoms_tmp.copy()
                    displacement = multiplier * step
                    displaced_atoms.positions[atom_idx, axis] += displacement
                    yield (displaced_atoms, dof, multiplier, step, _nx, _ny, _nz) if n_checked else (displaced_atoms, dof, multiplier, step)

        if output_option == 'xyz':
            for displacement_info in displacement_generator():
                displaced_atoms, dof, multiplier = displacement_info[0], displacement_info[1], displacement_info[2]
                xyz_file_name = f"dof_{dof}_disp_{multiplier}.xyz"
                xyz_file_path = join(custom_directory, xyz_file_name)
                try:
                    additional_info = f"Step: {step} Displacement_Number: {displacement_number} "
                    if self._charge is not None:
                        additional_info += f"Charge: {self._charge} "
                    if self._multiplicity is not None:
                        additional_info += f"Multiplicity: {self._multiplicity} "
                    if n_checked:
                        additional_info += f"Supercell_Repetitions [nx, ny, nz] = {[_nx, _ny, _nz]} "
                    if self._method_type in ["UNIT_CELL", "SUPERCELL"]:
                        additional_info += f"{"Cell" if self._method_type == "UNIT_CELL" and not _supercell else "Supercell"} parameters [a, b, c, alpha, beta, gamma]: {atoms_tmp.get_cell_lengths_and_angles().tolist()} "
                    write(xyz_file_path, displaced_atoms, comment=f"{additional_info}Created by SlothPy from File/Group '{self._slt_group._hdf5}/{self._slt_group._group_name}")
                except Exception as exc:
                    raise IOError(f"Failed to write XYZ file '{xyz_file_path}': {exc}")
            return None

        elif output_option == 'iterator':
            return (displacement_info for displacement_info in displacement_generator())

        elif output_option == 'slt':
            try:
                with File(self._slt_group._hdf5, 'a') as slt:
                    displacement_group = slt.create_group(slt_group_name)
                    displacement_group.attrs["Type"] = "DISPLACEMENTS_XYZ" if not n_checked else "DISPLACEMENTs_SUPERCELL"
                    displacement_group.attrs["Description"] = "Group containing displaced XYZ coordinates groups."
                    displacement_group.attrs["Displacement_Number"] = displacement_number
                    displacement_group.attrs["Step"] = step
                    displacement_group.attrs["Original_Group"] = self._slt_group._group_name
                    if n_checked:
                        displacement_group.attrs["Supercell_Repetitions"] = [_nx, _ny, _nz]
                for displacement_info in displacement_generator():
                    displaced_atoms, dof, multiplier = displacement_info[0], displacement_info[1], displacement_info[2]
                    subgroup_name = f"{slt_group_name}/dof_{dof}_disp_{multiplier}"
                    _xyz_to_slt(self._slt_group._hdf5, subgroup_name, displaced_atoms.get_chemical_symbols(), displaced_atoms.get_positions(), self._charge, self._multiplicity)
            except Exception as exc:
                raise SltFileError(self._slt_group._hdf5, exc, f"Failed to write displacement Group '{slt_group_name}' to the .slt file") from None
            return SltGroup(self._slt_group._hdf5, slt_group_name)


class SltUnitCell(SltXyz):
    _method_type = "UNIT_CELL"

    __slots__ = SltXyz.__slots__

    def __init__(self, slt_group: SltGroup) -> None:
        super().__init__(slt_group)
        self._atoms.set_cell(slt_group["CELL"][:])

    def cell_object(self):
        return self._atoms.get_cell()
    
    def supercell(self, nx: int, ny: int, nz: int, output_option: Literal["xyz", "slt"] = "slt", xyz_filepath: Optional[str] = None, slt_group_name: Optional[str] = None) -> Optional[SltGroup]:
        if self._method_type == "SUPERCELL":
            warnings.warn("You are trying to construct a supercell out of another supercell, creating a ... mega-cell with all parameters multiplied!")
        if output_option not in ['xyz', 'slt']:
            raise SltInputError(ValueError("Invalid output_option. Choose from 'xyz' or 'slt'.")) from None
        if output_option == 'xyz' and not xyz_filepath:
            raise SltInputError(ValueError("The xyz_filepath must be specified when output_option is 'xyz'.")) from None
        if output_option == 'slt' and not slt_group_name:
            raise SltInputError(ValueError("The slt_group_name must be specified when output_option is 'slt'.")) from None
        _check_n(nx, ny, nz)

        atoms: Atoms = self._atoms.repeat((nx, ny, nz))

        multiplicity = None
        if self._multiplicity:
            multiplicity = ((self._multiplicity - 1) * nx * ny * nz) + 1

        if output_option == "xyz":
            additional_info = f"Supercell parameters [a, b, c, alpha, beta, gamma]: {atoms.get_cell_lengths_and_angles()} "
            write(xyz_filepath, self._atoms, comment=f"{additional_info}Created by SlothPy from File/Group '{self._slt_group._hdf5}/{self._slt_group._group_name}'")
        else:
            _supercell_to_slt(self._hdf5, slt_group_name, atoms.get_chemical_symbols(), atoms.get_positions(), atoms.get_cell().array, nx, ny, nz, multiplicity)
            return SltGroup(self._hdf5, slt_group_name)

    def generate_supercell_finite_stencil_displacements(self, nx: int, ny: int, nz: int, displacement_number: int, step: float, output_option: Literal["xyz", "iterator", "slt"] = "xyz", custom_directory: Optional[str] = None, slt_group_name: Optional[str] = None, save_supercell_to_slt: Optional[str] = None) -> Optional[Iterator[Atoms]]:
        if self._method_type == "SUPERCELL":
            warnings.warn("You are trying to construct a supercell for finite displacements out of another supercell, creating a ... mega-cell with all parameters multiplied! If you wish to make displacements within a given supercell, use generate_finite_stencil_displacements instead.")
        if save_supercell_to_slt:
            self.supercell(nx, ny, nz, 'slt', slt_group_name=save_supercell_to_slt)
        return self.generate_finite_stencil_displacements(displacement_number, step, output_option, custom_directory, slt_group_name, True, nx, ny, nz)


class SltSuperCell(SltUnitCell):
    _method_type = "SUPERCELL"

    __slots__ = SltUnitCell.__slots__ + ["_nxnynz"]

    def __init__(self, slt_group: SltGroup) -> None:
        super().__init__(slt_group)
        self._atoms.set_cell(slt_group["CELL"][:])
        self._nxnynz = slt_group.attributes["Supercell_Repetitions"]

    def hessian_from_finite_displacements(self, dirpath: str, format: Literal["CP2K"], slt_group_name: str, displacement_number: int, step: float, accoustic_sum_rule: Literal["symmetric", "self_term", "without"] = "symmetric", born_charges: bool = False, force_files_suffix: Optional[str] = None, dipole_momenta_files_suffix: Optional[str] = None):

        if not isinstance(displacement_number, (int, int32, int64)) or displacement_number < 0:
            raise SltInputError(ValueError("The displacement_number must be a nonnegative integer.")) from None
        try:
            float(step)
        except Exception as exc:
            raise SltInputError(exc, "Invalid step provided.") from None

        dof_number = 3 * len(self._atoms) // self._nxnynz.prod()
        hessian, born_charges = _read_hessian_born_charges_from_dir(dirpath, format, dof_number, self._nxnynz[0], self._nxnynz[1], self._nxnynz[2], displacement_number, step, accoustic_sum_rule, born_charges, force_files_suffix, dipole_momenta_files_suffix)
        _hessian_to_slt(self._slt_group._hdf5, slt_group_name, self._atoms.get_chemical_symbols(), self._atoms.get_positions(), self._atoms.get_cell().array, self._nxnynz[0], self._nxnynz[1], self._nxnynz[2], self._multiplicity, hessian, born_charges)
        
        return SltGroup(self._slt_group._hdf5, slt_group_name)


class SltHessian(SltSuperCell):
    _method_type = "HESSIAN"

    __slots__ = SltXyz.__slots__ + ["_hessian"]

    def __init__(self, slt_group) -> None:
        super().__init__(slt_group)
        self._hessian = slt_group["HESSIAN"]

    def hessian(self) -> ndarray: #### From here add methods to SltGroup
        return self._hessian[:]
    
    def phonons(): #### This to slt group as delayed methoo with input parser
        pass


class SltPropertyCoordinateDerivative(SltXyz):
    _method_type = "PROPERTY_DERIVATIVE"

    __slots__ = SltXyz.__slots__

    def __init__(self, slt_group) -> None:
        super().__init__(slt_group)

 
class SltCrystalLattice(SltHessian):
    _method_type = "CRYSTAL_LATTICE"

    __slots__ = ["_hdf5", "_magnetic_centers", "_exchange_interactions", "_states", "_electric_dipole", "_magnetic_interactions", "_electric_interactions", "_mode", "_local_states"]

    def __init__(self, slt_group: SltGroup, states_cutoff=[0,0], rotation=None, hyperfine=None, local_states=True) -> None:
        self._hdf5: str = slt_group._hdf5
        self._magnetic_centers, self._exchange_interactions, self._states, self._electric_dipole, self._magnetic_interactions, self._electric_interactions, self._local_states = slt_group._retrieve_hamiltonian_dict(states_cutoff, rotation, hyperfine, local_states)
        self._mode: str = None # "eslpjm"


