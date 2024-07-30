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

from typing import Union
import warnings
from ast import literal_eval

from h5py import File, Group, Dataset
from numpy import ndarray, array, empty, float32, float64, tensordot, abs
from numpy.exceptions import ComplexWarning
from numpy.linalg import norm
warnings.filterwarnings("ignore", category=ComplexWarning)

from slothpy.core._slothpy_exceptions import slothpy_exc, KeyError
from slothpy.core._config import settings
from slothpy._general_utilities._constants import RED, GREEN, BLUE, PURPLE, YELLOW, RESET
from slothpy.core._input_parser import validate_input
from slothpy._general_utilities._math_expresions import _magnetic_dipole_momenta_from_spins_angular_momenta, _total_angular_momenta_from_spins_angular_momenta
from slothpy._general_utilities._io import _get_dataset_slt_dtype, _group_exists
from slothpy._general_utilities._constants import MU_T
from slothpy._general_utilities._direct_product_space import _kron_mult
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
        
    @property
    def attr(self):
        """
        Property to mimic h5py's attribute access convention.
        """
        return self.attributes
    
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
    
    def _retrieve_hamiltonian_dict(self, states_cutoff=[0,0], rotation=None, coordinates=None, hyperfine=None):
        states = self.attributes["States"]
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
                exchange_interactions = load_dict_from_group(group, "EXCHANGE_INTERACTIONS")
        else:
            magnetic_centers = {0:(self._group_name, (states_cutoff[0],0,states_cutoff[1]), rotation, coordinates, hyperfine)}
            exchange_interactions = None
        
        return magnetic_centers, exchange_interactions, states
    
    @validate_input("HAMILTONIAN", only_hamiltonian_check=True)
    def _hamiltonian_from_slt_group(self, states_cutoff=[0,0], rotation=None, hyperfine=None):
            return SltHamiltonian(self, states_cutoff, rotation, hyperfine)
    
    @validate_input("HAMILTONIAN")
    def zeeman_splitting(
        self,
        magnetic_fields: ndarray[Union[float32, float64]],
        orientations: ndarray[Union[float32, float64]],
        number_of_states: int = 0,
        states_cutoff: int = [0, "auto"],
        rotation: ndarray = None,
        hyperfine: dict = None,
        number_cpu: int = None,
        number_threads: int = None,
        slt_save: str = None,
        autotune: bool = False,
    ) -> SltZeemanSplitting:
        return SltZeemanSplitting(self, magnetic_fields, orientations, number_of_states, states_cutoff, rotation, hyperfine, number_cpu, number_threads, autotune, slt_save)
    
    @validate_input("HAMILTONIAN")
    def magnetisation(
        self,
        magnetic_fields: ndarray[Union[float32, float64]],
        orientations: ndarray[Union[float32, float64]],
        temperatures: ndarray[Union[float32, float64]],
        states_cutoff: int = [0, "auto"],
        rotation: ndarray = None,
        hyperfine: dict = None,
        number_cpu: int = None,
        number_threads: int = None,
        slt_save: str = None,
        autotune: bool = False,
    ) -> SltMagnetisation:
        return SltMagnetisation(self, magnetic_fields, orientations, temperatures, states_cutoff, rotation, hyperfine, number_cpu, number_threads, autotune, slt_save)


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
    def attr(self):
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
            

class SltHamiltonian():

    __slots__ = ["_hdf5", "_magnetic_centers", "_exchange_interactions", "_states", "_mode"]

    def __init__(self, slt_group: SltGroup, states_cutoff=[0,0], rotation=None, hyperfine=None) -> None:
        self._hdf5: str = slt_group._hdf5
        self._magnetic_centers, self._exchange_interactions, self._states = slt_group._retrieve_hamiltonian_dict(states_cutoff, rotation, hyperfine)
        self._mode: str = None # "eslpjm"
    
    @property
    def e(self):
        data = []
        for center in self._magnetic_centers.values():
            data.append(SltGroup(self._hdf5, center[0]).states_energies_au(stop_state=center[1][0]).eval())
        return data

    @property
    def s(self):
        data = []
        for center in self._magnetic_centers.values():
            data.append(SltGroup(self._hdf5, center[0]).spin_matrices(stop_state=center[1][0], rotation=center[2]).eval().conj()) #return conj of hermitian matrix in c-order to prepare it already for lapack f-order using .T
        return data
        
    @property
    def l(self):
        data = []
        for center in self._magnetic_centers.values():
            data.append(SltGroup(self._hdf5, center[0]).angular_momentum_matrices(stop_state=center[1][0], rotation=center[2]).eval().conj())
        return data
    
    @property
    def p(self):
        data = []
        for center in self._magnetic_centers.values():
            data.append(SltGroup(self._hdf5, center[0]).electric_dipole_momentum_matrices(stop_state=center[1][0], rotation=center[2]).eval().conj())
        return data

    @property
    def j(self):
        data = []
        for center in self._magnetic_centers.values():
            data.append(SltGroup(self._hdf5, center[0]).total_angular_momentum_matrices(stop_state=center[1][0], rotation=center[2]).eval().conj())
        return data

    @property
    def m(self):
        data = []
        for center in self._magnetic_centers.values():
            data.append(SltGroup(self._hdf5, center[0]).magnetic_dipole_momentum_matrices(stop_state=center[1][0], rotation=center[2]).eval().conj())
        return data
    
    @property
    def interaction_matrix(self): # you will have to move implementation of this somewhere else not to import linalg and numpy etc. and tha same with everything because slt file is used everywhere 
        result = zeros((self._states, self._states), dtype=settings.complex)
        n = len(self._magnetic_centers.keys())
        if not any(value[3] is None for value in self._magnetic_centers.values()):
            dipole_magnetic_momenta_dict = {key: SltGroup(self._hdf5, self._magnetic_centers[key][0]).magnetic_dipole_momentum_matrices(stop_state=self._magnetic_centers[key][1][1], rotation=self._magnetic_centers[key][2]).eval().conj() for key in self._magnetic_centers.keys()}
            for key1 in self._magnetic_centers.keys():
                for key2 in range(key1+1, n):
                    r_vec = self._magnetic_centers[key1][3] - self._magnetic_centers[key2][3]
                    r_norm = norm(r_vec)
                    if r_norm <= 1e-2:
                        raise ValueError("Magnetic centers are closer than 0.01 Angstrom. Please double-check the SlothPy Hamiltonian dictionary. Quitting here.")
                    coeff = MU_T / r_norm ** 3
                    r_vec = r_vec / r_norm
                    op1 = tensordot(dipole_magnetic_momenta_dict[key1], - 3. * coeff * r_vec ,axes=(0, 0))
                    op2 = tensordot(dipole_magnetic_momenta_dict[key2], r_vec, axes=(0, 0))
                    ops = [op1 if k == key1 else op2 if k == key2 else dipole_magnetic_momenta_dict[k].shape[1] for k in range(n)]
                    result += _kron_mult(ops)
                    for i in range(3):
                        ops[key1] = coeff * dipole_magnetic_momenta_dict[key1][i]
                        ops[key2] = dipole_magnetic_momenta_dict[key2][i]
                        result += _kron_mult(ops)
        
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

        #TODO: hyperfine interactions and different types of interactions (J,L???)

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
        return (self._mode, info_list)


