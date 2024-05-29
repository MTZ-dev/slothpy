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
from h5py import File, Group, Dataset
from numpy import ndarray, errstate, array, diagonal, empty, float32, float64
from slothpy.core._slothpy_exceptions import slothpy_exc, SltCompError, SltSaveError, KeyError
from slothpy.core._config import settings
from slothpy._general_utilities._constants import RED, GREEN, BLUE, PURPLE, YELLOW, RESET
from slothpy.core._input_parser import validate_input
from slothpy._general_utilities._math_expresions import _magnetic_dipole_momenta_from_spins_angular_momenta, _total_angular_momenta_from_spins_angular_momenta
from slothpy._general_utilities._utils import _rotate_and_return_components, _return_components, _return_components_diag
from slothpy._general_utilities._io import _get_dataset_slt_dtype, _group_exists
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
    @validate_input("HAMILTONIAN")
    def energies(self):
        return self["STATES_ENERGIES"]
    
    e = energies
    
    @property
    @validate_input("HAMILTONIAN")
    def spins(self):
        return SltDatasetSLP(self._hdf5, f"{self._group_name}/SPINS", "S")
    
    s = spins
    
    @property
    @validate_input("HAMILTONIAN")
    def sx(self):
        return SltDatasetSLP(self._hdf5, f"{self._group_name}/SPINS", "S", 0)
    
    @property
    @validate_input("HAMILTONIAN")
    def sy(self):
        return SltDatasetSLP(self._hdf5, f"{self._group_name}/SPINS", "S", 1)
    
    @property
    @validate_input("HAMILTONIAN")
    def sz(self):
        return SltDatasetSLP(self._hdf5, f"{self._group_name}/SPINS", "S", 2)
    
    @property
    @validate_input("HAMILTONIAN")
    def angular_momenta(self):
        return SltDatasetSLP(self._hdf5, f"{self._group_name}/ANGULAR_MOMENTA", "L")
    
    l = angular_momenta
    
    @property
    @validate_input("HAMILTONIAN")
    def lx(self):
        return SltDatasetSLP(self._hdf5, f"{self._group_name}/ANGULAR_MOMENTA", "L", 0)
    
    @property
    @validate_input("HAMILTONIAN")
    def ly(self):
        return SltDatasetSLP(self._hdf5, f"{self._group_name}/ANGULAR_MOMENTA", "L", 1)
    
    @property
    @validate_input("HAMILTONIAN")
    def lz(self):
        return SltDatasetSLP(self._hdf5, f"{self._group_name}/ANGULAR_MOMENTA", "L", 2)
    
    @property
    @validate_input("HAMILTONIAN")
    def electric_dipole_momenta(self):
        return SltDatasetSLP(self._hdf5, f"{self._group_name}/ELECTRIC_DIPOLE_MOMENTA", "P")
    
    p = electric_dipole_momenta
    
    @property
    @validate_input("HAMILTONIAN")
    def px(self):
        return SltDatasetSLP(self._hdf5, f"{self._group_name}/ELECTRIC_DIPOLE_MOMENTA", "P", 0)
    
    @property
    @validate_input("HAMILTONIAN")
    def py(self):
        return SltDatasetSLP(self._hdf5, f"{self._group_name}/ELECTRIC_DIPOLE_MOMENTA", "P", 1)
    
    @property
    @validate_input("HAMILTONIAN")
    def pz(self):
        return SltDatasetSLP(self._hdf5, f"{self._group_name}/ELECTRIC_DIPOLE_MOMENTA", "P", 2)
    
    @property
    @validate_input("HAMILTONIAN")
    def total_angular_momenta(self):
        return SltDatasetJM(self._hdf5, f"{self._group_name}", "J")
    
    j = total_angular_momenta
    
    @property
    @validate_input("HAMILTONIAN")
    def jx(self):
        return SltDatasetJM(self._hdf5, f"{self._group_name}", "J", 0)
    
    @property
    @validate_input("HAMILTONIAN")
    def jy(self):
        return SltDatasetJM(self._hdf5, f"{self._group_name}", "J", 1)
    
    @property
    @validate_input("HAMILTONIAN")
    def jz(self):
        return SltDatasetJM(self._hdf5, f"{self._group_name}", "J", 2)
    
    @property
    @validate_input("HAMILTONIAN")
    def magnetic_dipole_momenta(self):
        return SltDatasetJM(self._hdf5, f"{self._group_name}", "M")
    
    m = magnetic_dipole_momenta
    
    @property
    @validate_input("HAMILTONIAN")
    def mx(self):
        return SltDatasetJM(self._hdf5, f"{self._group_name}", "M", 0)
    
    @property
    @validate_input("HAMILTONIAN")
    def my(self):
        return SltDatasetJM(self._hdf5, f"{self._group_name}", "M", 1)
    
    @property
    @validate_input("HAMILTONIAN")
    def mz(self):
        return SltDatasetJM(self._hdf5, f"{self._group_name}", "M", 2)

    @validate_input("HAMILTONIAN")
    def states_energies_cm_1(self, start_state=0, stop_state=0, slt_save=None) -> SltStatesEnergiesCm1:
        return SltStatesEnergiesCm1(self, start_state, stop_state, slt_save)
    
    @validate_input("HAMILTONIAN")
    def states_energies_au(self, start_state=0, stop_state=0, slt_save=None) -> SltStatesEnergiesAu:
        return SltStatesEnergiesAu(self, start_state, stop_state, slt_save)
    
    @validate_input("HAMILTONIAN")
    def spin_matrices(self, xyz='xyz', start_state=0, stop_state=0, rotation=None, slt_save=None):
        return SltSpinMatrices(self, xyz, start_state, stop_state, rotation, slt_save)
    
    @validate_input("HAMILTONIAN")
    def states_spins(self, xyz='xyz', start_state=0, stop_state=0, rotation=None, slt_save=None):
        return SltStatesSpin(self, xyz, start_state, stop_state, rotation, slt_save)
    
    @validate_input("HAMILTONIAN")
    def angular_momentum_matrices(self, xyz='xyz', start_state=0, stop_state=0, rotation=None, slt_save=None):
        return SltAngularMomentumMatrices(self, xyz, start_state, stop_state, rotation, slt_save)
    
    @validate_input("HAMILTONIAN")
    def states_angular_momenta(self, xyz='xyz', start_state=0, stop_state=0, rotation=None, slt_save=None):
        return SltStatesAngularMomenta(self, xyz, start_state, stop_state, rotation, slt_save)

    @validate_input("HAMILTONIAN")
    def electric_dipole_momentum_matrices(self, xyz='xyz', start_state=0, stop_state=0, rotation=None, slt_save=None):
        return SltElectricDipoleMomentumMatrices(self, xyz, start_state, stop_state, rotation, slt_save)
    
    @validate_input("HAMILTONIAN")
    def states_electric_dipole_momenta(self, xyz='xyz', start_state=0, stop_state=0, rotation=None, slt_save=None):
        if rotation is not None:
            states_electric_dipole_momenta = diagonal(self.electric_dipole_momenta[:, start_state:stop_state, start_state:stop_state].real, axis1=1, axis2=2).astype(settings.float, order="C")
            states_electric_dipole_momenta = _rotate_and_return_components(xyz, states_electric_dipole_momenta, rotation)
        else:
            states_electric_dipole_momenta = _return_components_diag(self.electric_dipole_momenta, self.px, self.py, self.pz, xyz, start_state, stop_state)   
        if slt_save is not None:
            new_group = SltGroup(self._hdf5, slt_save)
            new_group["STATES_ELECTRIC_DIPOLE_MOMENTA"] = states_electric_dipole_momenta
            new_group["STATES_ELECTRIC_DIPOLE_MOMENTA"].attributes["Description"] = f"{str(xyz).upper()}{' [(x-0, y-1, z-2), :]' if isinstance(xyz, str) and xyz == 'xyz' else ''} component{'s' if isinstance(xyz, str) and xyz == 'xyz' else ''} of the states' electric dipole momenta."
            new_group.attributes["Type"] = "STATES_ELECTRIC_DIPOLE_MOMENTA"
            new_group.attributes["Kind"] = f"{xyz.upper() if isinstance(xyz, str) else 'ORIENTATIONAL'}"
            new_group.attributes["States"] = states_electric_dipole_momenta.shape[1] if xyz == "xyz" or isinstance(xyz, ndarray) else states_electric_dipole_momenta.shape[0]
            new_group.attributes["Precision"] = settings.precision.upper()
            new_group.attributes["Description"] = f"States' expectation values of the electric dipole momentum from Group '{self._group_name}'."
            if rotation is not None:
                new_group["ROTATION"] = rotation
                new_group["ROTATION"].attributes["Description"] = "Rotation used to rotate the electric dipole momentum components."

        return states_electric_dipole_momenta

    @validate_input("HAMILTONIAN")
    def total_angular_momentum_matrices(self, xyz='xyz', start_state=0, stop_state=0, rotation=None, slt_save=None):
        if rotation is not None:
            total_angular_momentum_matrices = self.total_angular_momenta[:, start_state:stop_state, start_state:stop_state]
            total_angular_momentum_matrices = _rotate_and_return_components(xyz, total_angular_momentum_matrices, rotation)
        else:
            total_angular_momentum_matrices = _return_components(self.total_angular_momenta, self.jx, self.jy, self.jz, xyz, start_state, stop_state)
        if slt_save is not None:
            new_group = SltGroup(self._hdf5, slt_save)
            new_group["TOTAL_ANGULAR_MOMENTUM_MATRICES"] = total_angular_momentum_matrices
            new_group["TOTAL_ANGULAR_MOMENTUM_MATRICES"].attributes["Description"] = f"{str(xyz).upper()}{' [(x-0, y-1, z-2), :, :]' if isinstance(xyz, str) and xyz == 'xyz' else ''} component{'s' if isinstance(xyz, str) and xyz == 'xyz' else ''} of the total angular momentum."
            new_group.attributes["Type"] = "TOTAL_ANGULAR_MOMENTA"
            new_group.attributes["Kind"] = f"{xyz.upper() if isinstance(xyz, str) else 'ORIENTATIONAL'}"
            new_group.attributes["States"] = total_angular_momentum_matrices.shape[1]
            new_group.attributes["Precision"] = settings.precision.upper()
            new_group.attributes["Description"] = f"Total angular momentum matrices from Group '{self._group_name}'."
            if rotation is not None:
                new_group["ROTATION"] = rotation
                new_group["ROTATION"].attributes["Description"] = "Rotation used to rotate the total angular momentum components."

        return total_angular_momentum_matrices

    @validate_input("HAMILTONIAN")
    def states_total_angular_momentum(self, xyz='xyz', start_state=0, stop_state=0, rotation=None, slt_save=None):
        if rotation is not None:
            states_total_angular_momenta = diagonal(self.total_angular_momenta[:, start_state:stop_state, start_state:stop_state].real, axis1=1, axis2=2).astype(settings.float, order="C")
            states_total_angular_momenta = _rotate_and_return_components(xyz, states_total_angular_momenta, rotation)
        else:
            states_total_angular_momenta = _return_components(self.total_angular_momenta, self.jx, self.jy, self.jz, xyz, start_state, stop_state)
        if slt_save is not None:
            new_group = SltGroup(self._hdf5, slt_save)
            new_group["STATES_TOTAL_ANGULAR_MOMENTA"] = states_total_angular_momenta
            new_group["STATES_TOTAL_ANGULAR_MOMENTA"].attributes["Description"] = f"{str(xyz).upper()}{' [(x-0, y-1, z-2), :]' if isinstance(xyz, str) and xyz == 'xyz' else ''} component{'s' if isinstance(xyz, str) and xyz == 'xyz' else ''} of the states' total angular momenta."
            new_group.attributes["Type"] = "STATES_TOTAL_ANGULAR_MOMENTA"
            new_group.attributes["Kind"] = f"{xyz.upper() if isinstance(xyz, str) else 'ORIENTATIONAL'}"
            new_group.attributes["States"] = states_total_angular_momenta.shape[1] if xyz == "xyz" or isinstance(xyz, ndarray) else states_total_angular_momenta.shape[0]
            new_group.attributes["Precision"] = settings.precision.upper()
            new_group.attributes["Description"] = f"States' expectation values of the total angular momentum from Group '{self._group_name}'."
            if rotation is not None:
                new_group["ROTATION"] = rotation
                new_group["ROTATION"].attributes["Description"] = "Rotation used to rotate the total angular momentum components."

        return states_total_angular_momenta

    @validate_input("HAMILTONIAN")
    def magnetic_dipole_momentum_matrices(self, xyz='xyz', start_state=0, stop_state=0, rotation=None, slt_save=None):
        if rotation is not None:
            magnetic_dipole_momentum_matrices = self.magnetic_dipole_momenta[:, start_state:stop_state, start_state:stop_state]
            magnetic_dipole_momentum_matrices = _rotate_and_return_components(xyz, magnetic_dipole_momentum_matrices, rotation)
        else:
            magnetic_dipole_momentum_matrices = _return_components(self.magnetic_dipole_momenta, self.mx, self.my, self.mz, xyz, start_state, stop_state)
        if slt_save is not None:
            new_group = SltGroup(self._hdf5, slt_save)
            new_group["MAGNETIC_DIPOLE_MOMENTUM_MATRICES"] = magnetic_dipole_momentum_matrices
            new_group["MAGNETIC_DIPOLE_MOMENTUM_MATRICES"].attributes["Description"] = f"{str(xyz).upper()}{' [(x-0, y-1, z-2), :, :]' if isinstance(xyz, str) and xyz == 'xyz' else ''} component{'s' if isinstance(xyz, str) and xyz == 'xyz' else ''} of the magnetic dipole momentum."
            new_group.attributes["Type"] = "MAGNETIC_DIPOLE_MOMENTA"
            new_group.attributes["Kind"] = f"{xyz.upper() if isinstance(xyz, str) else 'ORIENTATIONAL'}"
            new_group.attributes["States"] = magnetic_dipole_momentum_matrices.shape[1]
            new_group.attributes["Precision"] = settings.precision.upper()
            new_group.attributes["Description"] = f"Total magnetic dipole matrices from Group '{self._group_name}'."
            if rotation is not None:
                new_group["ROTATION"] = rotation
                new_group["ROTATION"].attributes["Description"] = "Rotation used to rotate the magnetic dipole momentum components."

        return magnetic_dipole_momentum_matrices

    @validate_input("HAMILTONIAN")
    def states_magnetic_dipole_momenta(self, xyz='xyz', start_state=0, stop_state=0, rotation=None, slt_save=None):
        if rotation is not None:
            states_magnetic_dipole_momenta = diagonal(self.magnetic_dipole_momenta[:, start_state:stop_state, start_state:stop_state].real, axis1=1, axis2=2).astype(settings.float, order="C")
            states_magnetic_dipole_momenta = _rotate_and_return_components(xyz, states_magnetic_dipole_momenta, rotation)
        else:
            states_magnetic_dipole_momenta = _return_components_diag(self.magnetic_dipole_momenta, self.mx, self.my, self.mz, xyz, start_state, stop_state) 
        if slt_save is not None:
            new_group = SltGroup(self._hdf5, slt_save)
            new_group["STATES_MAGNETIC_DIPOLE_MOMENTA"] = states_magnetic_dipole_momenta
            new_group["STATES_MAGNETIC_DIPOLE_MOMENTA"].attributes["Description"] = f"{str(xyz).upper()}{' [(x-0, y-1, z-2), :]' if isinstance(xyz, str) and xyz == 'xyz' else ''} component{'s' if isinstance(xyz, str) and xyz == 'xyz' else ''} of the states' magnetic dipole momenta."
            new_group.attributes["Type"] = "STATES_MAGNETIC_DIPOLE_MOMENTA"
            new_group.attributes["Kind"] = f"{xyz.upper() if isinstance(xyz, str) else 'ORIENTATIONAL'}"
            new_group.attributes["States"] = states_magnetic_dipole_momenta.shape[1] if xyz == "xyz" or isinstance(xyz, ndarray) else states_magnetic_dipole_momenta.shape[0]
            new_group.attributes["Precision"] = settings.precision.upper()
            new_group.attributes["Description"] = f"States' expectation values of the magnetic dipole momentum from Group '{self._group_name}'."
            if rotation is not None:
                new_group["ROTATION"] = rotation
                new_group["ROTATION"].attributes["Description"] = "Rotation used to rotate the magnetic dipole momentum components."

        return states_magnetic_dipole_momenta
    
    @validate_input("HAMILTONIAN")
    def zeeman_splitting(
        self,
        magnetic_fields: ndarray[Union[float32, float64]],
        orientations: ndarray[Union[float32, float64]],
        states_cutoff: int = 0,
        number_of_states: int = 0,
        number_cpu: int = None,
        number_threads: int = None,
        slt_save: str = None,
        autotune: bool = False,
    ) -> SltZeemanSplitting:
        return SltZeemanSplitting(self, magnetic_fields, orientations, states_cutoff, number_of_states, number_cpu, number_threads, autotune, slt_save)


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
            with errstate(all='ignore'):
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
            with errstate(all='ignore'):
                if self._jm == "J":
                    return _total_angular_momenta_from_spins_angular_momenta(diag_s.astype(settings.float), diag_l.astype(settings.float))
                elif self._jm == "M":
                    return  _magnetic_dipole_momenta_from_spins_angular_momenta(diag_s.astype(settings.float), diag_l.astype(settings.float))
                else:
                    raise ValueError("The only supported options are 'J' for total angular momenta or 'M' for magnetic dipole momenta.")
