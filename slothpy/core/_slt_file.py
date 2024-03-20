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

import warnings
from typing import Literal
from h5py import File, Group, Dataset
from numpy import array, ComplexWarning, diagonal
from slothpy.core._slothpy_exceptions import slothpy_exc, SltFileError, SltReadError, SltCompError
from slothpy.core._config import settings
from slothpy._general_utilities._constants import RED, GREEN, BLUE, PURPLE, YELLOW, RESET, H_CM_1
from slothpy.core._input_parser import validate_input
from slothpy._general_utilities._math_expresions import _magnetic_dipole_momenta_from_spins_angular_momenta, _total_angular_momenta_from_spins_angular_momenta
from slothpy._angular_momentum._rotation import _rotate_vector_operator

class SltAttributes:
    def __init__(self, hdf5_file_path, item_path):
        self._hdf5 = hdf5_file_path
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
    def __init__(self, hdf5_file_path, group_path, exists=True):
        self._hdf5 = hdf5_file_path
        self._group_path = group_path
        self._exists = exists
        self.attributes = SltAttributes(hdf5_file_path, group_path)

    @slothpy_exc("SltFileError")
    def __getitem__(self, key):
        full_path = f"{self._group_path}/{key}"
        with File(self._hdf5, 'r') as file:
            if full_path in file:
                item = file[full_path]
                if isinstance(item, Dataset):
                    return SltDataset(self._hdf5, full_path)
                elif isinstance(item, Group):
                    raise KeyError(f"Hierarchy only up to {BLUE}Group{RESET}/{PURPLE}Dataset{RESET} or standalone {PURPLE}Datasets{RESET} are supported in .slt files.")
            else:
                raise KeyError(f"{PURPLE}Dataset{RESET} '{key}' doesn't exist in the {BLUE}Group{RESET} '{self._group_path}'.")

    @slothpy_exc("SltSaveError")        
    def __setitem__(self, key, value):
        with File(self._hdf5, 'r+') as file:
            group = file.require_group(self._group_path)
            self._exists = True
            if key in group:
                raise KeyError(f"{PURPLE}Dataset{RESET} '{key}' already exists within the {BLUE}Group{RESET} '{self._group_path}'. Delete it manually to ensure your data safety.")
            group.create_dataset(key, data=array(value))

    @slothpy_exc("SltFileError")
    def __repr__(self): 
        if self._exists:
            return f"<{BLUE}SltGroup{RESET} '{self._group_path}' in {GREEN}File{RESET} '{self._hdf5}'.>"
        else:
            raise RuntimeError(f"This is a {BLUE}Proxy Group{RESET} and it does not exist in the .slt file yet. Initialize it by setting dataset within it - group['new_dataset'] = value.")

    @slothpy_exc("SltFileError")
    def __str__(self):
        if self._exists:
            with File(self._hdf5, 'r+') as file:
                item = file[self._group_path]
                representation = f"{RED}Group{RESET}: {BLUE}{self._group_path}{RESET} from File: {GREEN}{self._hdf5}{RESET}"
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
            group = file[self._group_path]
            if key not in group:
                raise KeyError(f"{PURPLE}Dataset{RESET} '{key}' does not exist in the {BLUE}Group{RESET} '{self._group_path}'.")
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
        return self["ENERGIES"]
    
    @property
    @validate_input("HAMILTONIAN")
    def spins(self):
        return self["SPINS"]
    
    @property
    @validate_input("HAMILTONIAN")
    def sx(self):
        return SltDatasetSLP(self._hdf5, f"{self._group_path}/SPINS", "S", 0)
    
    @property
    @validate_input("HAMILTONIAN")
    def sy(self):
        return SltDatasetSLP(self._hdf5, f"{self._group_path}/SPINS", "S", 1)
    
    @property
    @validate_input("HAMILTONIAN")
    def sz(self):
        return SltDatasetSLP(self._hdf5, f"{self._group_path}/SPINS", "S", 2)
    
    @property
    @validate_input("HAMILTONIAN")
    def angular_momenta(self):
        return self["ANGULAR_MOMENTA"]
    
    @property
    @validate_input("HAMILTONIAN")
    def lx(self):
        return SltDatasetSLP(self._hdf5, f"{self._group_path}/ANGULAR_MOMENTA", "L", 0)
    
    @property
    @validate_input("HAMILTONIAN")
    def ly(self):
        return SltDatasetSLP(self._hdf5, f"{self._group_path}/ANGULAR_MOMENTA", "L", 1)
    
    @property
    @validate_input("HAMILTONIAN")
    def lz(self):
        return SltDatasetSLP(self._hdf5, f"{self._group_path}/ANGULAR_MOMENTA", "L", 2)
    
    @property
    @validate_input("HAMILTONIAN")
    def electric_dipole_momenta(self):
        return self["ELECTRIC_DIPOLE_MOMENTA"]
    
    @property
    @validate_input("HAMILTONIAN")
    def px(self):
        return SltDatasetSLP(self._hdf5, f"{self._group_path}/ELECTRIC_DIPOLE_MOMENTA", "P", 0)
    
    @property
    @validate_input("HAMILTONIAN")
    def py(self):
        return SltDatasetSLP(self._hdf5, f"{self._group_path}/ELECTRIC_DIPOLE_MOMENTA", "P", 1)
    
    @property
    @validate_input("HAMILTONIAN")
    def pz(self):
        return SltDatasetSLP(self._hdf5, f"{self._group_path}/ELECTRIC_DIPOLE_MOMENTA", "P", 2)
    
    @property
    @validate_input("HAMILTONIAN")
    def total_angular_momenta(self):
        return SltDatasetJM(self._hdf5, f"{self._group_path}", "J")
    
    @property
    @validate_input("HAMILTONIAN")
    def jx(self):
        return SltDatasetJM(self._hdf5, f"{self._group_path}", "J", 0)
    
    @property
    @validate_input("HAMILTONIAN")
    def jy(self):
        return SltDatasetJM(self._hdf5, f"{self._group_path}", "J", 1)
    
    @property
    @validate_input("HAMILTONIAN")
    def jz(self):
        return SltDatasetJM(self._hdf5, f"{self._group_path}", "J", 2)
    
    @property
    @validate_input("HAMILTONIAN")
    def magnetic_dipole_momenta(self):
        return SltDatasetJM(self._hdf5, f"{self._group_path}", "M")
    
    @property
    @validate_input("HAMILTONIAN")
    def mx(self):
        return SltDatasetJM(self._hdf5, f"{self._group_path}", "M", 0)
    
    @property
    @validate_input("HAMILTONIAN")
    def my(self):
        return SltDatasetJM(self._hdf5, f"{self._group_path}", "M", 1)
    
    @property
    @validate_input("HAMILTONIAN")
    def mz(self):
        return SltDatasetJM(self._hdf5, f"{self._group_path}", "M", 2)

    @validate_input("HAMILTONIAN")   
    def states_energies_cm_1(self, start_state=0, stop_state=0, slt_save=None):
        try:
            energies_cm_1 = self.energies[start_state:stop_state] * H_CM_1
        except Exception as exc:
            raise SltCompError(self._hdf5, exc, f"Failed to compute energies in cm-1 from {BLUE}Group{RESET}: '{self._hdf5}'.")
        if slt_save is not None:
            new_group = SltGroup(self._hdf5, slt_save, exists=False)
            new_group["STATES_ENERGIES_CM_1"] = energies_cm_1
            new_group["STATES_ENERGIES_CM_1"].attributes["Description"] = "States' energies in cm-1."
            new_group.attributes["Type"] = "ENERGIES"
            new_group.attributes["Kind"] = "CM_1"
            new_group.attributes["States"] = energies_cm_1.shape[0]
            new_group.attributes["Precision"] = settings.precision.upper()
            new_group.attributes["Description"] = "States' energies in cm-1."
        return energies_cm_1
    
    @validate_input("HAMILTONIAN")
    def states_energies_au(self, start_state=0, stop_state=0, slt_save=None):
        try:
            energies_au = self.energies[start_state:stop_state]
        except Exception as exc:
            raise SltCompError(self._hdf5, exc, f"Failed to compute energies in a.u. from {BLUE}Group{RESET}: '{self._hdf5}'.")
        if slt_save is not None:
            new_group = SltGroup(self._hdf5, slt_save, exists=False)
            new_group["STATES_ENERGIES_AU"] = energies_au
            new_group["STATES_ENERGIES_AU"].attributes["Description"] = "States' energies in a.u.."
            new_group.attributes["Type"] = "ENERGIES"
            new_group.attributes["Kind"] = "AU"
            new_group.attributes["States"] = energies_au.shape[0]
            new_group.attributes["Precision"] = settings.precision.upper()
            new_group.attributes["Description"] = "States' energies in a.u.."
        return energies_au
    
    @validate_input("HAMILTONIAN")
    def spin_matrices(self, xyz='xyz', start_state=0, stop_state=0, rotation=None, slt_save=None):
        if rotation is not None:
            spin_matrices = self.spins[:, start_state:stop_state, start_state:stop_state]
            spin_matrices = _rotate_vector_operator(spin_matrices, rotation)
            match xyz:
                case "x":
                    spin_matrices =  spin_matrices[0]
                case "y":
                    spin_matrices = spin_matrices[1]
                case "z":
                    spin_matrices = spin_matrices[2]
        else:
            match xyz:
                case "xyz":
                    spin_matrices = self.spins[:, start_state:stop_state, start_state:stop_state]
                case "x":
                    spin_matrices = self.sx[start_state:stop_state, start_state:stop_state]
                case "y":
                    spin_matrices = self.sy[start_state:stop_state, start_state:stop_state]
                case "z":
                    spin_matrices = self.sz[start_state:stop_state, start_state:stop_state]
        
        if slt_save is not None:
            new_group = SltGroup(self._hdf5, slt_save, exists=False)
            new_group["SPIN_MATRICES"] = spin_matrices
            new_group["SPIN_MATRICES"].attributes["Description"] = f"{xyz.upper()}{" [(x-0, y-1, z-2), :, :]" if xyz == 'xyz' else ''} component{'s' if xyz == 'xyz' else ''} of the spin."
            new_group.attributes["Type"] = "SPINS"
            new_group.attributes["Kind"] = f"{xyz.upper()}"
            new_group.attributes["States"] = spin_matrices.shape[1] if xyz == "xyz" else spin_matrices.shape[0]
            new_group.attributes["Precision"] = settings.precision.upper()
            new_group.attributes["Description"] = f"Spin matrices from Group '{self._group_path}'."
            if rotation is not None:
                new_group["ROTATION"] = rotation
                new_group["ROTATION"].attributes["Description"] = "Rotation used to rotate the spin components."

        return spin_matrices
    
    @validate_input("HAMILTONIAN")
    def states_spins(self, xyz='xyz', start_state=0, stop_state=0, rotation=None, slt_save=None):
        if rotation is not None:
            states_spins = diagonal(self.spins[:, start_state:stop_state, start_state:stop_state], axis1=1, axis2=2).astype(settings.float, order="C")
            states_spins = _rotate_vector_operator(states_spins, rotation)
            match xyz:
                case "x":
                    states_spins =  states_spins[0]
                case "y":
                    states_spins = states_spins[1]
                case "z":
                    states_spins = states_spins[2]
        else:
            match xyz:
                case "xyz":
                    states_spins = diagonal(self.spins[:, start_state:stop_state, start_state:stop_state], axis1=1, axis2=2).astype(settings.float, order="C")
                case "x":
                    states_spins = diagonal(self.sx[start_state:stop_state, start_state:stop_state]).astype(settings.float, order="C")
                case "y":
                    states_spins = diagonal(self.sy[start_state:stop_state, start_state:stop_state]).astype(settings.float, order="C")
                case "z":
                    states_spins = diagonal(self.sz[start_state:stop_state, start_state:stop_state]).astype(settings.float, order="C")
        
        if slt_save is not None:
            new_group = SltGroup(self._hdf5, slt_save, exists=False)
            new_group["STATES_SPINS"] = states_spins
            new_group["STATES_SPINS"].attributes["Description"] = f"{xyz.upper()}{" [(x-0, y-1, z-2), :]" if xyz == 'xyz' else ''} component{'s' if xyz == 'xyz' else ''} of the states's spins."
            new_group.attributes["Type"] = "STATES_SPINS"
            new_group.attributes["Kind"] = f"{xyz.upper()}"
            new_group.attributes["States"] = states_spins.shape[1] if xyz == "xyz" else states_spins.shape[0]
            new_group.attributes["Precision"] = settings.precision.upper()
            new_group.attributes["Description"] = f"States' expectation values of the spin from Group '{self._group_path}'."
            if rotation is not None:
                new_group["ROTATION"] = rotation
                new_group["ROTATION"].attributes["Description"] = "Rotation used to rotate the spin components."

        return states_spins
    
    @validate_input("HAMILTONIAN")
    def angular_momentum_matrices(self, xyz='xyz', start_state=0, stop_state=0, rotation=None, slt_save=None):
        if rotation is not None:
            angular_momentum_matrices = self.angular_momenta[:, start_state:stop_state, start_state:stop_state]
            angular_momentum_matrices = _rotate_vector_operator(angular_momentum_matrices, rotation)
            match xyz:
                case "x":
                    angular_momentum_matrices =  angular_momentum_matrices[0]
                case "y":
                    angular_momentum_matrices = angular_momentum_matrices[1]
                case "z":
                    angular_momentum_matrices = angular_momentum_matrices[2]
        else:
            match xyz:
                case "xyz":
                    angular_momentum_matrices = self.angular_momenta[:, start_state:stop_state, start_state:stop_state]
                case "x":
                    angular_momentum_matrices = self.lx[start_state:stop_state, start_state:stop_state]
                case "y":
                    angular_momentum_matrices = self.ly[start_state:stop_state, start_state:stop_state]
                case "z":
                    angular_momentum_matrices = self.lz[start_state:stop_state, start_state:stop_state]
        
        if slt_save is not None:
            new_group = SltGroup(self._hdf5, slt_save, exists=False)
            new_group["ANGULAR_MOMENTUM_MATRICES"] = angular_momentum_matrices
            new_group["ANGULAR_MOMENTUM_MATRICES"].attributes["Description"] = f"{xyz.upper()}{" [(x-0, y-1, z-2), :, :]" if xyz == 'xyz' else ''} component{'s' if xyz == 'xyz' else ''} of the angular momentum."
            new_group.attributes["Type"] = "ANGULAR_MOMENTA"
            new_group.attributes["Kind"] = f"{xyz.upper()}"
            new_group.attributes["States"] = angular_momentum_matrices.shape[1] if xyz == "xyz" else angular_momentum_matrices.shape[0]
            new_group.attributes["Precision"] = settings.precision.upper()
            new_group.attributes["Description"] = f"Angular momentum matrices from Group '{self._group_path}'."
            if rotation is not None:
                new_group["ROTATION"] = rotation
                new_group["ROTATION"].attributes["Description"] = "Rotation used to rotate the angular momentum components."

        return angular_momentum_matrices
    
    @validate_input("HAMILTONIAN")
    def states_angular_momenta(self, xyz='xyz', start_state=0, stop_state=0, rotation=None, slt_save=None):
        if rotation is not None:
            states_angular_momenta = diagonal(self.angular_momenta[:, start_state:stop_state, start_state:stop_state], axis1=1, axis2=2).astype(settings.float, order="C")
            states_angular_momenta = _rotate_vector_operator(states_angular_momenta, rotation)
            match xyz:
                case "x":
                    states_angular_momenta =  states_angular_momenta[0]
                case "y":
                    states_angular_momenta = states_angular_momenta[1]
                case "z":
                    states_angular_momenta = states_angular_momenta[2]
        else:
            match xyz:
                case "xyz":
                    states_angular_momenta = diagonal(self.angular_momenta[:, start_state:stop_state, start_state:stop_state], axis1=1, axis2=2).astype(settings.float, order="C")
                case "x":
                    states_angular_momenta = diagonal(self.lx[start_state:stop_state, start_state:stop_state]).astype(settings.float, order="C")
                case "y":
                    states_angular_momenta = diagonal(self.ly[start_state:stop_state, start_state:stop_state]).astype(settings.float, order="C")
                case "z":
                    states_angular_momenta = diagonal(self.lz[start_state:stop_state, start_state:stop_state]).astype(settings.float, order="C")
        
        if slt_save is not None:
            new_group = SltGroup(self._hdf5, slt_save, exists=False)
            new_group["STATES_ANGULAR_MOMENTA"] = states_angular_momenta
            new_group["STATES_ANGULAR_MOMENTA"].attributes["Description"] = f"{xyz.upper()}{" [(x-0, y-1, z-2), :]" if xyz == 'xyz' else ''} component{'s' if xyz == 'xyz' else ''} of the states's angular momenta."
            new_group.attributes["Type"] = "STATES_ANGULAR_MOMENTA"
            new_group.attributes["Kind"] = f"{xyz.upper()}"
            new_group.attributes["States"] = states_angular_momenta.shape[1] if xyz == "xyz" else states_angular_momenta.shape[0]
            new_group.attributes["Precision"] = settings.precision.upper()
            new_group.attributes["Description"] = f"States' expectation values of the angular momentum from Group '{self._group_path}'."
            if rotation is not None:
                new_group["ROTATION"] = rotation
                new_group["ROTATION"].attributes["Description"] = "Rotation used to rotate the angular momentum components."

        return states_angular_momenta

    @validate_input("HAMILTONIAN")
    def electric_dipole_momentum_matrices(self, xyz='xyz', start_state=0, stop_state=0, rotation=None, slt_save=None):
        if rotation is not None:
            electric_dipole_momentum_matrices = self.electric_dipole_momenta[:, start_state:stop_state, start_state:stop_state]
            electric_dipole_momentum_matrices = _rotate_vector_operator(electric_dipole_momentum_matrices, rotation)
            match xyz:
                case "x":
                    electric_dipole_momentum_matrices =  electric_dipole_momentum_matrices[0]
                case "y":
                    electric_dipole_momentum_matrices = electric_dipole_momentum_matrices[1]
                case "z":
                    electric_dipole_momentum_matrices = electric_dipole_momentum_matrices[2]
        else:
            match xyz:
                case "xyz":
                    electric_dipole_momentum_matrices = self.electric_dipole_momenta[:, start_state:stop_state, start_state:stop_state]
                case "x":
                    electric_dipole_momentum_matrices = self.px[start_state:stop_state, start_state:stop_state]
                case "y":
                    electric_dipole_momentum_matrices = self.py[start_state:stop_state, start_state:stop_state]
                case "z":
                    electric_dipole_momentum_matrices = self.pz[start_state:stop_state, start_state:stop_state]
        
        if slt_save is not None:
            new_group = SltGroup(self._hdf5, slt_save, exists=False)
            new_group["ELECTRIC_DIPOLE_MOMENTUM_MATRICES"] = electric_dipole_momentum_matrices
            new_group["ELECTRIC_DIPOLE_MOMENTUM_MATRICES"].attributes["Description"] = f"{xyz.upper()}{" [(x-0, y-1, z-2), :, :]" if xyz == 'xyz' else ''} component{'s' if xyz == 'xyz' else ''} of the electric dipole momentum."
            new_group.attributes["Type"] = "ELECTRIC_DIPOLE_MOMENTA"
            new_group.attributes["Kind"] = f"{xyz.upper()}"
            new_group.attributes["States"] = electric_dipole_momentum_matrices.shape[1] if xyz == "xyz" else electric_dipole_momentum_matrices.shape[0]
            new_group.attributes["Precision"] = settings.precision.upper()
            new_group.attributes["Description"] = f"Total electric dipole matrices from Group '{self._group_path}'."
            if rotation is not None:
                new_group["ROTATION"] = rotation
                new_group["ROTATION"].attributes["Description"] = "Rotation used to rotate the electric dipole momentum components."

        return electric_dipole_momentum_matrices
    
    @validate_input("HAMILTONIAN")
    def states_electric_dipole_momenta(self, xyz='xyz', start_state=0, stop_state=0, rotation=None, slt_save=None):
        if rotation is not None:
            states_electric_dipole_momenta = diagonal(self.electric_dipole_momenta[:, start_state:stop_state, start_state:stop_state], axis1=1, axis2=2).astype(settings.float, order="C")
            states_electric_dipole_momenta = _rotate_vector_operator(states_electric_dipole_momenta, rotation)
            match xyz:
                case "x":
                    states_electric_dipole_momenta =  states_electric_dipole_momenta[0]
                case "y":
                    states_electric_dipole_momenta = states_electric_dipole_momenta[1]
                case "z":
                    states_electric_dipole_momenta = states_electric_dipole_momenta[2]
        else:
            match xyz:
                case "xyz":
                    states_electric_dipole_momenta = diagonal(self.electric_dipole_momenta[:, start_state:stop_state, start_state:stop_state], axis1=1, axis2=2).astype(settings.float, order="C")
                case "x":
                    states_electric_dipole_momenta = diagonal(self.px[start_state:stop_state, start_state:stop_state]).astype(settings.float, order="C")
                case "y":
                    states_electric_dipole_momenta = diagonal(self.py[start_state:stop_state, start_state:stop_state]).astype(settings.float, order="C")
                case "z":
                    states_electric_dipole_momenta = diagonal(self.pz[start_state:stop_state, start_state:stop_state]).astype(settings.float, order="C")
        
        if slt_save is not None:
            new_group = SltGroup(self._hdf5, slt_save, exists=False)
            new_group["STATES_ELECTRIC_DIPOLE_MOMENTA"] = states_electric_dipole_momenta
            new_group["STATES_ELECTRIC_DIPOLE_MOMENTA"].attributes["Description"] = f"{xyz.upper()}{" [(x-0, y-1, z-2), :]" if xyz == 'xyz' else ''} component{'s' if xyz == 'xyz' else ''} of the states' electric dipole momenta."
            new_group.attributes["Type"] = "STATES_ELECTRIC_DIPOLE_MOMENTA"
            new_group.attributes["Kind"] = f"{xyz.upper()}"
            new_group.attributes["States"] = states_electric_dipole_momenta.shape[1] if xyz == "xyz" else states_electric_dipole_momenta.shape[0]
            new_group.attributes["Precision"] = settings.precision.upper()
            new_group.attributes["Description"] = f"States' expectation values of the electric dipole momentum from Group '{self._group_path}'."
            if rotation is not None:
                new_group["ROTATION"] = rotation
                new_group["ROTATION"].attributes["Description"] = "Rotation used to rotate the electric dipole momentum components."

        return states_electric_dipole_momenta

    @validate_input("HAMILTONIAN")
    def total_angular_momentum_matrices(self, xyz='xyz', start_state=0, stop_state=0, rotation=None, slt_save=None):
        if rotation is not None:
            total_angular_momentum_matrices = self.total_angular_momenta[:, start_state:stop_state, start_state:stop_state]
            total_angular_momentum_matrices = _rotate_vector_operator(total_angular_momentum_matrices, rotation)
            match xyz:
                case "x":
                    total_angular_momentum_matrices =  total_angular_momentum_matrices[0]
                case "y":
                    total_angular_momentum_matrices = total_angular_momentum_matrices[1]
                case "z":
                    total_angular_momentum_matrices = total_angular_momentum_matrices[2]
        else:
            match xyz:
                case "xyz":
                    total_angular_momentum_matrices = self.total_angular_momenta[:, start_state:stop_state, start_state:stop_state]
                case "x":
                    total_angular_momentum_matrices = self.jx[start_state:stop_state, start_state:stop_state]
                case "y":
                    total_angular_momentum_matrices = self.jy[start_state:stop_state, start_state:stop_state]
                case "z":
                    total_angular_momentum_matrices = self.jz[start_state:stop_state, start_state:stop_state]
        
        if slt_save is not None:
            new_group = SltGroup(self._hdf5, slt_save, exists=False)
            new_group["TOTAL_ANGULAR_MOMENTUM_MATRICES"] = total_angular_momentum_matrices
            new_group["TOTAL_ANGULAR_MOMENTUM_MATRICES"].attributes["Description"] = f"{xyz.upper()}{" [(x-0, y-1, z-2), :, :]" if xyz == 'xyz' else ''} component{'s' if xyz == 'xyz' else ''} of the total angular momentum."
            new_group.attributes["Type"] = "TOTAL_ANGULAR_MOMENTA"
            new_group.attributes["Kind"] = f"{xyz.upper()}"
            new_group.attributes["States"] = total_angular_momentum_matrices.shape[1] if xyz == "xyz" else total_angular_momentum_matrices.shape[0]
            new_group.attributes["Precision"] = settings.precision.upper()
            new_group.attributes["Description"] = f"Total angular momentum matrices from Group '{self._group_path}'."
            if rotation is not None:
                new_group["ROTATION"] = rotation
                new_group["ROTATION"].attributes["Description"] = "Rotation used to rotate the total angular momentum components."

        return total_angular_momentum_matrices

    @validate_input("HAMILTONIAN")
    def states_total_angular_momentum(self, xyz='xyz', start_state=0, stop_state=0, rotation=None, slt_save=None):
        if rotation is not None:
            states_total_angular_momenta = diagonal(self.total_angular_momenta[:, start_state:stop_state, start_state:stop_state], axis1=1, axis2=2).astype(settings.float, order="C")
            states_total_angular_momenta = _rotate_vector_operator(states_total_angular_momenta, rotation)
            match xyz:
                case "x":
                    states_total_angular_momenta =  states_total_angular_momenta[0]
                case "y":
                    states_total_angular_momenta = states_total_angular_momenta[1]
                case "z":
                    states_total_angular_momenta = states_total_angular_momenta[2]
        else:
            match xyz:
                case "xyz":
                    states_total_angular_momenta = diagonal(self.total_angular_momenta[:, start_state:stop_state, start_state:stop_state], axis1=1, axis2=2).astype(settings.float, order="C")
                case "x":
                    states_total_angular_momenta = diagonal(self.jx[start_state:stop_state, start_state:stop_state]).astype(settings.float, order="C")
                case "y":
                    states_total_angular_momenta = diagonal(self.jy[start_state:stop_state, start_state:stop_state]).astype(settings.float, order="C")
                case "z":
                    states_total_angular_momenta = diagonal(self.jz[start_state:stop_state, start_state:stop_state]).astype(settings.float, order="C")
        
        if slt_save is not None:
            new_group = SltGroup(self._hdf5, slt_save, exists=False)
            new_group["STATES_TOTAL_ANGULAR_MOMENTA"] = states_total_angular_momenta
            new_group["STATES_TOTAL_ANGULAR_MOMENTA"].attributes["Description"] = f"{xyz.upper()}{" [(x-0, y-1, z-2), :]" if xyz == 'xyz' else ''} component{'s' if xyz == 'xyz' else ''} of the states' total angular momenta."
            new_group.attributes["Type"] = "STATES_TOTAL_ANGULAR_MOMENTA"
            new_group.attributes["Kind"] = f"{xyz.upper()}"
            new_group.attributes["States"] = states_total_angular_momenta.shape[1] if xyz == "xyz" else states_total_angular_momenta.shape[0]
            new_group.attributes["Precision"] = settings.precision.upper()
            new_group.attributes["Description"] = f"States' expectation values of the total angular momentum from Group '{self._group_path}'."
            if rotation is not None:
                new_group["ROTATION"] = rotation
                new_group["ROTATION"].attributes["Description"] = "Rotation used to rotate the total angular momentum components."

        return states_total_angular_momenta

    @validate_input("HAMILTONIAN")
    def magnetic_dipole_momentum_matrices(self, xyz='xyz', start_state=0, stop_state=0, rotation=None, slt_save=None):
        if rotation is not None:
            magnetic_dipole_momentum_matrices = self.magnetic_dipole_momenta[:, start_state:stop_state, start_state:stop_state]
            magnetic_dipole_momentum_matrices = _rotate_vector_operator(magnetic_dipole_momentum_matrices, rotation)
            match xyz:
                case "x":
                    magnetic_dipole_momentum_matrices =  magnetic_dipole_momentum_matrices[0]
                case "y":
                    magnetic_dipole_momentum_matrices = magnetic_dipole_momentum_matrices[1]
                case "z":
                    magnetic_dipole_momentum_matrices = magnetic_dipole_momentum_matrices[2]
        else:
            match xyz:
                case "xyz":
                    magnetic_dipole_momentum_matrices = self.magnetic_dipole_momenta[:, start_state:stop_state, start_state:stop_state]
                case "x":
                    magnetic_dipole_momentum_matrices = self.mx[start_state:stop_state, start_state:stop_state]
                case "y":
                    magnetic_dipole_momentum_matrices = self.my[start_state:stop_state, start_state:stop_state]
                case "z":
                    magnetic_dipole_momentum_matrices = self.mz[start_state:stop_state, start_state:stop_state]
        
        if slt_save is not None:
            new_group = SltGroup(self._hdf5, slt_save, exists=False)
            new_group["MAGNETIC_DIPOLE_MOMENTUM_MATRICES"] = magnetic_dipole_momentum_matrices
            new_group["MAGNETIC_DIPOLE_MOMENTUM_MATRICES"].attributes["Description"] = f"{xyz.upper()}{" [(x-0, y-1, z-2), :, :]" if xyz == 'xyz' else ''} component{'s' if xyz == 'xyz' else ''} of the magnetic dipole momentum."
            new_group.attributes["Type"] = "MAGNETIC_DIPOLE_MOMENTA"
            new_group.attributes["Kind"] = f"{xyz.upper()}"
            new_group.attributes["States"] = magnetic_dipole_momentum_matrices.shape[1] if xyz == "xyz" else magnetic_dipole_momentum_matrices.shape[0]
            new_group.attributes["Precision"] = settings.precision.upper()
            new_group.attributes["Description"] = f"Total magnetic dipole matrices from Group '{self._group_path}'."
            if rotation is not None:
                new_group["ROTATION"] = rotation
                new_group["ROTATION"].attributes["Description"] = "Rotation used to rotate the magnetic dipole momentum components."

        return magnetic_dipole_momentum_matrices

    @validate_input("HAMILTONIAN")
    def states_magnetic_dipole_momenta(self, xyz='xyz', start_state=0, stop_state=0, rotation=None, slt_save=None):
        if rotation is not None:
            states_magnetic_dipole_momenta = diagonal(self.magnetic_dipole_momenta[:, start_state:stop_state, start_state:stop_state], axis1=1, axis2=2).astype(settings.float, order="C")
            states_magnetic_dipole_momenta = _rotate_vector_operator(states_magnetic_dipole_momenta, rotation)
            match xyz:
                case "x":
                    states_magnetic_dipole_momenta =  states_magnetic_dipole_momenta[0]
                case "y":
                    states_magnetic_dipole_momenta = states_magnetic_dipole_momenta[1]
                case "z":
                    states_magnetic_dipole_momenta = states_magnetic_dipole_momenta[2]
        else:
            match xyz:
                case "xyz":
                    states_magnetic_dipole_momenta = diagonal(self.magnetic_dipole_momenta[:, start_state:stop_state, start_state:stop_state], axis1=1, axis2=2).astype(settings.float, order="C")
                case "x":
                    states_magnetic_dipole_momenta = diagonal(self.sx[start_state:stop_state, start_state:stop_state]).astype(settings.float, order="C")
                case "y":
                    states_magnetic_dipole_momenta = diagonal(self.sy[start_state:stop_state, start_state:stop_state]).astype(settings.float, order="C")
                case "z":
                    states_magnetic_dipole_momenta = diagonal(self.sz[start_state:stop_state, start_state:stop_state]).astype(settings.float, order="C")
        
        if slt_save is not None:
            new_group = SltGroup(self._hdf5, slt_save, exists=False)
            new_group["STATES_MAGNETIC_DIPOLE_MOMENTA"] = states_magnetic_dipole_momenta
            new_group["STATES_MAGNETIC_DIPOLE_MOMENTA"].attributes["Description"] = f"{xyz.upper()}{" [(x-0, y-1, z-2), :]" if xyz == 'xyz' else ''} component{'s' if xyz == 'xyz' else ''} of the states' magnetic dipole momenta."
            new_group.attributes["Type"] = "STATES_MAGNETIC_DIPOLE_MOMENTA"
            new_group.attributes["Kind"] = f"{xyz.upper()}"
            new_group.attributes["States"] = states_magnetic_dipole_momenta.shape[1] if xyz == "xyz" else states_magnetic_dipole_momenta.shape[0]
            new_group.attributes["Precision"] = settings.precision.upper()
            new_group.attributes["Description"] = f"States' expectation values of the magnetic dipole momentum from Group '{self._group_path}'."
            if rotation is not None:
                new_group["ROTATION"] = rotation
                new_group["ROTATION"].attributes["Description"] = "Rotation used to rotate the magnetic dipole momentum components."

        return states_magnetic_dipole_momenta


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
        with File(self._hdf5, 'r') as file:
            dataset = file[self._dataset_path]
            return dataset.dtype


class SltDatasetSLP():
    def __init__(self, hdf5_file_path, dataset_path, slp, xyz):
        self._hdf5 = hdf5_file_path
        self._dataset_path = dataset_path
        self._slp = slp
        self._xyz = xyz
        self._xyz_dict = {0: "x", 1: "y", 2: "z"}

    @slothpy_exc("SltFileError")
    def __getitem__(self, slice_):
        with File(self._hdf5, 'r') as file:
            dataset = file[self._dataset_path].astype(settings.complex)
            return dataset[self._xyz, *(slice_,) if isinstance(slice_, slice) else slice_]
        
    @slothpy_exc("SltFileError")
    def __repr__(self):
        with File(self._hdf5, 'r+') as file:
            file[self._dataset_path]
            return f"<{PURPLE}SltDataset{self._slp}{self._xyz_dict[self._xyz]}{RESET} from '{self._dataset_path}' in {GREEN}File{RESET} '{self._hdf5}'.>"
        

class SltDatasetJM():
    def __init__(self, hdf5_file_path, group_path, jm, xyz=None):
        self._hdf5 = hdf5_file_path
        self._group_path = group_path
        self._jm = jm
        self._xyz = xyz
        self._xyz_dict = {0: "x", 1: "y", 2: "z"}

    @slothpy_exc("SltFileError")
    def __getitem__(self, slice_):
        with File(self._hdf5, 'r') as file:
            group = file[self._group_path]
            if self._xyz is not None:
                dataset_s = group["SPINS"].astype(settings.complex)[self._xyz, *(slice_,) if isinstance(slice_, slice) else slice_]
                dataset_l = group["ANGULAR_MOMENTA"].astype(settings.complex)[self._xyz, *(slice_,) if isinstance(slice_, slice) else slice_]
            else:
                dataset_s = group["SPINS"][slice_].astype(settings.complex)
                dataset_l = group["ANGULAR_MOMENTA"][slice_].astype(settings.complex)
            if self._jm == "J":
                return _total_angular_momenta_from_spins_angular_momenta(dataset_s, dataset_l)
            elif self._jm == "M":
                return  _magnetic_dipole_momenta_from_spins_angular_momenta(dataset_s, dataset_l)
            else:
                raise ValueError("The only supported options are 'J' for total angular momenta or 'M' for magnetic dipole momenta.")
        
    @slothpy_exc("SltFileError")
    def __repr__(self):
        with File(self._hdf5, 'r+') as file:
            file[self._group_path]
            return f"<{PURPLE}SltDataset{self._jm}{self._xyz_dict[self._xyz]}{RESET} from '{self._dataset_path}' in {GREEN}File{RESET} '{self._hdf5}'.>"