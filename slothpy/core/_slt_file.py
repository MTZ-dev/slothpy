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
from numpy import array, ComplexWarning
from slothpy.core._slothpy_exceptions import slothpy_exc, SltFileError, SltReadError, SltCompError
from slothpy.core._config import settings
from slothpy._general_utilities._constants import RED, GREEN, BLUE, PURPLE, YELLOW, RESET, H_CM_1
from slothpy.core._input_parser import validate_input
from slothpy._general_utilities._math_expresions import _magnetic_momenta_from_spins_angular_momenta, _total_angular_momenta_from_spins_angular_momenta
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
    def soc_energies(self):
        return self["SOC_ENERGIES"]
    
    @property
    @validate_input("HAMILTONIAN")
    def spins(self):
        return self["SOC_SPINS"]
    
    @property
    @validate_input("HAMILTONIAN")
    def sx(self):
        return SltDatasetSLP(self._hdf5, f"{self._group_path}/SOC_SPINS", "S", 0)
    
    @property
    @validate_input("HAMILTONIAN")
    def sy(self):
        return SltDatasetSLP(self._hdf5, f"{self._group_path}/SOC_SPINS", "S", 1)
    
    @property
    @validate_input("HAMILTONIAN")
    def sz(self):
        return SltDatasetSLP(self._hdf5, f"{self._group_path}/SOC_SPINS", "S", 2)
    
    @property
    @validate_input("HAMILTONIAN")
    def angular_momenta(self):
        return self["SOC_ANGULAR_MOMENTA"]
    
    @property
    @validate_input("HAMILTONIAN")
    def lx(self):
        return SltDatasetSLP(self._hdf5, f"{self._group_path}/SOC_ANGULAR_MOMENTA", "L", 0)
    
    @property
    @validate_input("HAMILTONIAN")
    def ly(self):
        return SltDatasetSLP(self._hdf5, f"{self._group_path}/SOC_ANGULAR_MOMENTA", "L", 1)
    
    @property
    @validate_input("HAMILTONIAN")
    def lz(self):
        return SltDatasetSLP(self._hdf5, f"{self._group_path}/SOC_ANGULAR_MOMENTA", "L", 2)
    
    @property
    @validate_input("HAMILTONIAN")
    def electric_dipole_momenta(self):
        return self["SOC_ELECTRIC_DIPOLE_MOMENTA"]
    
    @property
    @validate_input("HAMILTONIAN")
    def px(self):
        return SltDatasetSLP(self._hdf5, f"{self._group_path}/SOC_ELECTRIC_DIPOLE_MOMENTA", "P", 0)
    
    @property
    @validate_input("HAMILTONIAN")
    def py(self):
        return SltDatasetSLP(self._hdf5, f"{self._group_path}/SOC_ELECTRIC_DIPOLE_MOMENTA", "P", 1)
    
    @property
    @validate_input("HAMILTONIAN")
    def pz(self):
        return SltDatasetSLP(self._hdf5, f"{self._group_path}/SOC_ELECTRIC_DIPOLE_MOMENTA", "P", 2)
    
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
    def magnetic_momenta(self):
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
    def soc_energies_cm_1(self, start_state=0, stop_state=0, slt_save=None):
        try:
            soc_energies_cm_1 = self.soc_energies[start_state:stop_state] * H_CM_1
        except Exception as exc:
            raise SltCompError(self._hdf5, exc, f"Failed to compute SOC energies in cm-1 from {BLUE}Group{RESET}: '{self._hdf5}'.")
        if slt_save is not None:
            new_group = SltGroup(self._hdf5, slt_save, exists=False)
            new_group["SOC_ENERGIES_CM_1"] = soc_energies_cm_1
            new_group["SOC_ENERGIES_CM_1"].attributes["Description"] = "SOC energies in cm-1."
            new_group.attributes["Type"] = "SOC_ENERGIES"
            new_group.attributes["Kind"] = "CM_1"
            new_group.attributes["States"] = soc_energies_cm_1.shape[0]
            new_group.attributes["Precision"] = settings.precision
            new_group.attributes["Description"] = "SOC energies in cm-1."
        return soc_energies_cm_1
    
    @validate_input("HAMILTONIAN")
    def soc_energies_au(self, start_state=0, stop_state=0, slt_save=None):
        try:
            soc_energies_au = self.soc_energies[start_state:stop_state]
        except Exception as exc:
            raise SltCompError(self._hdf5, exc, f"Failed to compute SOC energies in a.u. from {BLUE}Group{RESET}: '{self._hdf5}'.")
        if slt_save is not None:
            new_group = SltGroup(self._hdf5, slt_save, exists=False)
            new_group["SOC_ENERGIES_AU"] = soc_energies_au
            new_group["SOC_ENERGIES_AU"].attributes["Description"] = "SOC energies in a.u.."
            new_group.attributes["Type"] = "SOC_ENERGIES"
            new_group.attributes["Kind"] = "AU"
            new_group.attributes["States"] = soc_energies_au.shape[0]
            new_group.attributes["Precision"] = settings.precision
            new_group.attributes["Description"] = "SOC energies in a.u.."
        return soc_energies_au
    
    @validate_input("HAMILTONIAN")
    def spin_matrices(self, xyz='xyz', start_state=0, stop_state=0, rotation=None, slt_save=None):
        if rotation is not None:
            spin_matrices = self.spin[:, start_state:stop_state, start_state:stop_state]
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
            new_group["SOC_SPIN_MATRICES"] = spin_matrices
            new_group["SOC_SPIN_MATRICES"].attributes["Description"] = f"{xyz.upper()} component{'s' if xyz == 'xyz' else ''} of the spin."
            new_group.attributes["Type"] = "SOC_SPINS"
            new_group.attributes["Kind"] = f"{xyz.upper()}"
            new_group.attributes["States"] = spin_matrices.shape[0]
            new_group.attributes["Precision"] = settings.precision
            new_group.attributes["Description"] = f"SOC spin matrices from Group '{self._group_path}'."
            if rotation is not None:
                new_group["ROTATION"] = rotation
                new_group["ROTATION"].attributes["Description"] = "Rotation used to rotate the spin components."

        return spin_matrices
    
    @validate_input("HAMILTONIAN")
    def states_spin(self, xyz='xyz', start_state=0, stop_state=0, rotation=None, slt_save=None):
        pass
    
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
            new_group["SOC_ANGULAR_MOMENUM_MATRICES"] = angular_momentum_matrices
            new_group["SOC_ANGULAR_MOMENTUM_MATRICES"].attributes["Description"] = f"{xyz.upper()} component{'s' if xyz == 'xyz' else ''} of the angular momentum."
            new_group.attributes["Type"] = "SOC_ANGULAR_MOMENTA"
            new_group.attributes["Kind"] = f"{xyz.upper()}"
            new_group.attributes["States"] = angular_momentum_matrices.shape[0]
            new_group.attributes["Precision"] = settings.precision
            new_group.attributes["Description"] = f"SOC angular momentum matrices from Group '{self._group_path}'."
            if rotation is not None:
                new_group["ROTATION"] = rotation
                new_group["ROTATION"].attributes["Description"] = "Rotation used to rotate the angular momentum components."

        return angular_momentum_matrices
    
    @validate_input("HAMILTONIAN")
    def states_angular_momenta(self, xyz='xyz', start_state=0, stop_state=0, rotation=None, slt_save=None):
        pass

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
            new_group["SOC_TOTAL_ANGULAR_MOMENUM_MATRICES"] = total_angular_momentum_matrices
            new_group["SOC_TOTAL_ANGULAR_MOMENTUM_MATRICES"].attributes["Description"] = f"{xyz.upper()} component{'s' if xyz == 'xyz' else ''} of the total angular momentum."
            new_group.attributes["Type"] = "SOC_TOTAL_ANGULAR_MOMENTA"
            new_group.attributes["Kind"] = f"{xyz.upper()}"
            new_group.attributes["States"] = total_angular_momentum_matrices.shape[0]
            new_group.attributes["Precision"] = settings.precision
            new_group.attributes["Description"] = f"SOC total angular momentum matrices from Group '{self._group_path}'."
            if rotation is not None:
                new_group["ROTATION"] = rotation
                new_group["ROTATION"].attributes["Description"] = "Rotation used to rotate the total angular momentum components."

        return total_angular_momentum_matrices

    @validate_input("HAMILTONIAN")
    def states_total_angular_momentum(self, xyz='xyz', start_state=0, stop_state=0, rotation=None, slt_save=None):
        pass

    @validate_input("HAMILTONIAN")
    def magnetic_momentum_matrices(self, xyz='xyz', start_state=0, stop_state=0, rotation=None, slt_save=None):
        if rotation is not None:
            magnetic_momentum_matrices = self.magnetic_momenta[:, start_state:stop_state, start_state:stop_state]
            magnetic_momentum_matrices = _rotate_vector_operator(magnetic_momentum_matrices, rotation)
            match xyz:
                case "x":
                    magnetic_momentum_matrices =  magnetic_momentum_matrices[0]
                case "y":
                    magnetic_momentum_matrices = magnetic_momentum_matrices[1]
                case "z":
                    magnetic_momentum_matrices = magnetic_momentum_matrices[2]
        else:
            match xyz:
                case "xyz":
                    magnetic_momentum_matrices = self.magnetic_momenta[:, start_state:stop_state, start_state:stop_state]
                case "x":
                    magnetic_momentum_matrices = self.jx[start_state:stop_state, start_state:stop_state]
                case "y":
                    magnetic_momentum_matrices = self.jy[start_state:stop_state, start_state:stop_state]
                case "z":
                    magnetic_momentum_matrices = self.jz[start_state:stop_state, start_state:stop_state]
        
        if slt_save is not None:
            new_group = SltGroup(self._hdf5, slt_save, exists=False)
            new_group["SOC_MAGNETIC_MOMENUM_MATRICES"] = magnetic_momentum_matrices
            new_group["SOC_MAGNETIC_MOMENTUM_MATRICES"].attributes["Description"] = f"{xyz.upper()} component{'s' if xyz == 'xyz' else ''} of the magnetic momentum."
            new_group.attributes["Type"] = "SOC_MAGNETIC_MOMENTA"
            new_group.attributes["Kind"] = f"{xyz.upper()}"
            new_group.attributes["States"] = magnetic_momentum_matrices.shape[0]
            new_group.attributes["Precision"] = settings.precision
            new_group.attributes["Description"] = f"SOC total magnetic matrices from Group '{self._group_path}'."
            if rotation is not None:
                new_group["ROTATION"] = rotation
                new_group["ROTATION"].attributes["Description"] = "Rotation used to rotate the magnetic momentum components."

        return magnetic_momentum_matrices

    @validate_input("HAMILTONIAN")
    def states_magnetic_momenta(self, xyz='xyz', start_state=0, stop_state=0, rotation=None, slt_save=None):
        pass


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
                    return dataset[slice_].astype(settings.complex)
                case "f":
                    return dataset[slice_].astype(settings.float)
                case "i":
                    return dataset[slice_].astype(settings.int)
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
            dataset = file[self._dataset_path]
            return dataset[self._xyz, *(slice_,) if isinstance(slice_, slice) else slice_].astype(settings.complex)
        
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
                dataset_s = group["SOC_SPINS"][self._xyz, *(slice_,) if isinstance(slice_, slice) else slice_].astype(settings.complex)
                dataset_l = group["SOC_ANGULAR_MOMENTA"][self._xyz, *(slice_,) if isinstance(slice_, slice) else slice_].astype(settings.complex)
            else:
                dataset_s = group["SOC_SPINS"][slice_].astype(settings.complex)
                dataset_l = group["SOC_ANGULAR_MOMENTA"][slice_].astype(settings.complex)
            if self._jm == "J":
                return _total_angular_momenta_from_spins_angular_momenta(dataset_s, dataset_l)
            elif self._jm == "M":
                return  _magnetic_momenta_from_spins_angular_momenta(dataset_s, dataset_l)
            else:
                raise ValueError("The only supported options are 'J' for total angular momenta or 'M' for magnetic momenta.")
        
    @slothpy_exc("SltFileError")
    def __repr__(self):
        with File(self._hdf5, 'r+') as file:
            file[self._group_path]
            return f"<{PURPLE}SltDataset{self._jm}{self._xyz_dict[self._xyz]}{RESET} from '{self._dataset_path}' in {GREEN}File{RESET} '{self._hdf5}'.>"