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

from typing import Literal
from h5py import File, Group, Dataset
from numpy import array
from slothpy.core._slothpy_exceptions import slothpy_exc, SltFileError, SltReadError, SltCompError
from slothpy.core._config import settings
from slothpy._general_utilities._constants import RED, GREEN, BLUE, PURPLE, YELLOW, RESET, H_CM_1
from slothpy.core._input_parser import validate_input
from slothpy._general_utilities._math_expresions import _magnetic_momenta_from_spin_angular_momenta, _total_angular_momenta_from_spin_angular_momenta

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
    def soc_energies(self):
        self._check_if_slt_valid_group("HAMILTONIAN", "which does not SOC energies.")
        return self["SOC_ENERGIES"]
    
    @property
    def spin(self):
        self._check_if_slt_valid_group("HAMILTONIAN", "which does not contain spin momenta.")
        return self["SOC_SPIN"]
    
    @property
    def sx(self):
        self._check_if_slt_valid_group("HAMILTONIAN", "which does not contain spin momenta.")
        return SltDatasetSLP(self._hdf5, f"{self._group_path}/SOC_SPIN", "S", 0)
    
    @property
    def sy(self):
        self._check_if_slt_valid_group("HAMILTONIAN", "which does not contain spin momenta.")
        return SltDatasetSLP(self._hdf5, f"{self._group_path}/SOC_SPIN", "S", 1)
    
    @property
    def sz(self):
        self._check_if_slt_valid_group("HAMILTONIAN", "which does not contain spin momenta.")
        return SltDatasetSLP(self._hdf5, f"{self._group_path}/SOC_SPIN", "S", 2)
    
    @property
    def angular_momenta(self):
        self._check_if_slt_valid_group("HAMILTONIAN", "which does not contain angular momenta.")
        return self["SOC_ANGULAR_MOMENTA"]
    
    @property
    def lx(self):
        self._check_if_slt_valid_group("HAMILTONIAN", "which does not contain angular momenta.")
        return SltDatasetSLP(self._hdf5, f"{self._group_path}/SOC_ANGULAR_MOMENTA", "L", 0)
    
    @property
    def ly(self):
        self._check_if_slt_valid_group("HAMILTONIAN", "which does not contain angular momenta.")
        return SltDatasetSLP(self._hdf5, f"{self._group_path}/SOC_ANGULAR_MOMENTA", "L", 1)
    
    @property
    def lz(self):
        self._check_if_slt_valid_group("HAMILTONIAN", "which does not contain angular momenta.")
        return SltDatasetSLP(self._hdf5, f"{self._group_path}/SOC_ANGULAR_MOMENTA", "L", 2)
    
    @property
    def electric_dipole_momenta(self):
        self._check_if_slt_valid_group("HAMILTONIAN", "which does not contain electric dipole momenta.")
        return self["SOC_ELECTRIC_DIPOLE_MOMENTA"]
    
    @property
    def px(self):
        self._check_if_slt_valid_group("HAMILTONIAN", "which does not contain electric dipole momenta.")
        return SltDatasetSLP(self._hdf5, f"{self._group_path}/SOC_ELECTRIC_DIPOLE_MOMENTA", "P", 0)
    
    @property
    def py(self):
        self._check_if_slt_valid_group("HAMILTONIAN", "which does not contain electric dipole momenta.")
        return SltDatasetSLP(self._hdf5, f"{self._group_path}/SOC_ELECTRIC_DIPOLE_MOMENTA", "P", 1)
    
    @property
    def pz(self):
        self._check_if_slt_valid_group("HAMILTONIAN", "which does not contain electric dipole momenta.")
        return SltDatasetSLP(self._hdf5, f"{self._group_path}/SOC_ELECTRIC_DIPOLE_MOMENTA", "P", 2)
    
    @property
    def total_angular_momenta(self):
        self._check_if_slt_valid_group("HAMILTONIAN", "which does not contain spin and angular momenta.")
        return SltDatasetJM(self._hdf5, f"{self._group_path}", "J")
    
    @property
    def jx(self):
        self._check_if_slt_valid_group("HAMILTONIAN", "which does not contain spin and angular momenta.")
        return SltDatasetJM(self._hdf5, f"{self._group_path}", "J", 0)
    
    @property
    def jy(self):
        self._check_if_slt_valid_group("HAMILTONIAN", "which does not contain spin and angular momenta.")
        return SltDatasetJM(self._hdf5, f"{self._group_path}", "J", 1)
    
    @property
    def jz(self):
        self._check_if_slt_valid_group("HAMILTONIAN", "which does not contain spin and angular momenta.")
        return SltDatasetJM(self._hdf5, f"{self._group_path}", "J", 2)
    
    @property
    def magnetic_momenta(self):
        self._check_if_slt_valid_group("HAMILTONIAN", "which does not contain spin and angular momenta.")
        return SltDatasetJM(self._hdf5, f"{self._group_path}", "M")
    
    @property
    def mx(self):
        self._check_if_slt_valid_group("HAMILTONIAN", "which does not contain spin and angular momenta.")
        return SltDatasetJM(self._hdf5, f"{self._group_path}", "M", 0)
    
    @property
    def my(self):
        self._check_if_slt_valid_group("HAMILTONIAN", "which does not contain spin and angular momenta.")
        return SltDatasetJM(self._hdf5, f"{self._group_path}", "M", 1)
    
    @property
    def mz(self):
        self._check_if_slt_valid_group("HAMILTONIAN", "which does not contain spin and angular momenta.")
        return SltDatasetJM(self._hdf5, f"{self._group_path}", "M", 2)
    
    def _check_if_slt_valid_group(self, group_type, error_message):
        try:
            self.attributes["Type"]
        except SltFileError as exc:
            raise SltReadError(self._hdf5, None, f"{BLUE}Group{RESET}: '{self._group_path}' is not a valid SlothPy group.") from None
        if self.attributes["Type"] != group_type:
            raise SltReadError(self._hdf5, KeyError(f"Wrong group type: {self.attributes['Type']} " + error_message + f" Expected '{group_type}' type."))

    @validate_input("HAMILTONIAN")   
    def soc_energies_cm_1(self, start_state, stop_state, slt_save=None):
        try:
            soc_energies_cm_1 = self.soc_energies[start_state:stop_state] * H_CM_1
        except Exception as exc:
            raise SltCompError(self._hdf5, exc, f"Failed to compute SOC energies from {BLUE}Group{RESET}: '{self._hdf5}'.")
        if slt_save is not None:
            new_group = SltGroup(self._hdf5, slt_save, exists=False)
            new_group["SOC_ENERGIES_CM_1"] = soc_energies_cm_1
            new_group["SOC_ENERGIES_CM_1"].attributes["Description"] = "SOC energies in cm-1."
            new_group.attributes["Type"] = "Soc_energies"
            new_group.attributes["Kind"] = "CM_1"
            new_group.attributes["States"] = soc_energies_cm_1.shape[0]
            new_group.attributes["Precision"] = settings.precision
            new_group.attributes["Description"] = "SOC energies in cm-1."
        return soc_energies_cm_1
    
    @validate_input("HAMILTONIAN")
    def soc_energies_a_u(self, start_state, stop_state, slt_save=None):
        result = self.soc_energies[start_state:stop_state]
        return result
    
    def spin_matrices(self, xyz, start_state, stop_state, rotation=None, slt_save=None):
        pass

    def states_spin(self, xyz, start_state, stop_state, rotation=None, slt_save=None):
        pass
    
    def angular_momenta_matrices(self, xyz, start_state, stop_state, rotation=None, slt_save=None):
        pass

    def states_angular_momenta(self, xyz, start_state, stop_state, rotation=None, slt_save=None):
        pass
    
    def total_angular_momenta_matrices(self, xyz, start_state, stop_state, rotation=None, slt_save=None):
        pass

    def states_total_angular_momenta(self, xyz, start_state, stop_state, rotation=None, slt_save=None):
        pass

    def magnetic_momenta_matrices(self, xyz, start_state, stop_state, rotation=None, slt_save=None):
        pass

    def states_magnetic_momenta(self, xyz, start_state, stop_state, rotation=None, slt_save=None):
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
                dataset_s = group["SOC_SPIN"][self._xyz, *(slice_,) if isinstance(slice_, slice) else slice_].astype(settings.complex)
                dataset_l = group["SOC_ANGULAR_MOMENTA"][self._xyz, *(slice_,) if isinstance(slice_, slice) else slice_].astype(settings.complex)
            else:
                dataset_s = group["SOC_SPIN"][slice_].astype(settings.complex)
                dataset_l = group["SOC_ANGULAR_MOMENTA"][slice_].astype(settings.complex)
            if self._jm == "J":
                return _total_angular_momenta_from_spin_angular_momenta(dataset_s, dataset_l)
            elif self._jm == "M":
                return  _magnetic_momenta_from_spin_angular_momenta(dataset_s, dataset_l)
            else:
                raise ValueError("The only supported options are 'J' for total angular momenta or 'M' for magnetic momenta.")
        
    @slothpy_exc("SltFileError")
    def __repr__(self):
        with File(self._hdf5, 'r+') as file:
            file[self._group_path]
            return f"<{PURPLE}SltDataset{self._jm}{self._xyz_dict[self._xyz]}{RESET} from '{self._dataset_path}' in {GREEN}File{RESET} '{self._hdf5}'.>"