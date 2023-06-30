from os import path
from typing import Any
import h5py
import numpy as np
from slothpy.magnetism.g_tensor import calculate_g_tensor_and_axes_doublet
from slothpy.magnetism.magnetisation import mth
from slothpy.magnetism.susceptibility import (chitht, chit_tensorht)
from slothpy.general_utilities.grids_over_hemisphere import lebedev_laikov_grid
from slothpy.general_utilities.io import get_soc_energies_cm_1

class Compound:

    @classmethod
    def _new(cls, filepath: str, filename: str):

        filename += ".slt"

        hdf5_file = path.join(filepath, filename)

        obj = super().__new__(cls)

        obj._hdf5 = hdf5_file
        obj.get_hdf5_groups_and_attributes()

        return obj

    def __new__(cls, *args, **kwargs):
        raise TypeError("The Compound object should not be instantiated directly. Use a Compound creation function instead.")
    
    def __repr__(self) -> str:

        representation = f"Compound from {self._hdf5} with the following groups of data:\n"
        
        for group, attributes in self._groups.items():
            representation += f"{group}: {attributes}\n"

        return representation

    def __str__(self) -> str:
      
        string = f"Compound from {self._hdf5} with the following groups of data:\n"
        
        for group, attributes in self._groups.items():
            string += f"{group}: {attributes}\n"

        return string
        
    # def __setitem__(self) -> None:
    #     pass

    # def __getitem__(self) -> Any:
    #     pass

    # def __getattr__(self, __name: str) -> Any:
    #     pass

    # def __setattr__(self, __name: str, __value: Any) -> None:
    #     pass

    def get_hdf5_groups_and_attributes(self):

        def collect_groups(name, obj):

            if isinstance(obj, h5py.Group):
                groups_dict[name] = dict(obj.attrs)

        groups_dict = {}

        with h5py.File(self._hdf5, 'r') as file:
            file.visititems(collect_groups)

        self._groups = groups_dict

    def delete_group(self, group: str) -> None:

        try:
            with h5py.File(self._hdf5, 'r+') as file:
                del file[group]

        except Exception as e:
            error_type = type(e).__name__
            error_message = str(e)
            print(f'Error encountered while trying to delete group {group} from .slt file: {self._hdf5}: {error_type}: {error_message}')
            return

        self.get_hdf5_groups_and_attributes()
        

    def calculate_g_tensor_and_axes_doublet(self, group: str, doublets: np.ndarray, slt: str = None):
        
        try:
            g_tensor_list, magnetic_axes_list = calculate_g_tensor_and_axes_doublet(self._hdf5, group, doublets)

        except Exception as e:
            error_type = type(e).__name__
            error_message = str(e)
            print(f'Error encountered while trying to compute g-tensors and main magnetic axes from file: {self._hdf5} - group {group}: {error_type}: {error_message}')
            return

        if slt is not None:
            try:
                with h5py.File(self._hdf5, 'r+') as file:
                    new_group = file.create_group(f'{slt}_g_tensors_axes')
                    new_group.attrs['Description'] = f'Group({slt}) containing g-tensors of doublets and their magnetic axes calulated from group: {group}.'
                    tensors = new_group.create_dataset(f'{slt}_g_tensors', shape=(g_tensor_list.shape[0], g_tensor_list.shape[1]), dtype=np.float64)
                    tensors.attrs['Description'] = f'Dataset containing number of doublet and respective g-tensors from group {group}.'
                    axes = new_group.create_dataset(f'{slt}_axes', shape=(magnetic_axes_list.shape[0], magnetic_axes_list.shape[1], magnetic_axes_list.shape[2]), dtype=np.float64)
                    axes.attrs['Description'] = f'Dataset containing rotation matrices from initial coordinate system to magnetic axes of respective g-tensors from group: {group}.'
                    tensors[:,:] = g_tensor_list[:,:]
                    axes[:,:,:] = magnetic_axes_list[:,:,:]
            
                self.get_hdf5_groups_and_attributes()

            except Exception as e:
                error_type = type(e).__name__
                error_message = str(e)
                print(f'Error encountered while trying to save g-tensors and magnetic axes to file: {self._hdf5} - group {slt}: {error_type}: {error_message}')
                return

        return g_tensor_list, magnetic_axes_list


    def calculate_mth(self, group: str, states_cutoff: np.int64, fields: np.ndarray, grid: np.ndarray, temperatures: np.ndarray, num_cpu: int, slt: str = None):

        fields = np.array(fields)
        temperatures = np.array(temperatures)

        if isinstance(grid, int):
            grid = lebedev_laikov_grid(grid)
        else:
            grid = np.array(grid)

        try:
            mth_array = mth(self._hdf5, group, states_cutoff, fields, grid, temperatures, num_cpu)
        except Exception as e:
            error_type = type(e).__name__
            error_message = str(e)
            print(f'Error encountered while trying to compute M(T,H) from file: {self._hdf5} - group {group}: {error_type}: {error_message}')
            return
        
        if slt is not None:
            try:
                with h5py.File(self._hdf5, 'r+') as file:
                    new_group = file.create_group(f'{slt}_magnetisation')
                    new_group.attrs['Description'] = f'Group({slt}) containing M(T,H) magnetisation calulated from group: {group}.'
                    mth_dataset = new_group.create_dataset(f'{slt}_mth', shape=(mth_array.shape[0], mth_array.shape[1]), dtype=np.float64)
                    mth_dataset.attrs['Description'] = f'Dataset containing M(T,H) magnetisation (T - rows, H - columns) calulated from group: {group}.'
                    fields_dataset = new_group.create_dataset(f'{slt}_fields', shape=(fields.shape[0],), dtype=np.float64)
                    fields_dataset.attrs['Description'] = f'Dataset containing magnetic field H values used in simulation of M(T,H) from group: {group}.'
                    temperatures_dataset = new_group.create_dataset(f'{slt}_temperatures', shape=(temperatures.shape[0],), dtype=np.float64)
                    temperatures_dataset.attrs['Description'] = f'Dataset containing temperature T values used in simulation of M(T,H) from group: {group}.'

                    mth_dataset[:,:] = mth_array[:,:]
                    fields_dataset[:] = fields[:]
                    temperatures_dataset[:] = temperatures[:]
            
                self.get_hdf5_groups_and_attributes()

            except Exception as e:
                error_type = type(e).__name__
                error_message = str(e)
                print(f'Error encountered while trying to save M(T,H) to file: {self._hdf5} - group {slt}: {error_type}: {error_message}')
                return

        return mth_array
     
        
    def calculate_chitht(self, group: str, fields: np.ndarray, states_cutoff: int, temperatures: np.ndarray, num_cpu: int, num_of_points: int, delta_h: np.float64, exp: bool = False, T: bool = False, grid: np.ndarray = None, slt: str = None) -> np.ndarray:

        fields = np.array(fields)
        temperatures = np.array(temperatures)

        if isinstance(grid, int):
            grid = lebedev_laikov_grid(grid)
        else:
            grid = np.array(grid)

        try:
            chitht_array = chitht(self._hdf5, group, fields, states_cutoff, temperatures, num_cpu, num_of_points, delta_h, exp, T, grid)
        except Exception as e:
            error_type = type(e).__name__
            error_message = str(e)
            print(f'Error encountered while trying to compute chiT(H,T) from file: {self._hdf5} - group {group}: {error_type}: {error_message}')
            return
        
        if slt is not None:
            try:
                with h5py.File(self._hdf5, 'r+') as file:
                    new_group = file.create_group(f'{slt}_susceptibility')
                    new_group.attrs['Description'] = f'Group({slt}) containing chiT(H,T) magnetic susceptibility calulated from group: {group}.'
                    chitht_dataset = new_group.create_dataset(f'{slt}_chitht', shape=(chitht_array.shape[0], chitht_array.shape[1]), dtype=np.float64)
                    chitht_dataset.attrs['Description'] = f'Dataset containing chiT(H,T) magnetic susceptibility (H - rows, T - columns) calulated from group: {group}.'
                    fields_dataset = new_group.create_dataset(f'{slt}_fields', shape=(fields.shape[0],), dtype=np.float64)
                    fields_dataset.attrs['Description'] = f'Dataset containing magnetic field H values used in simulation of chiT(H,T) from group: {group}.'
                    temperatures_dataset = new_group.create_dataset(f'{slt}_temperatures', shape=(temperatures.shape[0],), dtype=np.float64)
                    temperatures_dataset.attrs['Description'] = f'Dataset containing temperature T values used in simulation of chiT(H,T) from group: {group}.'

                    chitht_dataset[:,:] = chitht_array[:,:]
                    fields_dataset[:] = fields[:]
                    temperatures_dataset[:] = temperatures[:]
            
                self.get_hdf5_groups_and_attributes()

            except Exception as e:
                error_type = type(e).__name__
                error_message = str(e)
                print(f'Error encountered while trying to save chiT(H,T) to file: {self._hdf5} - group {slt}: {error_type}: {error_message}')
                return

        return chitht_array 
        
    
    def calculate_chit_tensorht(self, group: str, fields: np.ndarray, states_cutoff: int, temperatures: np.ndarray, num_cpu: int, num_of_points: int, delta_h: np.float64, T: bool = True, slt: str = None):

        fields = np.array(fields)
        temperatures = np.array(temperatures)

        try:
            chit_tensorht_array = chit_tensorht(self._hdf5, group, fields, states_cutoff, temperatures, num_cpu, num_of_points, delta_h, T)
        except Exception as e:
            error_type = type(e).__name__
            error_message = str(e)
            print(f'Error encountered while trying to compute chi_tensor(H,T) from file: {self._hdf5} - group {group}: {error_type}: {error_message}')
            return
        
        if slt is not None:
            try:
                with h5py.File(self._hdf5, 'r+') as file:
                    new_group = file.create_group(f'{slt}_susceptibility_tensor')
                    new_group.attrs['Description'] = f'Group({slt}) containing chiT_tensor(H,T) Van Vleck susceptibility tensor calulated from group: {group}.'
                    chit_tensorht_dataset = new_group.create_dataset(f'{slt}_chit_tensorht', shape=(chit_tensorht_array.shape[0], chit_tensorht_array.shape[1],3,3), dtype=np.float64)
                    chit_tensorht_dataset.attrs['Description'] = f'Dataset containing chiT_tensor(H,T) Van Vleck susceptibility tensor (H, T, 3, 3) calulated from group: {group}.'
                    fields_dataset = new_group.create_dataset(f'{slt}_fields', shape=(fields.shape[0],), dtype=np.float64)
                    fields_dataset.attrs['Description'] = f'Dataset containing magnetic field H values used in simulation of chiT_tensor(H,T) Van Vleck susceptibility tensor from group: {group}.'
                    temperatures_dataset = new_group.create_dataset(f'{slt}_temperatures', shape=(temperatures.shape[0],), dtype=np.float64)
                    temperatures_dataset.attrs['Description'] = f'Dataset containing temperature T values used in simulation of chiT_tensor(H,T) Van Vleck susceptibility tensor from group: {group}.'

                    chit_tensorht_dataset[:,:] = chit_tensorht_array[:,:]
                    fields_dataset[:] = fields[:]
                    temperatures_dataset[:] = temperatures[:]
            
                self.get_hdf5_groups_and_attributes()

            except Exception as e:
                error_type = type(e).__name__
                error_message = str(e)
                print(f'Error encountered while trying to save chiT(H,T) to file: {self._hdf5} - group {slt}: {error_type}: {error_message}')
                return

        return chit_tensorht_array
    

    def soc_energies_cm_1(filename: str, group: str, num_of_states: int = None) -> np.ndarray:
         
        pass