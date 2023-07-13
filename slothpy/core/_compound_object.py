from os import path
from typing import Any
import h5py
import numpy as np
from slothpy.magnetism.g_tensor import calculate_g_tensor_and_axes_doublet
from slothpy.magnetism.magnetisation import (mth, mag_3d)
from slothpy.magnetism.susceptibility import (chitht, chit_tensorht, chit_3d)
from slothpy.general_utilities.grids_over_hemisphere import lebedev_laikov_grid
from slothpy.general_utilities.io import (get_soc_energies_cm_1, get_states_magnetic_momenta, get_states_total_angular_momneta,
                                           get_total_angular_momneta_matrix, get_magnetic_momenta_matrix)
from slothpy.magnetism.zeeman import (zeeman_splitting, get_zeeman_matrix)
from slothpy.angular_momentum.pseudo_spin_ito import (get_decomposition_in_z_total_angular_momentum_basis, get_decomposition_in_z_magnetic_momentum_basis, ito_real_decomp_matrix, 
                                                      ito_complex_decomp_matrix, get_soc_matrix_in_z_magnetic_momentum_basis, get_soc_matrix_in_z_total_angular_momentum_basis, 
                                                      get_zeeman_matrix_in_z_magnetic_momentum_basis, get_zeeman_matrix_in_z_total_angular_momentum_basis, 
                                                      matrix_from_ito_complex, matrix_from_ito_real)

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
        
    
    def __setitem__(self, key, value) -> None:

        value = np.array(value)

        if isinstance(key, str):

            try:
                with h5py.File(self._hdf5, 'r+') as file:
                    new_dataset = file.create_dataset(key, shape=value.shape, dtype=value.dtype)
                    new_dataset[:] = value[:]
                
                self.get_hdf5_groups_and_attributes()
                return
                    
            except Exception as e:
                error_type = type(e).__name__
                error_message = str(e)
                raise Exception(f'Error encountered while trying to set dataset {key} in .slt file: {self._hdf5}: {error_type}: {error_message}')
        
        elif isinstance(key, tuple) and len(key) == 2 and isinstance(key[0], str) and isinstance(key[1], str):
            
            try:
                with h5py.File(self._hdf5, 'r+') as file:
                    new_group = file.create_group(key[0])
                    new_dataset = new_group.create_dataset(key[1], shape=value.shape, dtype=value.dtype)
                    new_dataset[:] = value[:]
                
                self.get_hdf5_groups_and_attributes()
                return
                    
            except Exception as e:
                error_type = type(e).__name__
                error_message = str(e)
                raise Exception(f'Error encountered while trying to set group "{key[0]}" and dataset "{key[1]}" in .slt file: {self._hdf5}: {error_type}: {error_message}')
        
        else:
            raise KeyError("Invalid key type. It has to be str or 2-tuple of str.")
        


    def __getitem__(self, key) -> Any:

        if isinstance(key, str):

            try:
                with h5py.File(self._hdf5, 'r') as file:

                    value = file[key][:]
                
                return value
                    
            except Exception as e:
                error_type = type(e).__name__
                error_message = str(e)
                raise Exception(f'Error encountered while trying to get dataset {key} from .slt file: {self._hdf5}: {error_type}: {error_message}')
        
        elif isinstance(key, tuple) and len(key) == 2 and isinstance(key[0], str) and isinstance(key[1], str):
            
            try:
                with h5py.File(self._hdf5, 'r') as file:

                    value = file[key[0]][key[1]][:]
                
                return value
                    
            except Exception as e:
                error_type = type(e).__name__
                error_message = str(e)
                raise Exception(f'Error encountered while trying to get group "{key[0]}" and dataset "{key[1]}" from .slt file: {self._hdf5}: {error_type}: {error_message}')
        
        else:
            raise KeyError("Invalid key type. It has to be str or 2-tuple of str.")

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
            raise Exception(f'Error encountered while trying to delete group {group} from .slt file: {self._hdf5}: {error_type}: {error_message}')

        self.get_hdf5_groups_and_attributes()
        

    def calculate_g_tensor_and_axes_doublet(self, group: str, doublets: np.ndarray, slt: str = None):
        
        try:
            g_tensor_list, magnetic_axes_list = calculate_g_tensor_and_axes_doublet(self._hdf5, group, doublets)

        except Exception as e:
            error_type = type(e).__name__
            error_message = str(e)
            raise Exception(f'Error encountered while trying to compute g-tensors and main magnetic axes from file: {self._hdf5} - group {group}: {error_type}: {error_message}')

        if slt is not None:
            try:
                with h5py.File(self._hdf5, 'r+') as file:
                    new_group = file.create_group(f'{slt}_g_tensors_axes')
                    new_group.attrs['Description'] = f'Group({slt}) containing g-tensors of doublets and their magnetic axes calculated from group: {group}.'
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
                raise Exception(f'Error encountered while trying to save g-tensors and magnetic axes to file: {self._hdf5} - group {slt}: {error_type}: {error_message}')

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
            raise Exception(f'Error encountered while trying to compute M(T,H) from file: {self._hdf5} - group {group}: {error_type}: {error_message}')
        
        if slt is not None:
            try:
                with h5py.File(self._hdf5, 'r+') as file:
                    new_group = file.create_group(f'{slt}_magnetisation')
                    new_group.attrs['Description'] = f'Group({slt}) containing M(T,H) magnetisation calculated from group: {group}.'
                    mth_dataset = new_group.create_dataset(f'{slt}_mth', shape=(mth_array.shape[0], mth_array.shape[1]), dtype=np.float64)
                    mth_dataset.attrs['Description'] = f'Dataset containing M(T,H) magnetisation (T - rows, H - columns) calculated from group: {group}.'
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
                raise Exception(f'Error encountered while trying to save M(T,H) to file: {self._hdf5} - group {slt}: {error_type}: {error_message}')

        return mth_array
     
        
    def calculate_chitht(self, group: str, fields: np.ndarray, states_cutoff: int, temperatures: np.ndarray, num_cpu: int, num_of_points: int, delta_h: np.float64, exp: bool = False, T: bool = True, grid: np.ndarray = None, slt: str = None) -> np.ndarray:

        fields = np.array(fields)
        temperatures = np.array(temperatures)

        if T:
            chi_name = 'chiT(H,T)'
            chi_file = 'chit'
        else:
            chi_name = 'chi(H,T)'
            chi_file = 'chi'            

        if isinstance(grid, int):
            grid = lebedev_laikov_grid(grid)
        else:
            grid = np.array(grid)

        try:
            chitht_array = chitht(self._hdf5, group, fields, states_cutoff, temperatures, num_cpu, num_of_points, delta_h, exp, T, grid)
        except Exception as e:
            error_type = type(e).__name__
            error_message = str(e)
            raise Exception(f'Error encountered while trying to compute {chi_name} from file: {self._hdf5} - group {group}: {error_type}: {error_message}')
        
        if slt is not None:
            try:
                with h5py.File(self._hdf5, 'r+') as file:
                    new_group = file.create_group(f'{slt}_susceptibility')
                    new_group.attrs['Description'] = f'Group({slt}) containing {chi_name} magnetic susceptibility calculated from group: {group}.'
                    chitht_dataset = new_group.create_dataset(f'{slt}_{chi_file}ht', shape=(chitht_array.shape[0], chitht_array.shape[1]), dtype=np.float64)
                    chitht_dataset.attrs['Description'] = f'Dataset containing {chi_name} magnetic susceptibility (H - rows, T - columns) calculated from group: {group}.'
                    fields_dataset = new_group.create_dataset(f'{slt}_fields', shape=(fields.shape[0],), dtype=np.float64)
                    fields_dataset.attrs['Description'] = f'Dataset containing magnetic field H values used in simulation of {chi_name} from group: {group}.'
                    temperatures_dataset = new_group.create_dataset(f'{slt}_temperatures', shape=(temperatures.shape[0],), dtype=np.float64)
                    temperatures_dataset.attrs['Description'] = f'Dataset containing temperature T values used in simulation of {chi_name} from group: {group}.'

                    chitht_dataset[:,:] = chitht_array[:,:]
                    fields_dataset[:] = fields[:]
                    temperatures_dataset[:] = temperatures[:]
            
                self.get_hdf5_groups_and_attributes()

            except Exception as e:
                error_type = type(e).__name__
                error_message = str(e)
                raise Exception(f'Error encountered while trying to save {chi_name} to file: {self._hdf5} - group {slt}: {error_type}: {error_message}')

        return chitht_array 
        
    
    def calculate_chit_tensorht(self, group: str, fields: np.ndarray, states_cutoff: int, temperatures: np.ndarray, num_cpu: int, num_of_points: int, delta_h: np.float64, T: bool = True, slt: str = None):

        fields = np.array(fields)
        temperatures = np.array(temperatures)

        try:
            chit_tensorht_array = chit_tensorht(self._hdf5, group, fields, states_cutoff, temperatures, num_cpu, num_of_points, delta_h, T)
        except Exception as e:
            error_type = type(e).__name__
            error_message = str(e)
            raise Exception(f'Error encountered while trying to compute chi_tensor(H,T) from file: {self._hdf5} - group {group}: {error_type}: {error_message}')
        
        if slt is not None:
            try:
                with h5py.File(self._hdf5, 'r+') as file:
                    new_group = file.create_group(f'{slt}_susceptibility_tensor')
                    new_group.attrs['Description'] = f'Group({slt}) containing chiT_tensor(H,T) Van Vleck susceptibility tensor calculated from group: {group}.'
                    chit_tensorht_dataset = new_group.create_dataset(f'{slt}_chit_tensorht', shape=(chit_tensorht_array.shape[0], chit_tensorht_array.shape[1],3,3), dtype=np.float64)
                    chit_tensorht_dataset.attrs['Description'] = f'Dataset containing chiT_tensor(H,T) Van Vleck susceptibility tensor (H, T, 3, 3) calculated from group: {group}.'
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
                raise Exception(f'Error encountered while trying to save chiT(H,T) to file: {self._hdf5} - group {slt}: {error_type}: {error_message}')

        return chit_tensorht_array
    

    def soc_energies_cm_1(self, group: str, num_of_states: int = None, slt: str = None) -> np.ndarray:

        try:
            soc_energies_array = get_soc_energies_cm_1(self._hdf5, group, num_of_states)
        except Exception as e:
            error_type = type(e).__name__
            error_message = str(e)
            raise Exception(f'Error encountered while trying to get SOC energies from file: {self._hdf5} - group {group}: {error_type}: {error_message}')
        
        if slt is not None:
            try:
                with h5py.File(self._hdf5, 'r+') as file:
                    new_group = file.create_group(f'{slt}_soc_energies')
                    new_group.attrs['Description'] = f'Group({slt}) containing SOC (Spin-Orbit Coupling) energies calculated from group: {group}.'
                    soc_energies_dataset = new_group.create_dataset(f'{slt}_soc_energies', shape=(soc_energies_array.shape[0],), dtype=np.float64)
                    soc_energies_dataset.attrs['Description'] = f'Dataset containing SOC (Spin-Orbit Coupling) energies calculated from group: {group}.'

                    soc_energies_dataset[:] = soc_energies_array[:]
            
                self.get_hdf5_groups_and_attributes()

            except Exception as e:
                error_type = type(e).__name__
                error_message = str(e)
                raise Exception(f'Error encountered while trying to save SOC (Spin-Orbit Coupling) energies to file: {self._hdf5} - group {slt}: {error_type}: {error_message}')

        return soc_energies_array
    

    def states_magnetic_momenta(self, group: str, states: np.ndarray = None, rotation = None, slt: str = None):

        states = np.array(states)

        try:
            states, magnetic_momenta_array = get_states_magnetic_momenta(self._hdf5, group, states, rotation)
        except Exception as e:
            error_type = type(e).__name__
            error_message = str(e)
            raise Exception(f'Error encountered while trying to get states magnetic momenta from file: {self._hdf5} - group {group}: {error_type}: {error_message}')
        
        if slt is not None:
            try:
                with h5py.File(self._hdf5, 'r+') as file:
                    new_group = file.create_group(f'{slt}_states_magnetic_momenta')
                    new_group.attrs['Description'] = f'Group({slt}) containing states magnetic momenta calculated from group: {group}.'
                    magnetic_momenta_dataset = new_group.create_dataset(f'{slt}_magnetic_momenta', shape=(magnetic_momenta_array.shape[0],magnetic_momenta_array.shape[1]), dtype=np.float64)
                    magnetic_momenta_dataset.attrs['Description'] = f'Dataset containing states magnetic momenta (0-x,1-y,2-z) calculated from group: {group}.'
                    states_dataset = new_group.create_dataset(f'{slt}_states', shape=(states.shape[0],), dtype=np.int64)
                    states_dataset.attrs['Description'] = f'Dataset containing indexes of states used in simulation of magnetic momenta from group: {group}.'

                    magnetic_momenta_dataset[:] = magnetic_momenta_array[:]
                    states_dataset[:] = states[:]
            
                self.get_hdf5_groups_and_attributes()

            except Exception as e:
                error_type = type(e).__name__
                error_message = str(e)
                raise Exception(f'Error encountered while trying to save states magnetic momenta to file: {self._hdf5} - group {slt}: {error_type}: {error_message}')

        return magnetic_momenta_array
    

    def states_total_angular_momenta(self, group: str, states: np.ndarray = None, rotation = None, slt: str = None):

        states = np.array(states)

        try:
            states, total_angular_momenta_array = get_states_total_angular_momneta(self._hdf5, group, states, rotation)
        except Exception as e:
            error_type = type(e).__name__
            error_message = str(e)
            raise Exception(f'Error encountered while trying to get states total angular momenta from file: {self._hdf5} - group {group}: {error_type}: {error_message}')
        
        if slt is not None:
            try:
                with h5py.File(self._hdf5, 'r+') as file:
                    new_group = file.create_group(f'{slt}_states_total_angular_momenta')
                    new_group.attrs['Description'] = f'Group({slt}) containing states total angular momenta calculated from group: {group}.'
                    total_angular_momenta_dataset = new_group.create_dataset(f'{slt}_total_angular_momenta', shape=(total_angular_momenta_array.shape[0],total_angular_momenta_array.shape[1]), dtype=np.float64)
                    total_angular_momenta_dataset.attrs['Description'] = f'Dataset containing states total angular momenta (0-x,1-y,2-z) calculated from group: {group}.'
                    states_dataset = new_group.create_dataset(f'{slt}_states', shape=(states.shape[0],), dtype=np.int64)
                    states_dataset.attrs['Description'] = f'Dataset containing indexes of states used in simulation of total angular momenta from group: {group}.'

                    total_angular_momenta_dataset[:] = total_angular_momenta_array[:]
                    states_dataset[:] = states[:]
            
                self.get_hdf5_groups_and_attributes()

            except Exception as e:
                error_type = type(e).__name__
                error_message = str(e)
                raise Exception(f'Error encountered while trying to save states total angular momenta to file: {self._hdf5} - group {slt}: {error_type}: {error_message}')

        return total_angular_momenta_array
    

    def calculate_zeeman_splitting(self, group: str, states_cutoff: int, num_of_states: int, fields: np.ndarray, grid: np.ndarray, num_cpu: int, average: bool = False, slt: str = None):

        fields = np.array(fields)

        if isinstance(grid, int):
            grid = lebedev_laikov_grid(grid)
            average = True

        grid = np.array(grid)

        try:
            zeeman_array = zeeman_splitting(self._hdf5, group, states_cutoff, num_of_states, fields, grid, num_cpu, average)
        except Exception as e:
            error_type = type(e).__name__
            error_message = str(e)
            raise Exception(f'Error encountered while trying to compute Zeeman splitting from file: {self._hdf5} - group {group}: {error_type}: {error_message}')
        
        if slt is not None:
            try:
                with h5py.File(self._hdf5, 'r+') as file:
                    new_group = file.create_group(f'{slt}_zeeman_splitting')
                    new_group.attrs['Description'] = f'Group({slt}) containing Zeeman splitting calculated from group: {group}.'
                    zeeman_splitting_dataset = new_group.create_dataset(f'{slt}_zeeman', shape=zeeman_array.shape, dtype=np.float64)
                    if average:
                        zeeman_splitting_dataset.attrs['Description'] = f'Dataset containing Zeeman splitting averaged over grid of directions with shape: (field, energy) calculated from group: {group}.'
                    else:
                        zeeman_splitting_dataset.attrs['Description'] = f'Dataset containing Zeeman splitting with shape: (orientation, field, energy) calculated from group: {group}.'
                    fields_dataset = new_group.create_dataset(f'{slt}_fields', shape=(fields.shape[0],), dtype=np.float64)
                    fields_dataset.attrs['Description'] = f'Dataset containing magnetic field H values used in simulation of Zeeman splitting from group: {group}.'
                    if average:
                        orientations_dataset = new_group.create_dataset(f'{slt}_orientations', shape=(grid.shape[0], grid.shape[1]), dtype=np.float64)
                        orientations_dataset.attrs['Description'] = f'Dataset containing magnetic field orientation grid with weights used in simulation of averaged Zeeman splitting from group: {group}.'
                        orientations_dataset[:] = grid[:]
                    else:
                        orientations_dataset = new_group.create_dataset(f'{slt}_orientations', shape=(grid.shape[0], 3), dtype=np.float64)
                        orientations_dataset.attrs['Description'] = f'Dataset containing orientations of magnetic field used in simulation of Zeeman splitting from group: {group}.'
                        orientations_dataset[:] = grid[:,:3]

                    zeeman_splitting_dataset[:,:] = zeeman_array[:,:]
                    fields_dataset[:] = fields[:]
                    
            
                self.get_hdf5_groups_and_attributes()

            except Exception as e:
                error_type = type(e).__name__
                error_message = str(e)
                raise Exception(f'Error encountered while trying to save Zeeman splitting to file: {self._hdf5} - group {slt}: {error_type}: {error_message}')

        return zeeman_array
    

    def total_angular_momenta_matrix(self, group: str, states_cutoff: np.int64 = None, rotation = None, slt: str = None):

        if (not isinstance(states_cutoff, np.int)) or (states_cutoff < 0):
            raise ValueError(f'Invalid states cutoff, set it to positive integer or 0 for all states.')

        try:
            total_angular_momenta_matrix_array = get_total_angular_momneta_matrix(self._hdf5, group, states_cutoff, rotation)
        except Exception as e:
            error_type = type(e).__name__
            error_message = str(e)
            raise Exception(f'Error encountered while trying to get total angular momenta matrix from file: {self._hdf5} - group {group}: {error_type}: {error_message}')
        
        if slt is not None:
            try:
                with h5py.File(self._hdf5, 'r+') as file:
                    new_group = file.create_group(f'{slt}_total_angular_momenta_matrix')
                    new_group.attrs['Description'] = f'Group({slt}) containing total angular momenta calculated from group: {group}.'
                    total_angular_momenta_matrix_dataset = new_group.create_dataset(f'{slt}_total_angular_momenta_matrix', shape=total_angular_momenta_matrix_array.shape, dtype=np.complex128)
                    total_angular_momenta_matrix_dataset.attrs['Description'] = f'Dataset containing total angular momenta matrix (0-x, 1-y, 2-z) calculated from group: {group}.'
                    states_dataset = new_group.create_dataset(f'{slt}_states', shape=(total_angular_momenta_matrix_array.shape[1],), dtype=np.int64)
                    states_dataset.attrs['Description'] = f'Dataset containing states indexes of total angular momenta matrix from group: {group}.'

                    total_angular_momenta_matrix_dataset[:] = total_angular_momenta_matrix_array[:]
                    states_dataset[:] = np.arange(total_angular_momenta_matrix_array.shape[1], dtype=np.int64)
            
                self.get_hdf5_groups_and_attributes()

            except Exception as e:
                error_type = type(e).__name__
                error_message = str(e)
                raise Exception(f'Error encountered while trying to save total angular momenta matrix to file: {self._hdf5} - group {slt}: {error_type}: {error_message}')
            
        return total_angular_momenta_matrix_array
            

    def magnetic_momenta_matrix(self, group: str, states_cutoff: np.ndarray = None, rotation = None, slt: str = None):

        if (not isinstance(states_cutoff, np.int)) or (states_cutoff < 0):
            raise ValueError(f'Invalid states cutoff, set it to positive integer or 0 for all states.')

        try:
            magnetic_momenta_matrix_array = get_magnetic_momenta_matrix(self._hdf5, group, states_cutoff, rotation)
        except Exception as e:
            error_type = type(e).__name__
            error_message = str(e)
            raise Exception(f'Error encountered while trying to get total angular momenta matrix from file: {self._hdf5} - group {group}: {error_type}: {error_message}')
        
        if slt is not None:
            try:
                with h5py.File(self._hdf5, 'r+') as file:
                    new_group = file.create_group(f'{slt}_magnetic_momenta_matrix')
                    new_group.attrs['Description'] = f'Group({slt}) containing magnetic momenta calculated from group: {group}.'
                    magnetic_momenta_matrix_dataset = new_group.create_dataset(f'{slt}_magnetic_momenta_matrix', shape=magnetic_momenta_matrix_array.shape, dtype=np.complex128)
                    magnetic_momenta_matrix_dataset.attrs['Description'] = f'Dataset containing magnetic momenta matrix (0-x, 1-y, 2-z) calculated from group: {group}.'
                    states_dataset = new_group.create_dataset(f'{slt}_states', shape=(magnetic_momenta_matrix_array.shape[1],), dtype=np.int64)
                    states_dataset.attrs['Description'] = f'Dataset containing states indexes of magnetic momenta matrix from group: {group}.'

                    magnetic_momenta_matrix_dataset[:] = magnetic_momenta_matrix_array[:]
                    states_dataset[:] = np.arange(magnetic_momenta_matrix_array.shape[1], dtype=np.int64)
            
                self.get_hdf5_groups_and_attributes()

            except Exception as e:
                error_type = type(e).__name__
                error_message = str(e)
                raise Exception(f'Error encountered while trying to save magnetic momenta matrix to file: {self._hdf5} - group {slt}: {error_type}: {error_message}')
            
        return magnetic_momenta_matrix_array
            
    
    def decomposition_in_z_magnetic_momentum_basis(self, group, start_state, stop_state, rotation = None, slt: str = None):

        if (not isinstance(stop_state, int)) or (stop_state < 0):
            raise ValueError(f'Invalid states number, set it to positive integer or 0 for all states.')

        try:
            decomposition = get_decomposition_in_z_magnetic_momentum_basis(self._hdf5, group, start_state, stop_state, rotation)
        except Exception as e:
            error_type = type(e).__name__
            error_message = str(e)
            raise Exception(f'Error encountered while trying to get decomposition in "z" magnetic momentum basis of SOC matrix from file: {self._hdf5} - group {group}: {error_type}: {error_message}')
        
        if slt is not None:
            try:
                with h5py.File(self._hdf5, 'r+') as file:
                    new_group = file.create_group(f'{slt}_magnetic_decomposition')
                    new_group.attrs['Description'] = f'Group({slt}) containing decomposition in "z" magnetic momentum basis of SOC matrix calculated from group: {group}.'
                    decomposition_dataset = new_group.create_dataset(f'{slt}_magnetic_momenta_matrix', shape=decomposition.shape, dtype=np.float64)
                    decomposition_dataset.attrs['Description'] = f'Dataset containing % decomposition (rows - SO-states, columns - basis) in "z" magnetic momentum basis of SOC matrix from group: {group}.'
                    states_dataset = new_group.create_dataset(f'{slt}_pseudo_spin_states', shape=(decomposition.shape[0],), dtype=np.float64)
                    states_dataset.attrs['Description'] = f'Dataset containing Sz pseudo-spin states corresponding to the decomposition of SOC matrix from group: {group}.'

                    decomposition_dataset[:] = decomposition[:]
                    dim = (decomposition.shape[1] - 1)/2
                    states_dataset[:] = np.arange(-dim, dim+1, step=1, dtype=np.float64)
            
                self.get_hdf5_groups_and_attributes()

            except Exception as e:
                error_type = type(e).__name__
                error_message = str(e)
                raise Exception(f'Error encountered while trying to save decomposition in "z" magnetic momentum basis of SOC matrix to file: {self._hdf5} - group {slt}: {error_type}: {error_message}')
    
        return decomposition


    def decomposition_in_z_angular_momentum_basis(self, group, start_state, stop_state, rotation = None, slt: str = None):

        if (not isinstance(stop_state, int)) or (stop_state < 0):
            raise ValueError(f'Invalid states number, set it to positive integer or 0 for all states.')

        try:
            decomposition = get_decomposition_in_z_total_angular_momentum_basis(self._hdf5, group, start_state, stop_state, rotation)
        except Exception as e:
            error_type = type(e).__name__
            error_message = str(e)
            raise Exception(f'Error encountered while trying to get decomposition in "z" total angular momentum basis of SOC matrix from file: {self._hdf5} - group {group}: {error_type}: {error_message}')
        
        if slt is not None:
            try:
                with h5py.File(self._hdf5, 'r+') as file:
                    new_group = file.create_group(f'{slt}_total_angular_decomposition')
                    new_group.attrs['Description'] = f'Group({slt}) containing decomposition in "z" total angular momentum basis of SOC matrix calculated from group: {group}.'
                    decomposition_dataset = new_group.create_dataset(f'{slt}_magnetic_momenta_matrix', shape=decomposition.shape, dtype=np.float64)
                    decomposition_dataset.attrs['Description'] = f'Dataset containing % decomposition (rows SO-states, columns - basis) in "z" total angular momentum basis of SOC matrix from group: {group}.'
                    states_dataset = new_group.create_dataset(f'{slt}_pseudo_spin_states', shape=(decomposition.shape[0],), dtype=np.float64)
                    states_dataset.attrs['Description'] = f'Dataset containing Sz pseudo-spin states corresponding to the decomposition of SOC matrix from group: {group}.'

                    decomposition_dataset[:] = decomposition[:]
                    dim = (decomposition.shape[1] - 1)/2
                    states_dataset[:] = np.arange(-dim, dim+1, step=1, dtype=np.float64)
            
                self.get_hdf5_groups_and_attributes()

            except Exception as e:
                error_type = type(e).__name__
                error_message = str(e)
                raise Exception(f'Error encountered while trying to save decomposition in "z" total angular momentum basis of SOC matrix to file: {self._hdf5} - group {slt}: {error_type}: {error_message}')
            
        return decomposition
            

    def soc_crystal_field_parameters(self, group,  start_state, stop_state, order, even_order: bool = True, imaginary: bool = False, magnetic: bool = False, rotation = None, slt: str = None):

        if magnetic:
            try:
                soc_matrix = get_soc_matrix_in_z_magnetic_momentum_basis(self._hdf5, group, start_state, stop_state, rotation)
            except Exception as e:
                error_type = type(e).__name__
                error_message = str(e)
                raise Exception(f'Error encountered while trying to get SOC matrix in "z" magnetic momentum basis from file: {self._hdf5} - group {group}: {error_type}: {error_message}')

        else:
            try:
                soc_matrix = get_soc_matrix_in_z_total_angular_momentum_basis(self._hdf5, group, start_state, stop_state, rotation)
            except Exception as e:
                error_type = type(e).__name__
                error_message = str(e)
                raise Exception(f'Error encountered while trying to get SOC matrix in "z" total angular momentum basis from file: {self._hdf5} - group {group}: {error_type}: {error_message}')
            
        dim = (soc_matrix.shape[1] - 1)/2

        if order > 2*dim:
            raise ValueError(f'Order of ITO parameters exeeds 2S. Set it less or equal.')
        
        if imaginary:
            try:
                cfp = ito_complex_decomp_matrix(soc_matrix, order, even_order)
            except Exception as e:
                error_type = type(e).__name__
                error_message = str(e)
                raise Exception(f'Error encountered while trying to ITO decompose SOC matrix in "z" magnetic momentum basis from file: {self._hdf5} - group {group}: {error_type}: {error_message}')
        else:
            try:
                cfp = ito_real_decomp_matrix(soc_matrix, order, even_order)
            except Exception as e:
                error_type = type(e).__name__
                error_message = str(e)
                raise Exception(f'Error encountered while trying to ITO decompose SOC matrix in "z" total angular momentum basis from file: {self._hdf5} - group {group}: {error_type}: {error_message}')
        
        cfp_return = cfp

        if slt is not None:

            cfp = np.array(cfp)

            try:
                with h5py.File(self._hdf5, 'r+') as file:
                    new_group = file.create_group(f'{slt}_soc_ito_decomposition')
                    new_group.attrs['Description'] = f'Group({slt}) containing ITO decomposition in "z" pseudo-spin basis of SOC matrix calculated from group: {group}.'
                    cfp_dataset = new_group.create_dataset(f'{slt}_ito_parameters', shape=cfp.shape, dtype=cfp.dtype)
                    cfp_dataset.attrs['Description'] = f'Dataset containing ITO decomposition in "z" pseudo-spin basis of SOC matrix from group: {group}.'
                    states_dataset = new_group.create_dataset(f'{slt}_pseudo_spin_states', shape=(1,), dtype=np.float64)
                    states_dataset.attrs['Description'] = f'Dataset containing S pseudo-spin number corresponding to the decomposition of SOC matrix from group: {group}.'

                    cfp_dataset[:] = cfp[:]
                    states_dataset[:] = dim
            
                self.get_hdf5_groups_and_attributes()

            except Exception as e:
                error_type = type(e).__name__
                error_message = str(e)
                raise Exception(f'Error encountered while trying to save ITO decomposition in "z" pseudo-spin basis of SOC matrix to file: {self._hdf5} - group {slt}: {error_type}: {error_message}')

        return cfp_return        
            

    def zeeman_matrix_ito_decpomosition(self, group,  start_state, stop_state, field, orientation, order, imaginary: bool = False, magnetic: bool = False, rotation = None, slt: str = None):

        if magnetic:
            try:
                zeeman_matrix = get_zeeman_matrix_in_z_magnetic_momentum_basis(self._hdf5, group, field, orientation, start_state, stop_state, rotation)
            except Exception as e:
                error_type = type(e).__name__
                error_message = str(e)
                raise Exception(f'Error encountered while trying to get Zeeman matrix in "z" magnetic momentum basis from file: {self._hdf5} - group {group}: {error_type}: {error_message}')
        else:
            try:
                zeeman_matrix = get_zeeman_matrix_in_z_total_angular_momentum_basis(self._hdf5, group, field, orientation, start_state, stop_state, rotation)
            except Exception as e:
                error_type = type(e).__name__
                error_message = str(e)
                raise Exception(f'Error encountered while trying to get Zeeman matrix in "z" total angular momentum basis from file: {self._hdf5} - group {group}: {error_type}: {error_message}')
            
        dim = (zeeman_matrix.shape[1] - 1)/2

        if order > 2*dim:
            raise ValueError(f'Order of ITO parameters exeeds 2S. Set it less or equal.')
        
        if imaginary:
            try:
                cfp = ito_complex_decomp_matrix(zeeman_matrix, order)
            except Exception as e:
                error_type = type(e).__name__
                error_message = str(e)
                raise Exception(f'Error encountered while trying to ITO decompose Zeeman matrix in "z" magnetic momentum basis from file: {self._hdf5} - group {group}: {error_type}: {error_message}')
        else:
            try:
                cfp = ito_real_decomp_matrix(zeeman_matrix, order)
            except Exception as e:
                error_type = type(e).__name__
                error_message = str(e)
                raise Exception(f'Error encountered while trying to ITO decompose Zeeman matrix in "z" total angular momentum basis from file: {self._hdf5} - group {group}: {error_type}: {error_message}')
        
        cfp_return = cfp

        if slt is not None:

            cfp = np.array(cfp)

            try:
                with h5py.File(self._hdf5, 'r+') as file:
                    new_group = file.create_group(f'{slt}_zeeman_ito_decomposition')
                    new_group.attrs['Description'] = f'Group({slt}) containing ITO decomposition in "z" pseudo-spin basis of Zeeman matrix calculated from group: {group}.'
                    cfp_dataset = new_group.create_dataset(f'{slt}_ito_parameters', shape=cfp.shape, dtype=cfp.dtype)
                    cfp_dataset.attrs['Description'] = f'Dataset containing ITO decomposition in "z" pseudo-spin basis of Zeeman matrix from group: {group}.'
                    states_dataset = new_group.create_dataset(f'{slt}_pseudo_spin_states', shape=(1,), dtype=np.float64)
                    states_dataset.attrs['Description'] = f'Dataset containing S pseudo-spin number corresponding to the decomposition of Zeeman matrix from group: {group}.'

                    cfp_dataset[:] = cfp[:]
                    states_dataset[:] = dim
            
                self.get_hdf5_groups_and_attributes()

            except Exception as e:
                error_type = type(e).__name__
                error_message = str(e)
                raise Exception(f'Error encountered while trying to save ITO decomposition in "z" pseudo-spin basis of Zeeman matrix to file: {self._hdf5} - group {slt}: {error_type}: {error_message}')

        return cfp_return


    def zeeman_matrix(self, group: str, states_cutoff, field, orientation, slt: str = None):

        if (not isinstance(states_cutoff, np.int)) or (states_cutoff < 0):
            raise ValueError(f'Invalid states cutoff, set it to positive integer or 0 for all states.')

        try:
            zeeman_matrix_array = get_zeeman_matrix(self._hdf5, group, states_cutoff, field, orientation)
        except Exception as e:
            error_type = type(e).__name__
            error_message = str(e)
            raise Exception(f'Error encountered while trying to get Zeeman matrix from file: {self._hdf5} - group {group}: {error_type}: {error_message}')
        
        if slt is not None:
            try:
                with h5py.File(self._hdf5, 'r+') as file:
                    new_group = file.create_group(f'{slt}_zeeman_matrix')
                    new_group.attrs['Description'] = f'Group({slt}) containing Zeeman matrix calculated from group: {group}.'
                    zeeman_matrix_dataset = new_group.create_dataset(f'{slt}_zeeman_matrix', shape=zeeman_matrix_array.shape, dtype=np.complex128)
                    zeeman_matrix_dataset.attrs['Description'] = f'Dataset containing Zeeman matrix calculated from group: {group}.'
                    states_dataset = new_group.create_dataset(f'{slt}_states', shape=(zeeman_matrix_array.shape[1],), dtype=np.int64)
                    states_dataset.attrs['Description'] = f'Dataset containing states indexes of Zeeman matrix from group: {group}.'

                    zeeman_matrix_dataset[:] = zeeman_matrix_array[:]
                    states_dataset[:] = np.arange(zeeman_matrix_array.shape[1], dtype=np.int64)
            
                self.get_hdf5_groups_and_attributes()

            except Exception as e:
                error_type = type(e).__name__
                error_message = str(e)
                raise Exception(f'Error encountered while trying to save Zeeman matrix to file: {self._hdf5} - group {slt}: {error_type}: {error_message}')
     
        return zeeman_matrix_array


    def matrix_from_ito(self, name, imaginary: bool = False, dataset: str = None, pseudo_spin: str = None, slt: str = None, matrix_type: str = None):

        if (dataset is not None) and (pseudo_spin is not None) and pseudo_spin > 0:

            try: 
                J = pseudo_spin
                coefficients = self[f'{name}', f'{dataset}']
                if imaginary:
                    matrix = matrix_from_ito_complex(J, coefficients)
                else:
                    matrix = matrix_from_ito_real(J, coefficients)

            except Exception as e:
                error_type_1 = type(e).__name__
                error_message_1 = str(e)
                error_print_1 = f"{error_type_1}: {error_message_1}"
                raise Exception(f'Failed to form matrix from ITO parameters.\n Error(s) encountered while trying compute the matrix: {error_print_1}')
    
        else:

            try:
                J = self[f'{name}_zeeman_ito_decomposition', f'{name}_pseudo_spin_states']
                coefficients = self[f'{name}_zeeman_ito_decomposition', f'{name}_ito_parameters']
        
            except Exception as e:
                error_type_2 = type(e).__name__
                error_message_2 = str(e)
                error_print_2 = f"{error_type_2}: {error_message_2}"
                try:
                    J = self[f'{name}_soc_ito_decomposition', f'{name}_pseudo_spin_states']
                    coefficients = self[f'{name}_soc_ito_decomposition', f'{name}_ito_parameters']

                except Exception as e:
                    error_type_3 = type(e).__name__
                    error_message_3 = str(e)
                    error_print_3 = f"{error_type_3}: {error_message_3}"
                    raise Exception(f'Failed to form matrix from ITO parameters.\n Error(s) encountered while trying compute the matrix: {error_print_2}, {error_print_3}')

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
                with h5py.File(self._hdf5, 'r+') as file:
                    new_group = file.create_group(f'{slt}_matrix')
                    new_group.attrs['Description'] = f'Group({slt}) containing matrix from ITO calculated from group: {name}.'
                    matrix_dataset = new_group.create_dataset(f'{slt}_matrix', shape=matrix.shape, dtype=np.complex128)
                    matrix_dataset.attrs['Description'] = f'Dataset containing matrix from ITO calculated from group: {name}.'
                    states_dataset = new_group.create_dataset(f'{slt}_pseudo_spin_states', shape=(1,), dtype=np.float64)
                    states_dataset.attrs['Description'] = f'Dataset containing S pseudo-spin number corresponding to the matrix from group: {name}.'

                    matrix_dataset[:] = matrix[:]
                    states_dataset[:] = J_result
            
                self.get_hdf5_groups_and_attributes()

            except Exception as e:
                error_type = type(e).__name__
                error_message = str(e)
                raise Exception(f'Error encountered while trying to save matrix from ITO to file: {self._hdf5} - group {slt}: {error_type}: {error_message}')

        return matrix


    def soc_zeem_in_angular_magnetic_momentum_basis(self, group, start_state, stop_state, matrix_type, basis_type, rotation = None, field = None, orientation = None, slt: str = None):

        if (matrix_type not in ['zeeman', 'soc']) or (basis_type not in ['angular', 'magnetic']):
            raise ValueError(f'Only valid matrix_type are "soc" or "zeeman" and basis_type are "angular" or "magnetic"')
        
        if matrix_type == 'zeeman' and ((field is None) or (orientation is None)):
            raise ValueError(f'For Zeeman matrix provide filed value and orientation.')
        
        try:
            if matrix_type == 'zeeman':
                if basis_type == 'angular':
                    matrix = get_zeeman_matrix_in_z_total_angular_momentum_basis(self._hdf5, group, field, orientation, start_state, stop_state, rotation)
                elif basis_type == 'magnetic':
                    matrix = get_zeeman_matrix_in_z_magnetic_momentum_basis(self._hdf5, group, field, orientation, start_state, stop_state, rotation)
            elif matrix_type == 'soc':
                if basis_type == 'angular':
                    matrix = get_soc_matrix_in_z_total_angular_momentum_basis(self._hdf5, group, start_state, stop_state, rotation)
                elif basis_type == 'magnetic':
                    matrix = get_soc_matrix_in_z_magnetic_momentum_basis(self._hdf5, group, start_state, stop_state, rotation)                
        except Exception as e:
            error_type = type(e).__name__
            error_message = str(e)
            raise Exception(f'Error encountered while trying to get {matrix_type} matrix from file in {basis_type} momentum basis: {self._hdf5} - group {group}: {error_type}: {error_message}')
        
        if slt is not None:
            try:
                with h5py.File(self._hdf5, 'r+') as file:
                    new_group = file.create_group(f'{slt}_{matrix_type}_matrix_in_{basis_type}_basis')
                    new_group.attrs['Description'] = f'Group({slt}) containing {matrix_type} matrix in {basis_type} momentum "z" basis calculated from group: {group}.'
                    matrix_dataset = new_group.create_dataset(f'{slt}_matrix', shape=matrix.shape, dtype=np.complex128)
                    matrix_dataset.attrs['Description'] = f'Dataset containing {matrix_type} matrix in {basis_type} momentum "z" basis calculated from group: {group}.'
                    states_dataset = new_group.create_dataset(f'{slt}_states', shape=(matrix.shape[1],), dtype=np.int64)
                    states_dataset.attrs['Description'] = f'Dataset containing states indexes of {matrix_type} matrix in {basis_type} momentum "z" basis from group: {group}.'

                    matrix_dataset[:] = matrix[:]
                    states_dataset[:] = np.arange(matrix.shape[1], dtype=np.int64)
            
                self.get_hdf5_groups_and_attributes()

            except Exception as e:
                error_type = type(e).__name__
                error_message = str(e)
                raise Exception(f'Error encountered while trying to save {matrix_type} matrix in {basis_type} momentum "z" basis to file: {self._hdf5} - group {slt}: {error_type}: {error_message}')
            
        return matrix
    

    def calculate_mag_3d(self, group: str, states_cutoff: int, field: np.ndarray, spherical_grid: int, temperature: np.float64, num_cpu: int, slt: str = None):

        temperature = np.array([temperature],dtype=np.float64)

        try:
            x, y, z = mag_3d(self._hdf5, group, states_cutoff, field, spherical_grid, temperature, num_cpu)
        except Exception as e:
            error_type = type(e).__name__
            error_message = str(e)
            raise Exception(f'Error encountered while trying to compute 3D magnetisation from file: {self._hdf5} - group {group}: {error_type}: {error_message}')
        
        if slt is not None:
            try:
                with h5py.File(self._hdf5, 'r+') as file:
                    new_group = file.create_group(f'{slt}_3d_magnetisation')
                    new_group.attrs['Description'] = f'Group({slt}) containing 3D magnetisation calculated from group: {group}.'
                    mag_3d_dataset = new_group.create_dataset(f'{slt}_mag_3d', shape=(3,x.shape[1],x.shape[2]), dtype=np.float64)
                    mag_3d_dataset.attrs['Description'] = f'Dataset containing 3D magnetisation as meshgird (0-x,1-y,2-z) arrays over sphere (T: {temperature} K, H: {field} T) calculated from group: {group}.'

                    mag_3d_dataset[0,:,:] = x[0]
                    mag_3d_dataset[1,:,:] = y[0]
                    mag_3d_dataset[2,:,:] = z[0]
            
                self.get_hdf5_groups_and_attributes()

            except Exception as e:
                error_type = type(e).__name__
                error_message = str(e)
                raise Exception(f'Error encountered while trying to save 3D magnetisation to file: {self._hdf5} - group {slt}: {error_type}: {error_message}')

        return x, y, z
    

    def calculate_chit_3d(self, group: str, field: np.float64, states_cutoff: int, temperature: np.float64, num_cpu: int, num_of_points: int, delta_h: np.float64, spherical_grid: int, exp: bool = False, T: bool = True, slt: str = None):

        temperature = np.array([temperature], dtype=np.float64)

        try:
            x, y, z = chit_3d(self._hdf5, group, field, states_cutoff, temperature, num_cpu, num_of_points, delta_h, spherical_grid, exp, T)
        except Exception as e:
            error_type = type(e).__name__
            error_message = str(e)
            raise Exception(f'Error encountered while trying to compute 3D magnetic susceptibility from file: {self._hdf5} - group {group}: {error_type}: {error_message}')
        
        if slt is not None:

            if T:
                chi_file = 'chit'
            else:
                chi_file = 'chi'

            try:
                with h5py.File(self._hdf5, 'r+') as file:
                    new_group = file.create_group(f'{slt}_3d_susceptibility')
                    new_group.attrs['Description'] = f'Group({slt}) containing 3D magnetic susceptibility calculated from group: {group}.'
                    chit_3d_dataset = new_group.create_dataset(f'{slt}_{chi_file}_3d', shape=(3,x.shape[1],x.shape[2]), dtype=np.float64)
                    chit_3d_dataset.attrs['Description'] = f'Dataset containing 3D magnetic susceptibility as meshgird (0-x,1-y,2-z) arrays over sphere (T: {temperature} K, H: {field} T) calculated from group: {group}.'

                    chit_3d_dataset[0,:,:] = x[0]
                    chit_3d_dataset[1,:,:] = y[0]
                    chit_3d_dataset[2,:,:] = z[0]
            
                self.get_hdf5_groups_and_attributes()

            except Exception as e:
                error_type = type(e).__name__
                error_message = str(e)
                raise Exception(f'Error encountered while trying to save 3D magneic susceptibility to file: {self._hdf5} - group {slt}: {error_type}: {error_message}')

        return x, y, z


        
                  




