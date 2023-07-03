from os import path
from typing import Any
import h5py
import numpy as np
from slothpy.magnetism.g_tensor import calculate_g_tensor_and_axes_doublet
from slothpy.magnetism.magnetisation import mth
from slothpy.magnetism.susceptibility import (chitht, chit_tensorht)
from slothpy.general_utilities.grids_over_hemisphere import lebedev_laikov_grid
from slothpy.general_utilities.io import (get_soc_energies_cm_1, get_states_magnetic_momenta, get_states_total_angular_momneta,
                                           get_total_angular_momneta_matrix, get_magnetic_momenta_matrix, get_soc_momenta_and_energies_from_hdf5)
from slothpy.magnetism.zeeman import zeeman_splitting
from slothpy.general_utilities.math_expresions import hermitian_x_in_basis_of_hermitian_y, decomposition_of_hermitian_matrix

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
     
        
    def calculate_chitht(self, group: str, fields: np.ndarray, states_cutoff: int, temperatures: np.ndarray, num_cpu: int, num_of_points: int, delta_h: np.float64, exp: bool = False, T: bool = True, grid: np.ndarray = None, slt: str = None) -> np.ndarray:

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
    

    def soc_energies_cm_1(self, group: str, num_of_states: int = None, slt: str = None) -> np.ndarray:

        try:
            soc_energies_array = get_soc_energies_cm_1(self._hdf5, group, num_of_states)
        except Exception as e:
            error_type = type(e).__name__
            error_message = str(e)
            print(f'Error encountered while trying to get SOC energies from file: {self._hdf5} - group {group}: {error_type}: {error_message}')
            return
        
        if slt is not None:
            try:
                with h5py.File(self._hdf5, 'r+') as file:
                    new_group = file.create_group(f'{slt}_soc_energies')
                    new_group.attrs['Description'] = f'Group({slt}) containing SOC (Spin-Orbit Coupling) energies calulated from group: {group}.'
                    soc_energies_dataset = new_group.create_dataset(f'{slt}_soc_energies', shape=(soc_energies_array.shape[0],), dtype=np.float64)
                    soc_energies_dataset.attrs['Description'] = f'Dataset containing SOC (Spin-Orbit Coupling) energies calulated from group: {group}.'

                    soc_energies_dataset[:] = soc_energies_array[:]
            
                self.get_hdf5_groups_and_attributes()

            except Exception as e:
                error_type = type(e).__name__
                error_message = str(e)
                print(f'Error encountered while trying to save SOC (Spin-Orbit Coupling) energies to file: {self._hdf5} - group {slt}: {error_type}: {error_message}')
                return

        return soc_energies_array
    

    def states_magnetic_momenta(self, group: str, states: np.ndarray = None, slt: str = None):

        states = np.array(states)

        try:
            states, magnetic_momenta_array = get_states_magnetic_momenta(self._hdf5, group, states)
        except Exception as e:
            error_type = type(e).__name__
            error_message = str(e)
            print(f'Error encountered while trying to get states magnetic momenta from file: {self._hdf5} - group {group}: {error_type}: {error_message}')
            return
        
        if slt is not None:
            try:
                with h5py.File(self._hdf5, 'r+') as file:
                    new_group = file.create_group(f'{slt}_states_magnetic_momenta')
                    new_group.attrs['Description'] = f'Group({slt}) containing states magnetic momenta calulated from group: {group}.'
                    magnetic_momenta_dataset = new_group.create_dataset(f'{slt}_magnetic_momenta', shape=(magnetic_momenta_array.shape[0],magnetic_momenta_array.shape[1]), dtype=np.float64)
                    magnetic_momenta_dataset.attrs['Description'] = f'Dataset containing states magnetic momenta (0-x,1-y,2-z) calulated from group: {group}.'
                    states_dataset = new_group.create_dataset(f'{slt}_states', shape=(states.shape[0],), dtype=np.int64)
                    states_dataset.attrs['Description'] = f'Dataset containing indexes of states used in simulation of magnetic momenta from group: {group}.'

                    magnetic_momenta_dataset[:] = magnetic_momenta_array[:]
                    states_dataset[:] = states[:]
            
                self.get_hdf5_groups_and_attributes()

            except Exception as e:
                error_type = type(e).__name__
                error_message = str(e)
                print(f'Error encountered while trying to save states magnetic momenta to file: {self._hdf5} - group {slt}: {error_type}: {error_message}')
                return

        return states, magnetic_momenta_array
    

    def states_total_angular_momenta(self, group: str, states: np.ndarray = None, slt: str = None):

        states = np.array(states)

        try:
            states, total_angular_momenta_array = get_states_total_angular_momneta(self._hdf5, group, states)
        except Exception as e:
            error_type = type(e).__name__
            error_message = str(e)
            print(f'Error encountered while trying to get states total angular momenta from file: {self._hdf5} - group {group}: {error_type}: {error_message}')
            return
        
        if slt is not None:
            try:
                with h5py.File(self._hdf5, 'r+') as file:
                    new_group = file.create_group(f'{slt}_states_total_angular_momenta')
                    new_group.attrs['Description'] = f'Group({slt}) containing states total angular momenta calulated from group: {group}.'
                    total_angular_momenta_dataset = new_group.create_dataset(f'{slt}_total_angular_momenta', shape=(total_angular_momenta_array.shape[0],total_angular_momenta_array.shape[1]), dtype=np.float64)
                    total_angular_momenta_dataset.attrs['Description'] = f'Dataset containing states total angular momenta (0-x,1-y,2-z) calulated from group: {group}.'
                    states_dataset = new_group.create_dataset(f'{slt}_states', shape=(states.shape[0],), dtype=np.int64)
                    states_dataset.attrs['Description'] = f'Dataset containing indexes of states used in simulation of total angular momenta from group: {group}.'

                    total_angular_momenta_dataset[:] = total_angular_momenta_array[:]
                    states_dataset[:] = states[:]
            
                self.get_hdf5_groups_and_attributes()

            except Exception as e:
                error_type = type(e).__name__
                error_message = str(e)
                print(f'Error encountered while trying to save states total angular momenta to file: {self._hdf5} - group {slt}: {error_type}: {error_message}')
                return

        return states, total_angular_momenta_array
    

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
            print(f'Error encountered while trying to compute Zeeman splitting from file: {self._hdf5} - group {group}: {error_type}: {error_message}')
            return
        
        if slt is not None:
            try:
                with h5py.File(self._hdf5, 'r+') as file:
                    new_group = file.create_group(f'{slt}_zeeman_splitting')
                    new_group.attrs['Description'] = f'Group({slt}) containing Zeeman splitting calulated from group: {group}.'
                    zeeman_splitting_dataset = new_group.create_dataset(f'{slt}_zeeman', shape=zeeman_array.shape, dtype=np.float64)
                    if average:
                        zeeman_splitting_dataset.attrs['Description'] = f'Dataset containing Zeeman splitting averaged over grid of directions with shape: (field, energy) calulated from group: {group}.'
                    else:
                        zeeman_splitting_dataset.attrs['Description'] = f'Dataset containing Zeeman splitting with shape: (orientation, field, energy) calulated from group: {group}.'
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
                print(f'Error encountered while trying to save Zeeman splitting to file: {self._hdf5} - group {slt}: {error_type}: {error_message}')
                return

        return zeeman_array
    

    def total_angular_momenta_matrix(self, group: str, states_cutoff: np.ndarray = None, slt: str = None):

        if (not isinstance(states_cutoff, np.int)) or (states_cutoff < 0):
            raise ValueError(f'Invalid states cutoff, set it to positive integer or 0 for all states.')

        try:
            total_angular_momenta_matrix_array = get_total_angular_momneta_matrix(self._hdf5, group, states_cutoff)
        except Exception as e:
            error_type = type(e).__name__
            error_message = str(e)
            print(f'Error encountered while trying to get total angular momenta matrix from file: {self._hdf5} - group {group}: {error_type}: {error_message}')
            return
        
        if slt is not None:
            try:
                with h5py.File(self._hdf5, 'r+') as file:
                    new_group = file.create_group(f'{slt}_total_angular_momenta_matrix')
                    new_group.attrs['Description'] = f'Group({slt}) containing total angular momenta calulated from group: {group}.'
                    total_angular_momenta_matrix_dataset = new_group.create_dataset(f'{slt}_total_angular_momenta_matrix', shape=total_angular_momenta_matrix_array.shape, dtype=np.complex128)
                    total_angular_momenta_matrix_dataset.attrs['Description'] = f'Dataset containing total angular momenta matrix (0-x, 1-y, 2-z) calulated from group: {group}.'
                    states_dataset = new_group.create_dataset(f'{slt}_states', shape=(total_angular_momenta_matrix_array.shape[1],), dtype=np.int64)
                    states_dataset.attrs['Description'] = f'Dataset containing states indexes of total angular momenta matrix from group: {group}.'

                    print(total_angular_momenta_matrix_array)
                    total_angular_momenta_matrix_dataset[:] = total_angular_momenta_matrix_array[:]
                    states_dataset[:] = np.arange(total_angular_momenta_matrix_array.shape[1], dtype=np.int64)
            
                self.get_hdf5_groups_and_attributes()

            except Exception as e:
                error_type = type(e).__name__
                error_message = str(e)
                print(f'Error encountered while trying to save total angular momenta matrix to file: {self._hdf5} - group {slt}: {error_type}: {error_message}')
                return
            

    def magnetic_momenta_matrix(self, group: str, states_cutoff: np.ndarray = None, slt: str = None):

        if (not isinstance(states_cutoff, np.int)) or (states_cutoff < 0):
            raise ValueError(f'Invalid states cutoff, set it to positive integer or 0 for all states.')

        try:
            magnetic_momenta_matrix_array = get_magnetic_momenta_matrix(self._hdf5, group, states_cutoff)
        except Exception as e:
            error_type = type(e).__name__
            error_message = str(e)
            print(f'Error encountered while trying to get total angular momenta matrix from file: {self._hdf5} - group {group}: {error_type}: {error_message}')
            return
        
        if slt is not None:
            try:
                with h5py.File(self._hdf5, 'r+') as file:
                    new_group = file.create_group(f'{slt}_magnetic_momenta_matrix')
                    new_group.attrs['Description'] = f'Group({slt}) containing magnetic momenta calulated from group: {group}.'
                    magnetic_momenta_matrix_dataset = new_group.create_dataset(f'{slt}_magnetic_momenta_matrix', shape=magnetic_momenta_matrix_array.shape, dtype=np.complex128)
                    magnetic_momenta_matrix_dataset.attrs['Description'] = f'Dataset containing magnetic momenta matrix (0-x, 1-y, 2-z) calulated from group: {group}.'
                    states_dataset = new_group.create_dataset(f'{slt}_states', shape=(magnetic_momenta_matrix_array.shape[1],), dtype=np.int64)
                    states_dataset.attrs['Description'] = f'Dataset containing states indexes of magnetic momenta matrix from group: {group}.'

                    magnetic_momenta_matrix_dataset[:] = magnetic_momenta_matrix_array[:]
                    states_dataset[:] = np.arange(magnetic_momenta_matrix_array.shape[1], dtype=np.int64)
            
                self.get_hdf5_groups_and_attributes()

            except Exception as e:
                error_type = type(e).__name__
                error_message = str(e)
                print(f'Error encountered while trying to save magnetic momenta matrix to file: {self._hdf5} - group {slt}: {error_type}: {error_message}')
                return
            
    
    def decomposition_in_z_magnetic_momentum_basis(self, group, number_of_states, slt: str = None):

        magnetic_momenta, soc_energies  = get_soc_momenta_and_energies_from_hdf5(self._hdf5, group, number_of_states)

        soc_matrix = np.diag(soc_energies)

        soc_matrix = hermitian_x_in_basis_of_hermitian_y(soc_matrix, magnetic_momenta[2])

        decopmosition = decomposition_of_hermitian_matrix(soc_matrix)

        return decopmosition


    def decomposition_in_z_angular_momentum_basis():
        
        pass