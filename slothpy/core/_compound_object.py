from os import path
from typing import Any
import h5py
from slothpy.magnetism.g_tensor import calculate_g_tensor_and_axes_doublet
import numpy as np

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
            print(f'Error encountered while trying to delete group {group} from .slt file {self._hdf5}: {error_type}: {error_message}')
            return

        self.get_hdf5_groups_and_attributes()

    def g_tensor_and_axes_doublet(self, group: str, doublets: np.ndarray, slt: str = None):
        
        try:
            g_tensor_list, magnetic_axes_list = calculate_g_tensor_and_axes_doublet(self._hdf5, group, doublets)
        except Exception as e:
            error_type = type(e).__name__
            error_message = str(e)
            print(f'Error encountered while trying to compute g-tensors and main magnetic axes from file {self._hdf5} - group {group}: {error_type}: {error_message}')
            return

        if slt is not None:

            with h5py.File(self._hdf5, 'r+') as file:
                new_group = file.create_group(f'{slt}_g_tensors_axes')
                new_group.attrs['Description'] = f'Group({slt}) containing g-tensors of doublets and their magnetic axes calulated from group {group}.'
                tensors = new_group.create_dataset(f'{slt}_g_tensors', shape=(g_tensor_list.shape[0], g_tensor_list.shape[1]), dtype=np.float64)
                tensors.attrs['Description'] = f'Dataset containing number of doublet and respective g-tensors from group {group}.'
                axes = new_group.create_dataset(f'{slt}_axes', shape=(magnetic_axes_list.shape[0], magnetic_axes_list.shape[1], magnetic_axes_list.shape[2]), dtype=np.float64)
                axes.attrs['Description'] = f'Dataset containing rotation matrices from initial coordinate system to magnetic axes of respective g-tensors from group {group}.'
                tensors[:,:] = g_tensor_list[:,:]
                axes[:,:,:] = magnetic_axes_list[:,:,:]
        
            self.get_hdf5_groups_and_attributes()

        return g_tensor_list, magnetic_axes_list

        