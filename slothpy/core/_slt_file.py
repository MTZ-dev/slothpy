from h5py import File, Group, Dataset
from numpy import array
from slothpy.core._slothpy_exceptions import slothpy_exc
from slothpy._general_utilities._constants import (
    GREEN,
    BLUE,
    PURPLE,
    RESET,
)

class SltAttributes:
    def __init__(self, hdf5_file_path, item_path):
        self._hdf5 = hdf5_file_path
        self._item_path = item_path

    @slothpy_exc("SltFileError")
    def __getitem__(self, attr_name):
        with File(self._hdf5, 'r') as file:
            item = file[self._item_path]
            if attr_name in item.attrs:
                return item.attrs[attr_name]
            else:
                raise KeyError(f"Attribute {attr_name} not found.")

    @slothpy_exc("SltFileError")
    def __setitem__(self, attr_name, value):
        with File(self._hdf5, 'r+') as file:
            item = file[self._item_path]
            item.attrs[attr_name] = value

    @slothpy_exc("SltFileError")
    def __repr__(self):
        with File(self._hdf5, 'r+') as file:
            file[self._item_path]
            return f"<SltAttributes for '{self._item_path}' in {self._hdf5} file.>"

    @slothpy_exc("SltFileError")
    def __str__(self):
        with File(self._hdf5, 'a') as file:
            item = file[self._item_path]
            return str(dict(item.attrs)).rstrip()


class SltGroup:
    def __init__(self, hdf5_file_path, group_path, exists=True):
        self._hdf5 = hdf5_file_path
        self._group_path = group_path
        self.attributes = SltAttributes(hdf5_file_path, group_path)
        self._exists = exists

    @slothpy_exc("SltFileError")
    def __getitem__(self, key):
        full_path = f"{self._group_path}/{key}"
        with File(self._hdf5, 'r') as file:
            if full_path in file:
                item = file[full_path]
                if isinstance(item, Dataset):
                    return SltDataset(self._hdf5, full_path)
                elif isinstance(item, Group):
                    raise KeyError("Hierarchy only up to Group/Dataset or standalone Datasets are supported in .slt files.")
            else:
                raise KeyError(f"Dataset '{key}' doesn't exist in the group '{self._group_path}'.")

    @slothpy_exc("SltFileError")        
    def __setitem__(self, key, value):
        with File(self._hdf5, 'r+') as file:
            group = file.require_group(self._group_path)
            self._exists = True
            if key in group:
                raise KeyError(f"Dataset '{key}' already exists within the group '{self._group_path}'. Delete it manually to ensure your data safety.")
            group.create_dataset(key, data=array(value))

    @slothpy_exc("SltFileError")
    def __repr__(self): 
        if self._exists:
            return f"<SltGroup '{self._group_path}' in {self._hdf5} file.>"
        else:
            raise RuntimeError("This is a proxy group and it does not exist in the .slt file yet. Initialize it by setting dataset within it - group['new_dataset'] = value.")

    @slothpy_exc("SltFileError")
    def __str__(self):
        if self._exists:
            with File(self._hdf5, 'r+') as file:
                item = file[self._group_path]
                representation = "Group: " + BLUE + f"{self._group_path}" + RESET + " from File: " + GREEN + f"{self._hdf5}" + RESET
                for attribute_name, attribute_text in item.attrs.items():
                    representation += f"| {attribute_name}: {attribute_text} "
                representation += "\nDatasets: \n"
                for dataset_name, dataset in item.items():
                    representation += PURPLE + f"{dataset_name}" + RESET
                    for attribute_name, attribute_text in dataset.attrs.items():
                        representation += f" | {attribute_name}: {attribute_text} "
                    representation += "\n"
                return representation.rstrip()
        else:
            raise RuntimeError("This is a proxy group and it does not exist in the .slt file yet. Initialize it by setting dataset within it - group['new_dataset'] = value.")


    @slothpy_exc("SltFileError")
    def __delitem__(self, key):
        with File(self._hdf5, 'r+') as file:
            group = file[self._group_path]
            if key not in group:
                raise KeyError(f"Dataset '{key}' does not exist in the group '{self._group_path}'.")
            del group[key]
        
    @property
    def attr(self):
        """
        Property to mimic h5py's attribute access convention.
        """
        return self.attributes


class SltDataset:
    def __init__(self, hdf5_file_path, dataset_path):
        self._hdf5 = hdf5_file_path
        self._dataset_path = dataset_path
        self.attributes = SltAttributes(hdf5_file_path, dataset_path)

    @slothpy_exc("SltFileError")
    def __getitem__(self, slice_):
        with File(self._hdf5, 'r') as file:
            dataset = file[self._dataset_path]
            return dataset[slice_]
        
    @slothpy_exc("SltFileError")
    def __setitem__(self, slice_, value):
        with File(self._hdf5, 'r+') as file:
            dataset = file[self._dataset_path]
            dataset[slice_] = array(value)

    @slothpy_exc("SltFileError")
    def __repr__(self):
        with File(self._hdf5, 'r+') as file:
            file[self._dataset_path]
            return f"<SltDataset '{self._dataset_path}' in {self._hdf5} file.>"
    
    @slothpy_exc("SltFileError")
    def __str__(self):
        with File(self._hdf5, 'r+') as file:
            item = file[self._dataset_path]
            representation = "Dataset: " + PURPLE + f"{self._dataset_path}" + RESET + " from File: " + GREEN + f"{self._hdf5}" + RESET
            for attribute_name, attribute_text in item.attrs.items():
                representation += f" | {attribute_name}: {attribute_text}"
            return representation.rstrip()
        
    @property
    def attr(self):
        """
        Property to mimic h5py's attribute access convention.
        """
        return self.attributes