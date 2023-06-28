import slothpy as slt
import numpy as np
import h5py

# a = slt.compound_from_orca(".", "NdCoNO2", "third", ".", "NdCoNO2_cas_super_tight_cas.out")

# print(a)

# b = slt.compound_from_slt(".", "NdCoNO2")

# print(b)

# axes, g_ten = b.g_tensor_and_axes_doublet("third", np.array([0,1,2,3,4,5]), "third")

# print(axes)
# print(g_ten)

# print(b)

with h5py.File("NdCoNO2.slt", 'r') as file:
    data = file["third_g_tensors_axes"]["third_g_tensors"][:]

print(data)

# b = slt.compound_from_slt(".", "NdCoNO2")
# b.delete_group("third_g_tensors_axes")