import slothpy as slt
import numpy as np
import h5py

a = slt.compound_from_molcas(".", "DyCo_test_hdf5", "molcas_test1234", ".", "DyCo_test_hdf5_bas0")

# print(a)

# b = slt.compound_from_slt(".", "DyCo_test_hdf5")

# print(b)

axes, g_ten = a.g_tensor_and_axes_doublet("molcas_test1234", np.array([0,1,2,3,4,5,6,7]))

print(axes)
print(g_ten)

# print(b)

# with h5py.File("NdCoNO2.slt", 'r') as file:
#     data = file["third_g_tensors_axes"]["third_g_tensors"][:]

# print(data)

# b = slt.compound_from_slt(".", "NdCoNO2")
# b.delete_group("third_g_tensors_axes")