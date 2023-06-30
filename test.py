import os
os.environ['OMP_NUM_THREADS'] = '2'
import slothpy as slt
import numpy as np
import h5py
import matplotlib.pyplot as plt

# a = slt.compound_from_molcas(".", "DyCo_test_hdf5", "molcas_test1234", ".", "DyCo_test_hdf5_bas0")

# print(a)

b = slt.compound_from_slt(".", "DyCo_test_hdf5")

# print(b)

# axes, g_ten = a.g_tensor_and_axes_doublet("molcas_test1234", np.array([0,1,2,3,4,5,6,7]))

# print(axes)
# print(g_ten)

# print(b)

# with h5py.File("NdCoNO2.slt", 'r') as file:
#     data = file["third_g_tensors_axes"]["third_g_tensors"][:]

# print(data)

# b = slt.compound_from_slt(".", "NdCoNO2")
# b.delete_group("third_g_tensors_axes")

fields1 = np.linspace(0.0001, 7, 50)
temperatures1 = np.linspace(2, 2, 1)
temperatures2 = np.linspace(0.0001, 300, 120)
fields2 = np.linspace(0.1,0.1,1)
grid = np.loadtxt('grid.txt', usecols = (1,2,3,4))
# grid2 = np.loadtxt('grid2.txt', usecols = (1,2,3,4))
# grid3 = np.loadtxt('grid3.txt', usecols = (1,2,3,4))
temperatures3 = np.linspace(1,5,5)


# mth = b.calculate_mth("molcas_test1234", 402, fields1, grid, temperatures1, 6)

# for mh in mth:
#     plt.plot(fields1, mh)
#     for i in mh:
#         print(i)

# plt.show()

# chitht = b.calculate_chitht("molcas_test1234", fields2, 402, temperatures2, 4, 1, 0.0001)

# for chitt in chitht:
#     plt.plot(temperatures2, chitt)
#     for i in chitt:
#         print(i)

# plt.show()

sus_tensor = b.calculate_chit_tensorht("molcas_test1234", fields2, 402, temperatures2, 4, 4, 0.00001, slt="first_tensorek")

print(sus_tensor)