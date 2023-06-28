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

fields = np.linspace(0.0001, 7, 50)
temperatures1 = np.linspace(2, 2, 1)
temperatures2 = np.linspace(1, 300, 300)
grid = np.loadtxt('grid.txt', usecols = (1,2,3,4))
# grid2 = np.loadtxt('grid2.txt', usecols = (1,2,3,4))
# grid3 = np.loadtxt('grid3.txt', usecols = (1,2,3,4))
temperatures3 = np.linspace(1,5,5)


mth6 = b.calculate_mth("molcas_test1234", 32, fields, grid, temperatures1, 4, slt="second_magnetisation")

for mh in mth6:
    plt.plot(fields, mh)
    for i in mh:
        print(i)

plt.show()