import os
os.environ['OMP_NUM_THREADS'] = '2'
import slothpy as slt
import numpy as np
import h5py
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#a = slt.compound_from_molcas(".", "aniso_test_molcas", "bas0", ".", "DyCo_test_hdf5_bas0")
a = slt.compound_from_slt(".", "aniso_test")

# fields = np.linspace(0,7,10)
# temperatures = np.linspace(2,5,4)

x, y, z = a.calculate_chit_3d("SVP", 20, 14, 100, 24, 5, 0.0001, 200, slt="test_zapisu_chittt")

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

max_array = np.array([np.max(x), np.max(y), np.max(z)])
max = np.max(max_array)

ax.plot_wireframe(x, y, z)
ax.set_xlim(-max,max)
ax.set_ylim(-max,max)
ax.set_zlim(-max,max)
ax.set_box_aspect([1, 1, 1])
plt.show()

# x, y, z = a.calculate_mag_3d("SVP", 14, 1, 200, 400, 4)

# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')

# max_array = np.array([np.max(x), np.max(y), np.max(z)])
# max = np.max(max_array)

# ax.plot_wireframe(x, y, z)
# ax.set_xlim(-max,max)
# ax.set_ylim(-max,max)
# ax.set_zlim(-max,max)
# ax.set_box_aspect([1, 1, 1])
# plt.show()

# decomposition = a.decomposition_in_z_angular_momentum_basis("bas0", 0, 15)
# print(decomposition)

# magnetisation = a.calculate_mth("bas0", 64, fields, np.array([[0.,0.,1.,1]]), temperatures, 4)
# print(fields)
# print(magnetisation)

# magn_momenta = a.states_magnetic_momenta("SVP", np.arange(14))
# print(magn_momenta)

#total_momenta = a.states_total_angular_momenta("SVP", np.arange(14))
#print(total_momenta)

# magn_matrix = a.magnetic_momenta_matrix("SVP", 14)
# print(magn_matrix)

# energy = a.soc_energies_cm_1("bas0", 25)
# print(energy)

# g_ten, axes = a.calculate_g_tensor_and_axes_doublet("bas0", np.array([0,1,2]),)
# print(axes)
# print(g_ten)

# sus = a.calculate_chitht("bas0", np.array([0.1]), 64, np.linspace(1,300,10), 4, 3, 0.0001, exp=True)
# print(sus)

# print(np.linspace(1,300,10))
# sus_tensor = a.calculate_chit_tensorht("bas0", np.array([0.1]), 64, np.linspace(1,300,10), 4, 2, 0.0001)
# print(sus_tensor)

# b_k_q = a.soc_crystal_field_parameters("SVP", 0, 5, 5, slt="fforsednsnnnnnnnnn", even_order=False, magnetic = False, imaginary=True)
# for i in b_k_q:
#     print(i)

# matrix1 = a.soc_zeem_in_angular_magnetic_momentum_basis("SVP", 0, 5, 'soc', 'angular', field=1)
# matrix2 = a.matrix_from_ito("fforsednsnnnnnnnnn", imaginary=True)

# print(matrix1-matrix2)

# eigenvalues1, eigenvectors1 = np.linalg.eigh(matrix1)
# eigenvalues2, eigenvectors2 = np.linalg.eigh(matrix2)

# print(eigenvalues1*219474.6)
# print(eigenvalues2*219474.6)

# rot = np.array([[-0.50117407,  0.13460855, -0.8548129 ],
#   [ 0.74915045, -0.42693886, -0.50645515],
#   [-0.43312604, -0.89420565,  0.11312863]])

# rot_inv = np.linalg.inv(rot)

# print(rot_inv)













#a = slt.compound_from_slt(".", "DyCo")

#a = slt.compound_from_molcas(".", "DyrCffodrfB", "TZVP", ".", "DyCo_test_hdf5_bas0")


#a.zeeman_matrix("TZVP", 10, 1, np.array([0.,0.,1.]), slt="TZVP_from_real")
# matrix2 = a.soc_zeem_in_angular_magnetic_momentum_basis("TZVP", 0, 9, 'soc', 'angular', 1, np.array([0.,0.,1.]))
# coeff = a.soc_crystal_field_parameters("TZVP", 0, 9, 9, magnetic=False, even_order=True, slt="1200", imaginary=False)
#coeff = a.zeeman_matrix_ito_decpomosition("TZVP", 0, 9, 1, np.array([0.,0.,1.]), 9, slt="fraqes", magnetic=False, imaginary=True)
# matrix1 = a.matrix_from_ito("1200", imaginary=False, matrix_type='soc')

# print(matrix1-matrix2)

# eigenvalues1, eigenvectors1 = np.linalg.eigh(matrix1)
# eigenvalues2, eigenvectors2 = np.linalg.eigh(matrix2)

# print(eigenvalues1*219474.6)
# print(eigenvalues2*219474.6)

# print(coeff)

# print(coeff)

# print(matrix1-matrix2)
# print(eigenvectors1-eigenvectors2)

# a = slt.compound_from_molcas(".", "DyCo_test_hdf5", "molcas_test1234", ".", "DyCo_test_hdf5_bas0")

# print(a)

#b = slt.compound_from_orca(".", "CeCoN3", "TZVP", ".", "CeCoN3_TZVP_cas_restart.out")

# fields = np.linspace(0.0001, 7, 50)
# temperatures = np.linspace(1.8, 1.8, 1)

# b = slt.compound_from_slt(".", "CeCoN3")

# b["super_data_set3"] = [1,2,3]
# b["super_grupa3", "z super datasetem3"] = [1.,3.,4.5]

# c = b["super_data_set3"]
# d = b["TZVP_magnetisation", "z super datasetem3"]

# print(c)
# print(d)

# alfa = b.calculate_zeeman_splitting("TZVP", 16, 8, fields, np.array([[1.,0.,0.], [0.,0.,1.]]), 4, slt = "TZVP_1")

# beta = b.calculate_chitht("TZVP", fields, 252, temperatures, 4, 3, 0.0001, slt = 'chit')



# plt.plot(fields, alfa[0], "-")
# plt.show()

#lol = b.soc_crystal_field_parameters("TZVP", 6, 4, slt="cfpp")

#te = b.decomposition_in_z_angular_momentum_basis("TZVP", 14, slt="TVZP")


# mth = a.calculate_mth("TZVP_nevpt22", 14, fields, 5, temperatures, 4)

# for mh in mth:
#     plt.plot(fields, mh)
#     for i in mh:
#         print(i)
# plt.show()

# print(a)


# with h5py.File("NdCoNO2.slt", 'r+') as file:

#     a = file["TZVP_magnetisation"]["TZVP_mth"]
#     print(a[:].T)


# print(b)

# axes, g_ten = a.g_tensor_and_axes_doublet("molcas_test1234", np.array([0,1,2,3,4,5,6,7]))

# print(axes)
# print(g_ten)

# print(a)

# with h5py.File("NdCoNO2.slt", 'r') as file:
#     data = file["third_g_tensors_axes"]["third_g_tensors"][:]

# print(data)

# b = slt.compound_from_slt(".", "NdCoNO2")
# b.delete_group("third_g_tensors_axes")


# temperatures1 = np.linspace(1.8, 1.8, 1)
# temperatures2 = np.linspace(0.0001, 300, 150)
# fields2 = np.linspace(0.1,0.1,1)
#grid = np.loadtxt('grid.txt', usecols = (1,2,3,4))
# grid2 = np.loadtxt('grid2.txt', usecols = (1,2,3,4))
# grid3 = np.loadtxt('grid3.txt', usecols = (1,2,3,4))
# temperatures3 = np.linspace(1,5,5)


# for i in fields1:
#     print(i)

# for k in temperatures2:
#     print(k)

# b.soc_energies_cm_1("CeCoN3_nevpt2_TVZP", 0, slt = "test_energy")

# b.states_magnetic_momenta("CeCoN3_nevpt2_TVZP", slt = "test6")

# b.states_total_angular_momenta("CeCoN3_nevpt2_TVZP", slt = "test6")

# b.calculate_zeeman_splitting("CeCoN3_nevpt2_TVZP", 0, 13, fields1, 2, 2, slt="test3")

# print(b)

# b.magnetic_momenta_matrix("CeCoN3_nevpt2_TVZP", 0, slt = "test15")

# decomposition = b.decomposition_in_z_magnetic_momentum_basis("molcas_test1234", 16)

# print(decomposition)

# mth = b.calculate_mth("NdCoNO2_nevpt2_TZVP", 16, fields1, 6, temperatures1, 2)

# for mh in mth:
#     plt.plot(fields1, mh)
#     for i in mh:
#         print(i)

# plt.show()

# chitht = b.calculate_chitht("NdCoNO2_nevpt2_TZVP", fields2, 0, temperatures2, 2, 3, 0.0001)

# for chitt in chitht:
#     plt.plot(temperatures2, chitt)
#     for i in chitt:
#         print(i)

# plt.show()

#sus_tensor = b.calculate_chit_tensorht("molcas_test1234", fields2, 402, temperatures2, 4, 4, 0.00001, slt="first_tensorek")

#print(sus_tensor)