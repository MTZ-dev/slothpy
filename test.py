import time
import os
#os.environ['OMP_NUM_THREADS'] = '2'
import slothpy as slt
import numpy as np
import h5py
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from slothpy.general_utilities.math_expresions import normalize_grid_vectors


if __name__ == '__main__':

    fields = np.linspace(0.1, 7, 50, dtype=np.float64)
    temperatures = np.linspace(0.1,700, 60, dtype=np.float64)

    CeCoN3 = slt.compound_from_molcas(".", "Lore", "rot", ".", "rot_lorenzo_bas0_corr")
    CeCoN3 = slt.compound_from_molcas(".", "Lore", "no_rot", ".", "no_rot_lorenzo_bas0")
    CeCoN3 = slt.compound_from_slt(".", "Lore")

    start_time = time.perf_counter()

    rotation = np.array([[-0.46452073, 0.1162711, -0.87789608],
  [0.11809391,  -0.97435556,  -0.19153345],
  [-0.87765273,  -0.19264544, 0.43887745]])
    
    rotatione = rotation.T

    #print(CeCoN3.states_total_angular_momenta("VTZP", [0,2], rotation=rotatione))
    # matrix = CeCoN3.magnetic_momenta_matrix("VTZP", 6, rotation=rotatione)
    # matrix = matrix[:, 4:6, 4:6]
    # eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    # print(eigenvalues)
    #CeCoN3.decomposition_in_z_magnetic_momentum_basis("VTZP", 0, 5, rotation=rotatione)

    axes, gtensor = CeCoN3.calculate_g_tensor_and_axes_doublet("rot", [0,1])

    print(axes)
    print(gtensor)




    axes, gtensor = CeCoN3.calculate_g_tensor_and_axes_doublet("no_rot", [0,1])

    print(axes)
    print(gtensor)

    #mth = CeCoN3.calculate_mth("VTZP", 0, fields, 5, [2.,4.,6.,8.], 64, 1)

    end_time = time.perf_counter()

    print(f'{end_time - start_time} s')

    # for mt in mth:
    #     print("temp")
    #     for m in mt:
    #         print(m)


    # start_time = time.perf_counter()

    # CeCoN3.calculate_hemholtz_energy_3d("TZVP", 0, fields, 100, temperatures, 64, 2, slt="twoja_stara")

    # end_time = time.perf_counter()

    # print(f'{end_time - start_time} s')


    # start_time = time.perf_counter()

    # #CeCoN3.animate_energy_3d("TZVP", 14, np.array([0.1]), 100, 0.1, 700, 60, 3, 1, fps = 20, dpi = 100, filename="lolz", ticks=100)

    # CeCoN3.animate_3d("twoja_stara", 'hemholtz_energy', 'temperature', fps=10, dpi=100)

    # end_time = time.perf_counter()

    # print(f'{end_time - start_time} s')



#a = slt.compound_from_orca(".", "DyCo", "bas0", ".", "DyCo_supercell_1800_0_0_cas.out")
#a = slt.compound_from_orca(".", "anisoop", "SVP", ".", "NdCoNO2_TZVP_cas.out")
#a = slt.compound_from_molcas(".", "aniso_benchmark", "bas0", ".", "DyCo_benchmark_aniso")
# a = slt.compound_from_slt(".", "CeCoN3")

# a.plot_mag_3d("QZVPPP", colour_map_name="dark_rainbow", r_density = 50, c_density = 50,  ticks=2)

# fields = np.linspace(0.1,0.1,1)
# temperatures = np.linspace(1,300,300)

# print(a.calculate_chit_tensorht("molcas_test", 64, temperatures, fields, 1, 0.0001, 1, 1, exp=False))

# chitht = a.calculate_chitht("bas0", 512, temperatures, fields, 3, 0.0001, 1, 1, exp=False)
# chitht2 = a.calculate_chitht("bas0", 512, temperatures, fields, 3, 0.0001, 1, 1, exp=True)

# for chitt in chitht:
#     plt.plot(temperatures, chitt)

# for chitt2 in chitht2:
#     plt.plot(temperatures, chitt2)


# plt.show()



# start_time = time.perf_counter()

# CeCoN3.calculate_hemholtz_energy_3d("TZVP", 0, fields, 100, temperatures, 16, 1, slt="twoja_stara")

# end_time = time.perf_counter()



# elapsed_time = (end_time - start_time)

# # Print the elapsed time
# print(f"Elapsed time: {elapsed_time} s")

# b_k_q = a.soc_crystal_field_parameters("bas0", 0, 15, 14, slt="wololol", even_order=False)

# for k in b_k_q:
#     print(f'{k[1]} {k[2]}')

# matrix = a.matrix_from_ito('wololo')

# eigenvalues, eigenvectors = np.linalg.eigh(matrix)
# print(eigenvalues)

# print(a.soc_energies_cm_1("bas0", 16))


# for mh in mth:
#     plt.plot(fields, mh)
# plt.show()

#a.plot_mth("test")

# a.animate_mag_3d('SVP',6,1,100,1,600,24, colour_map_name='NdCoNO222bpdo_l', lim_scalar=0.75, ticks=0)



# energy = a.calculate_hemholtz_energyth("molcas_test", 128, fields, 0, temperatures, 24)
# print(fields)
# print(energy)

# for eh in mth:
#     plt.plot(fields, eh)
#     for i in eh:
#         print(i)
# plt.show()

# x, y, z = a.calculate_chit_3d("SVP", [500,2], 14, [300,2], 24, 7, 0.00001, 100)

# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')

# max_array = np.array([np.max(x[0][0]), np.max(y[0][0]), np.max(z[0][0])])
# max = np.max(max_array)

# ax.plot_wireframe(x[0][0], y[0][0], z[0][0])
# ax.set_xlim(-max,max)
# ax.set_ylim(-max,max)
# ax.set_zlim(-max,max)
# ax.set_box_aspect([1, 1, 1])
# plt.show()

# x, y, z = a.calculate_mag_3d("SVP", 14, [1,2], 50, [1.,2], 4)

# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')

# max_array = np.array([np.max(x[0][0]), np.max(y[0][0]), np.max(z[0][0])])
# max = np.max(max_array)

# ax.plot_wireframe(x[0][1], y[0][1], z[0][1])
# ax.set_xlim(-max,max)
# ax.set_ylim(-max,max)
# ax.set_zlim(-max,max)
# ax.set_box_aspect([1, 1, 1])
# plt.show()


# x, y, z = a.calculate_hemholtz_energy_3d("molcas_test", 32, [0.1], 100, [1], 24)

# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')

# max_array = np.array([np.max(x[0][0]), np.max(y[0][0]), np.max(z[0][0])])
# max = np.max(max_array)

# ax.plot_wireframe(x[0][0], y[0][0], z[0][0])
# ax.set_xlim(-max,max)
# ax.set_ylim(-max,max)
# ax.set_zlim(-max,max)
# ax.set_box_aspect([1, 1, 1])
# plt.show()





# g_ten, axes = a.calculate_g_tensor_and_axes_doublet("SVP", np.array([0,1,2]))
# print(axes)
# print(g_ten)

# rotation1 = axes[1,:,:].T

# matrix = a.magnetic_momenta_matrix("SVP", 6, rotation1)
# print(matrix)

# eigenvalues, eigenvectors = np.linalg.eigh(matrix[2])
# print(eigenvalues)


# decomposition = a.decomposition_in_z_total_angular_momentum_basis("SVP", 0, 5)
# print(decomposition)

# magn_momenta = a.states_total_angular_momenta("SVP", np.arange(6), rotation1)
# print(magn_momenta)



# magnetisation = a.calculate_mth("SVP", 14, fields, 5, temperatures, 4)
# print(fields)
# print(magnetisation)

# magn_matrix = a.total_angular_momenta_matrix("bas0", 16)
# # print(magn_matrix)

# eigenvalues, eigenvectors = np.linalg.eigh(magn_matrix[2,:,:])

# print(eigenvalues)

# for i in range(3):
#     magn_matrix[i,:,:] = eigenvectors.conj().T @ magn_matrix[i,:,:] @ eigenvectors
#     # print(magn_matrix[i])

# J_plus = magn_matrix[0,:,:] + 1j * magn_matrix[1,:,:]
# J_minus = magn_matrix[0,:,:] - 1j * magn_matrix[1,:,:]
# J_z = magn_matrix[2,:,:]

# print(np.diagonal(J_plus @ J_minus + J_z @ J_z - J_z))

# print(J_minus)
# print(J_plus)
# print(J_z)

# c = np.zeros(magn_matrix.shape[1], dtype=np.complex128)
# c[0] = 1.

# for i in range(magn_matrix[1,:,:].shape[0]-1):
#     if np.real(magn_matrix[1,i,i+1]).any() > 1e-12:
#         c[i+1] = 1j*magn_matrix[1,i,i+1].conj()/np.abs(magn_matrix[1,i,i+1])/c[i].conj()
#     else:
#         c[i+1] = 1.

# phase_magn_matrix = magn_matrix

# for index in range(3):
#     for i in range(phase_magn_matrix.shape[1]):
#         for j in range(phase_magn_matrix.shape[1]):
#             phase_magn_matrix[index,i,j] = magn_matrix[index,i,j] * c[i].conj() * c[j]
    
#     # print(phase_magn_matrix[index])


# eigenvalues, eigenvectors = np.linalg.eigh(phase_magn_matrix[2,:,:])

# # print(eigenvalues)

# J_plus = phase_magn_matrix[0,:,:] + 1j * phase_magn_matrix[1,:,:]
# J_minus = phase_magn_matrix[0,:,:] - 1j * phase_magn_matrix[1,:,:]
# J_z = phase_magn_matrix[2,:,:]

# print(J_minus)
# print(J_plus)
# print(J_z)

# energy = a.soc_energies_cm_1("bas0", 25)
# print(energy)

# g_ten, axes = a.calculate_g_tensor_and_axes_doublet("SVP", np.array([0,1,2]))
# print(axes)
# print(g_ten)

# sus = a.calculate_chitht("SVP", np.array([20]), 64, np.linspace(1,300,10), 24, 5, 0.0001, exp=False, T = False)
# print(sus)

# print(np.linspace(1,300,10))
# sus_tensor = a.calculate_chit_tensorht("bas0", np.array([0.1]), 64, np.linspace(1,300,10), 4, 2, 0.0001)
# print(sus_tensor)

# b_k_q = a.soc_crystal_field_parameters("SVP", 0, 5, 5, slt="fforsednsnnnnnnnnn", even_order=False, magnetic = False, imaginary=True)
# for i in b_k_q:
#     print(i)

# rotation1 = axes[0,:,:].T


# total_momenta = a.states_total_angular_momenta("SVP", np.arange(6))
# print(total_momenta)

# magn_momenta = a.states_magnetic_momenta("SVP", np.arange(14))
# print(magn_momenta)

# matrix1 = a.soc_zeem_in_angular_magnetic_momentum_basis("SVP", 0, 5, 'zeeman', 'magnetic', field=1, orientation=np.array([0.,0.,1.]))
# matrix2 = a.matrix_from_ito("fforsednsnnnnnnnnn", imaginary=True)

# print(matrix1 * 219474.6)

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