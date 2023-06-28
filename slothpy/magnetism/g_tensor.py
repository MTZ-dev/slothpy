import numpy as np
from slothpy.general_utilities.io import get_soc_energies_and_soc_angular_momenta_from_hdf5

def calculate_g_tensor_and_axes_doublet(filename: str, group: str, doublets: np.ndarray):

    ge = 2.00231930436256
    doublets = doublets.astype(np.int64)
    g_tensor_list = np.zeros((doublets.size,4), dtype=np.float64)
    magnetic_axes_list = np.zeros((doublets.size,3,3), dtype=np.float64)
    index = 0

    _, sx, sy, sz, lx, ly, lz = get_soc_energies_and_soc_angular_momenta_from_hdf5(filename, group)

    for doublet in doublets:

        magnetic_moment = np.zeros((3,2,2), dtype=np.complex128)
        states = 2*doublet

        # Slice arrays based on states_cutoff
        sx_tmp = sx[states:states+2, states:states+2]
        sy_tmp = sy[states:states+2, states:states+2]
        sz_tmp = sz[states:states+2, states:states+2]
        lx_tmp = lx[states:states+2, states:states+2]
        ly_tmp = ly[states:states+2, states:states+2]
        lz_tmp = lz[states:states+2, states:states+2]

        # Compute and save magnetic momenta in a.u.
        magnetic_moment[0] =  -(ge * sx_tmp + lx_tmp)
        magnetic_moment[1] =  -(ge * sy_tmp + ly_tmp)
        magnetic_moment[2] =  -(ge * sz_tmp + lz_tmp)

        A_tensor = np.zeros((3,3), dtype=np.float64)

        for i in range(3):
            for j in range(3):
                A_tensor[i,j] = 0.5 * np.trace(magnetic_moment[i] @ magnetic_moment[j]).real

        A_tensor = (A_tensor + A_tensor.T)/2

        g_tensor_squared, magnetic_axes = np.linalg.eigh(A_tensor)

        for i in range(3):
            if g_tensor_squared[i] < 0:
                g_tensor_squared[i] = 0
        g_tensor = 2 * np.sqrt(g_tensor_squared)

        if np.linalg.det(magnetic_axes) < 0:
            magnetic_axes[:,1] = -magnetic_axes[:,1]

        g_tensor_list[index, 0] = doublet
        g_tensor_list[index, 1:4] = g_tensor
        magnetic_axes_list[index, :, :] = magnetic_axes

        index += 1

    return g_tensor_list, magnetic_axes_list