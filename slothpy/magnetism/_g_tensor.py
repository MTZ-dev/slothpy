import numpy as np
from slothpy.general_utilities.io import (
    _get_soc_ener_and_soc_ang_mom_from_hdf5,
)
from slothpy.general_utilities._math_expresions import _mag_mom_from_ang_mom


def _calculate_g_tensor_and_axes_doublet(
    filename: str, group: str, doublets: np.ndarray[np.int64]
):
    doublets = np.array(doublets, dtype=np.int64)
    g_tensor_list = np.zeros((doublets.size, 4), dtype=np.float64)
    magnetic_axes_list = np.zeros((doublets.size, 3, 3), dtype=np.float64)

    _, ang_mom = _get_soc_ener_and_soc_ang_mom_from_hdf5(filename, group)

    for index, doublet in enumerate(doublets):
        mag_mom = _mag_mom_from_ang_mom(ang_mom, 2 * doublet, 2 * doublet + 2)

        a_tensor = np.zeros((3, 3), dtype=np.float64)

        for i in range(3):
            for j in range(3):
                a_tensor[i, j] = 0.5 * np.trace(mag_mom[i] @ mag_mom[j]).real

        a_tensor = (a_tensor + a_tensor.T) / 2

        g_tensor_squared, magnetic_axes = np.linalg.eigh(a_tensor)

        for i in range(3):
            if g_tensor_squared[i] < 0:
                g_tensor_squared[i] = 0
        g_tensor = 2 * np.sqrt(g_tensor_squared)

        # Simply flip new "z" axis if wrong handednes
        if np.linalg.det(magnetic_axes) < 0:
            magnetic_axes[:, 2] = -magnetic_axes[:, 2]

        g_tensor_list[index, 0] = doublet
        g_tensor_list[index, 1:4] = g_tensor
        magnetic_axes_list[index, :, :] = magnetic_axes

    return g_tensor_list, magnetic_axes_list
