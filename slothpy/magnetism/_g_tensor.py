from numpy import (
    ndarray,
    array,
    zeros,
    trace,
    ascontiguousarray,
    sqrt,
    float64,
    int64,
)
from numpy.linalg import eigh, det
from numba import jit
from numba.types import Tuple
from slothpy.general_utilities.io import (
    _get_soc_energies_and_soc_angular_momenta_from_hdf5,
)
from slothpy.general_utilities._math_expresions import (
    _magnetic_momenta_from_angular_momenta,
)


@jit(
    "Tuple((float64[:,:],float64[:,:,:]))(complex128[:,:,:], int64[:])",
    nopython=True,
    cache=True,
    nogil=True,
    fastmath=True,
)
def _calculate_g_tensor_and_axes_doublet(angular_momenta, doublets):
    g_tensor_list = zeros((doublets.size, 4), dtype=float64)
    magnetic_axes_list = zeros((doublets.size, 3, 3), dtype=float64)

    for index, doublet in enumerate(doublets):
        magnetic_momenta = ascontiguousarray(
            _magnetic_momenta_from_angular_momenta(
                angular_momenta, 2 * doublet, 2 * doublet + 2
            )
        )

        a_tensor = zeros((3, 3), dtype=float64)

        for i in range(3):
            for j in range(3):
                a_tensor[i, j] = (
                    0.5 * trace(magnetic_momenta[i] @ magnetic_momenta[j]).real
                )

        a_tensor = (a_tensor + a_tensor.T) / 2

        g_tensor_squared, magnetic_axes = eigh(a_tensor)

        for i in range(3):
            if g_tensor_squared[i] < 0:
                g_tensor_squared[i] = 0
        g_tensor = 2 * sqrt(g_tensor_squared)

        # Simply flip new "z" axis if wrong handednes
        if det(magnetic_axes) < 0:
            magnetic_axes[:, 2] = -magnetic_axes[:, 2]

        g_tensor_list[index, 0] = doublet
        g_tensor_list[index, 1:4] = g_tensor
        magnetic_axes_list[index, :, :] = magnetic_axes

    return g_tensor_list, magnetic_axes_list


def _g_tensor_and_axes_doublet(
    filename: str, group: str, doublets: ndarray[int64]
):
    doublets = array(doublets, dtype=int64)

    _, angular_momenta = _get_soc_energies_and_soc_angular_momenta_from_hdf5(
        filename, group
    )

    return _calculate_g_tensor_and_axes_doublet(angular_momenta, doublets)
