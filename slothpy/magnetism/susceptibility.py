import time
import multiprocessing
import threadpoolctl
from numpy import (
    ndarray,
    array,
    zeros,
    zeros_like,
    arange,
    dot,
    newaxis,
    float64,
    int64,
)
from numba import jit
from slothpy.magnetism._magnetisation import (
    _mth,
    _calculate_magnetization,
    _mag_3d,
)
from slothpy.general_utilities._math_expresions import _finite_diff_stencil
from slothpy.general_utilities.system import _get_num_of_processes
from slothpy.general_utilities.io import (
    get_soc_magnetic_momenta_and_energies_from_hdf5,
)
from slothpy.general_utilities._constants import MU_B_CM_3


def _chitht(
    filename: str,
    group: str,
    temperatures: ndarray[float64],
    fields: ndarray[float64],
    num_of_points: int,
    delta_h: float64,
    states_cutoff: int,
    num_cpu: int,
    num_threads: int,
    exp: bool = False,
    T: bool = True,
    grid: ndarray[float64] = None,
) -> ndarray[float64]:
    chitht_array = zeros(
        (fields.shape[0], temperatures.shape[0]), dtype=float64
    )

    # Default XYZ grid
    if grid is None or grid == None:
        grid = array(
            [
                [1.0, 0.0, 0.0, 0.3333333333333333],
                [0.0, 1.0, 0.0, 0.3333333333333333],
                [0.0, 0.0, 1.0, 0.3333333333333333],
            ],
            dtype=float64,
        )

    # Experimentalist model
    if (exp == True) or (num_of_points == 0):
        mth_array = _mth(
            filename,
            group,
            fields,
            grid,
            temperatures,
            states_cutoff,
            num_cpu,
            num_threads,
        )

        if T:
            for index_temp, temp in enumerate(temperatures):
                chitht_array[:, index_temp] = (
                    temp * mth_array[index_temp, :] * MU_B_CM_3 / fields
                )
        else:
            for index_temp in range(temperatures.size):
                chitht_array[:, index_temp] = (
                    mth_array[index_temp, :] * MU_B_CM_3 / fields
                )

    else:
        fields_diffs = (
            arange(-num_of_points, num_of_points + 1).astype(int64) * delta_h
        )[:, newaxis] + fields
        fields_diffs = fields_diffs.T.astype(float64)
        fields_diffs = fields_diffs.flatten()

        # Get M(T,H) for adjacent values of field
        mth_array = _mth(
            filename,
            group,
            fields_diffs,
            grid,
            temperatures,
            states_cutoff,
            num_cpu,
            num_threads,
        )

        stencil_coeff = _finite_diff_stencil(1, num_of_points, delta_h)

        mth_array = mth_array.reshape(
            (temperatures.size, fields.size, stencil_coeff.size)
        )

        if T:
            # Numerical derivative of M(T,H) around given field value
            for index, temp in enumerate(temperatures):
                chitht_array[:, index] = temp * dot(
                    mth_array[index, :, :], stencil_coeff
                )

        else:
            for index in range(temperatures.size):
                chitht_array[:, index] = dot(
                    mth_array[index, :, :], stencil_coeff
                )

        chitht_array = chitht_array * MU_B_CM_3

    return chitht_array


def chit_tensorht(
    filename: str,
    group: str,
    temperatures: ndarray,
    fields: ndarray,
    num_of_points: int,
    delta_h: float64,
    states_cutoff: int,
    num_cpu: int,
    num_threads: int,
    exp: bool = False,
    T: bool = True,
    rotation: ndarray[float64] = None,
) -> ndarray[float64]:
    chitht_tensor_array = zeros(
        (fields.shape[0], temperatures.shape[0], 3, 3), dtype=float64
    )

    # When passed to _mth activates tensor calculation
    grid = array([1])

    # Experimentalist model
    if (exp == True) or (num_of_points == 0):
        mht_tensor_array = _mth(
            filename,
            group,
            fields,
            grid,
            temperatures,
            states_cutoff,
            num_cpu,
            num_threads,
        )

        if T:
            chitht_tensor_array = (
                mht_tensor_array
                * temperatures[newaxis, :, :, :]
                * MU_B_CM_3
                / fields[:, newaxis, newaxis, newaxis]
            )
        else:
            chitht_tensor_array = (
                mht_tensor_array
                * MU_B_CM_3
                / fields[:, newaxis, newaxis, newaxis]
            )

    else:
        fields_diffs = (
            arange(-num_of_points, num_of_points + 1).astype(int64) * delta_h
        )[:, newaxis] + fields
        fields_diffs = fields_diffs.T.astype(float64)
        fields_diffs = fields_diffs.flatten()

        # Get M(T,H) for adjacent values of field
        mht_tensor_array = _mth(
            filename,
            group,
            fields_diffs,
            grid,
            temperatures,
            states_cutoff,
            num_cpu,
            num_threads,
        )

        stencil_coeff = _finite_diff_stencil(1, num_of_points, delta_h)

        mht_tensor_array = mht_tensor_array.reshape(
            (fields.size, stencil_coeff.size, temperatures.size, 3, 3)
        )

        if T:
            # Numerical derivative of M(T,H) around given field value
            for i in range(3):
                for j in range(3):
                    for index, temp in enumerate(temperatures):
                        chitht_tensor_array[:, index, i, j] = temp * dot(
                            mht_tensor_array[:, :, index, i, j], stencil_coeff
                        )

        else:
            for i in range(3):
                for j in range(3):
                    for index in range(temperatures.size):
                        chitht_tensor_array[:, index, i, j] = dot(
                            mht_tensor_array[:, :, index, i, j], stencil_coeff
                        )

        chitht_tensor_array = chitht_tensor_array * MU_B_CM_3

    return chitht_tensor_array


def chit_3d(
    filename: str,
    group: str,
    fields: ndarray,
    states_cutoff: int,
    temperatures: ndarray,
    num_cpu: int,
    num_threads: int,
    num_of_points: int,
    delta_h: float64,
    spherical_grid: int,
    exp: bool = False,
    T: bool = True,
):
    pass
    # if num_of_points < 0 or (not isinstance(num_of_points, int)):
    #     raise ValueError(
    #         f"Number of points for finite difference method has to be a"
    #         f" possitive integer!"
    #     )

    # bohr_magneton_to_cm3 = 0.5584938904  # Conversion factor for chi in cm3

    # # Experimentalist model
    # if (exp == True) or (num_of_points == 0):
    #     x, y, z = mag_3d(
    #         filename,
    #         group,
    #         states_cutoff,
    #         fields,
    #         spherical_grid,
    #         temperatures,
    #         num_cpu,
    #         num_threads,
    #     )

    #     for field_index, field in enumerate(fields):
    #         x[field_index] = x[field_index] / field * bohr_magneton_to_cm3
    #         y[field_index] = y[field_index] / field * bohr_magneton_to_cm3
    #         z[field_index] = z[field_index] / field * bohr_magneton_to_cm3

    #         if T:
    #             for temp_index, temp in enumerate(temperatures):
    #                 x[field_index, temp_index] = (
    #                     x[field_index, temp_index] * temp
    #                 )
    #                 y[field_index, temp_index] = (
    #                     y[field_index, temp_index] * temp
    #                 )
    #                 z[field_index, temp_index] = (
    #                     z[field_index, temp_index] * temp
    #                 )

    # else:
    #     dim = 2 * num_of_points + 1
    #     # Comments here modyfied!!!!
    #     mag_x = np.zeros(
    #         (
    #             fields.shape[0],
    #             temperatures.shape[0],
    #             spherical_grid,
    #             2 * spherical_grid,
    #             dim,
    #         ),
    #         dtype=np.float64,
    #     )
    #     mag_y = np.zeros(
    #         (
    #             fields.shape[0],
    #             temperatures.shape[0],
    #             spherical_grid,
    #             2 * spherical_grid,
    #             dim,
    #         ),
    #         dtype=np.float64,
    #     )
    #     mag_z = np.zeros(
    #         (
    #             fields.shape[0],
    #             temperatures.shape[0],
    #             spherical_grid,
    #             2 * spherical_grid,
    #             dim,
    #         ),
    #         dtype=np.float64,
    #     )
    #     x = np.zeros(
    #         (
    #             fields.shape[0],
    #             temperatures.shape[0],
    #             spherical_grid,
    #             2 * spherical_grid,
    #         ),
    #         dtype=np.float64,
    #     )
    #     y = np.zeros(
    #         (
    #             fields.shape[0],
    #             temperatures.shape[0],
    #             spherical_grid,
    #             2 * spherical_grid,
    #         ),
    #         dtype=np.float64,
    #     )
    #     z = np.zeros(
    #         (
    #             fields.shape[0],
    #             temperatures.shape[0],
    #             spherical_grid,
    #             2 * spherical_grid,
    #         ),
    #         dtype=np.float64,
    #     )

    #     # Set fields for finite difference method
    #     fields_diffs = (
    #         np.arange(-num_of_points, num_of_points + 1).astype(np.int64)
    #         * delta_h
    #     )[:, np.newaxis] + fields
    #     fields_diffs = fields_diffs.astype(np.float64)

    #     for diff_index, fields_diff in enumerate(fields_diffs):
    #         print(f"STENCIL DIFFERENCE: {diff_index}")
    #         start_time = time.perf_counter()
    #         # Get M(t,H) for adjacent values of field
    #         (
    #             mag_x[:, :, :, :, diff_index],
    #             mag_y[:, :, :, :, diff_index],
    #             mag_z[:, :, :, :, diff_index],
    #         ) = mag_3d(
    #             filename,
    #             group,
    #             states_cutoff,
    #             fields_diff,
    #             spherical_grid,
    #             temperatures,
    #             num_cpu,
    #             num_threads,
    #         )
    #         end_time = time.perf_counter()
    #         print(f"{end_time - start_time} s")
    #     stencil_coeff = finite_diff_stencil(1, num_of_points, delta_h)

    #     if T:
    #         for temp_index, temp in enumerate(temperatures):
    #             x[:, temp_index, :, :] = temp * np.dot(
    #                 mag_x[:, temp_index, :, :, :], stencil_coeff
    #             )
    #             y[:, temp_index, :, :] = temp * np.dot(
    #                 mag_y[:, temp_index, :, :, :], stencil_coeff
    #             )
    #             z[:, temp_index, :, :] = temp * np.dot(
    #                 mag_z[:, temp_index, :, :, :], stencil_coeff
    #             )
    #     else:
    #         x[:, :, :, :] = np.dot(mag_x[:, :, :, :, :], stencil_coeff)
    #         y[:, :, :, :] = np.dot(mag_y[:, :, :, :, :], stencil_coeff)
    #         z[:, :, :, :] = np.dot(mag_z[:, :, :, :, :], stencil_coeff)

    # return x, y, z
