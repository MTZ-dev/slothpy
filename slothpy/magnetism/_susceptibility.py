from numpy import (
    ndarray,
    array,
    arange,
    dot,
    linspace,
    meshgrid,
    zeros,
    newaxis,
    float64,
    int64,
    sin,
    cos,
    pi,
)
from slothpy.magnetism._magnetisation import _mth, _mag_3d
from slothpy.general_utilities._math_expresions import _finite_diff_stencil
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
    if exp or (num_of_points == 0):
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

        chitht_array = mth_array.T / fields[:, newaxis]

        if T:
            chitht_array = chitht_array * temperatures[newaxis, :]

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
        # Numerical derivative of M(T,H) around given field value
        chitht_array = dot(mth_array, stencil_coeff).T

        if T:
            chitht_array = chitht_array * temperatures[newaxis, :]

    chitht_array = chitht_array * MU_B_CM_3

    return chitht_array


def _chitht_tensor(
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
    # When passed to _mth activates tensor calculation and _mth actually
    # returns mht format
    grid = array([1])

    # Experimentalist model
    if exp or (num_of_points == 0):
        mht_tensor_array = _mth(
            filename,
            group,
            fields,
            grid,
            temperatures,
            states_cutoff,
            num_cpu,
            num_threads,
            rotation,
        )

        chitht_tensor_array = (
            mht_tensor_array / fields[:, newaxis, newaxis, newaxis]
        )

        if T:
            chitht_tensor_array = (
                chitht_tensor_array
                * temperatures[newaxis, :, newaxis, newaxis]
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

        mht_tensor_array = mht_tensor_array.transpose((0, 2, 3, 4, 1))
        # Numerical derivative of M(T,H) around given field value
        chitht_tensor_array = dot(mht_tensor_array, stencil_coeff)

        if T:
            chitht_tensor_array = (
                chitht_tensor_array
                * temperatures[newaxis, :, newaxis, newaxis]
            )

    chitht_tensor_array = chitht_tensor_array * MU_B_CM_3

    return chitht_tensor_array


def _chit_3d(
    filename: str,
    group: str,
    temperatures: ndarray,
    fields: ndarray,
    spherical_grid: int,
    num_of_points: int,
    delta_h: float64,
    states_cutoff: int,
    num_cpu: int,
    num_threads: int,
    exp: bool = False,
    T: bool = True,
) -> ndarray[float64]:
    # Experimentalist model
    if exp or (num_of_points == 0):
        mag_3d_array = _mag_3d(
            filename,
            group,
            fields,
            spherical_grid,
            temperatures,
            states_cutoff,
            num_cpu,
            num_threads,
        )

        chi_3d_array = (
            mag_3d_array / fields[newaxis, :, newaxis, newaxis, newaxis]
        )

        if T:
            chi_3d_array = (
                chi_3d_array
                * temperatures[newaxis, newaxis, :, newaxis, newaxis]
            )

    else:
        fields_diffs = (
            arange(-num_of_points, num_of_points + 1).astype(int64) * delta_h
        )[:, newaxis] + fields
        fields_diffs = fields_diffs.T.astype(float64)
        fields_diffs = fields_diffs.flatten()

        # Get M(T,H) for adjacent values of field
        mag_3d = _mag_3d(
            filename,
            group,
            fields_diffs,
            spherical_grid,
            temperatures,
            states_cutoff,
            num_cpu,
            num_threads,
            sus_3d_num=True,
        )

        stencil_coeff = _finite_diff_stencil(1, num_of_points, delta_h)

        mag_3d_array = mag_3d.reshape(
            (
                fields.size,
                stencil_coeff.size,
                temperatures.size,
                spherical_grid,
                2 * spherical_grid,
            )
        )
        mag_3d_array = mag_3d_array.transpose((0, 2, 3, 4, 1))
        # Numerical derivative of M(T,H) around given field value
        chi_3d = dot(mag_3d_array, stencil_coeff)

        if T:
            chi_3d = chi_3d * temperatures[newaxis, :, newaxis, newaxis]

    theta = linspace(0, 2 * pi, 2 * spherical_grid, dtype=float64)
    phi = linspace(0, pi, spherical_grid, dtype=float64)
    theta, phi = meshgrid(theta, phi)

    chi_3d = chi_3d * MU_B_CM_3

    chi_3d_array = zeros((3, *chi_3d.shape), dtype=float64)

    chi_3d_array[0] = (sin(phi) * cos(theta))[newaxis, newaxis, :, :] * chi_3d
    chi_3d_array[1] = (sin(phi) * sin(theta))[newaxis, newaxis, :, :] * chi_3d
    chi_3d_array[2] = (cos(phi))[newaxis, newaxis, :, :] * chi_3d

    return chi_3d_array
