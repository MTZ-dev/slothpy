# SlothPy
# Copyright (C) 2023 Mikolaj Tadeusz Zychowicz (MTZ)

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
from typing import Literal
from numpy import (
    ndarray,
    array,
    arange,
    dot,
    linspace,
    meshgrid,
    zeros,
    newaxis,
    float32,
    int64,
    sin,
    cos,
    pi,
)
from slothpy._magnetism._magnetisation import _mth, _mag_3d
from slothpy._general_utilities._math_expresions import _finite_diff_stencil
from slothpy._general_utilities._constants import MU_B_CM_3


def _chitht(
    filename: str,
    group: str,
    temperatures: ndarray[float32],
    fields: ndarray[float32],
    num_of_points: int,
    delta_h: float32,
    states_cutoff: int,
    num_cpu: int,
    num_threads: int,
    exp: bool = False,
    T: bool = True,
    grid: ndarray[float32] = None,
) -> ndarray[float32]:
    # Default XYZ grid
    if grid is None or grid == None:
        grid = array(
            [
                [1.0, 0.0, 0.0, 0.3333333333333333],
                [0.0, 1.0, 0.0, 0.3333333333333333],
                [0.0, 0.0, 1.0, 0.3333333333333333],
            ],
            dtype=float32,
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
        fields_diffs = fields_diffs.T.astype(float32)
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
    delta_h: float32,
    states_cutoff: int,
    num_cpu: int,
    num_threads: int,
    exp: bool = False,
    T: bool = True,
    rotation: ndarray[float32] = None,
) -> ndarray[float32]:
    # When passed to _mth activates tensor calculation and _mth actually
    # returns mht[3,3]-tensor format
    grid = array([1.0])

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
        fields_diffs = fields_diffs.T.astype(float32)
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
            rotation,
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
    grid_type: Literal["mesh", "fibonacci"],
    grid_number: int,
    num_of_points: int,
    delta_h: float32,
    states_cutoff: int,
    num_cpu: int,
    num_threads: int,
    exp: bool = False,
    T: bool = True,
    rotation: ndarray = None,
) -> ndarray[float32]:
    if grid_type != "mesh" and grid_type != "fibonacci":
        raise ValueError(
            'The only allowed grid types are "mesh" or "fibonacci".'
        ) from None
    # Experimentalist model
    if exp or (num_of_points == 0):
        mag_3d_array = _mag_3d(
            filename,
            group,
            fields,
            grid_type,
            grid_number,
            temperatures,
            states_cutoff,
            num_cpu,
            num_threads,
            rotation,
        )
        if grid_type == "mesh":
            chi_3d_array = (
                mag_3d_array / fields[newaxis, :, newaxis, newaxis, newaxis]
            )
            if T:
                chi_3d_array = (
                    chi_3d_array
                    * temperatures[newaxis, newaxis, :, newaxis, newaxis]
                )
        if grid_type == "fibonacci":
            chi_3d_array = mag_3d_array / fields[:, newaxis, newaxis, newaxis]
            if T:
                chi_3d_array = (
                    chi_3d_array * temperatures[newaxis, :, newaxis, newaxis]
                )

    else:
        fields_diffs = (
            arange(-num_of_points, num_of_points + 1).astype(int64) * delta_h
        )[:, newaxis] + fields
        fields_diffs = fields_diffs.T.astype(float32)
        fields_diffs = fields_diffs.flatten()

        # Get M(T,H) for adjacent values of field
        mag_3d, grid = _mag_3d(
            filename,
            group,
            fields_diffs,
            grid_type,
            grid_number,
            temperatures,
            states_cutoff,
            num_cpu,
            num_threads,
            rotation,
            True,
        )

        stencil_coeff = _finite_diff_stencil(1, num_of_points, delta_h)

        if grid_type == "mesh":
            mag_3d_array = mag_3d.reshape(
                (
                    fields.size,
                    stencil_coeff.size,
                    temperatures.size,
                    grid_number,
                    grid_number * 2,
                )
            )
            mag_3d_array = mag_3d_array.transpose((0, 2, 3, 4, 1))
            # Numerical derivative of M(T,H) around given field value
            chi_3d = dot(mag_3d_array, stencil_coeff)
            if T:
                chi_3d = chi_3d * temperatures[newaxis, :, newaxis, newaxis]
            chi_3d_array = grid[:, newaxis, newaxis, :, :] * chi_3d
        elif grid_type == "fibonacci":
            mag_3d_array = mag_3d.reshape(
                (
                    grid_number,
                    fields.size,
                    stencil_coeff.size,
                    temperatures.size,
                )
            )
            mag_3d_array = mag_3d_array.transpose((0, 1, 3, 2))
            chi_3d = dot(mag_3d_array, stencil_coeff)
            if T:
                chi_3d = chi_3d * temperatures[newaxis, newaxis, :]
            chi_3d_array = (
                grid[:, :, newaxis, newaxis] * chi_3d[:, newaxis, :, :]
            )
            chi_3d_array = chi_3d_array.transpose((2, 3, 0, 1))

    return chi_3d_array
