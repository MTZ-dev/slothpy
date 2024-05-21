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

from time import perf_counter
# from openfermion.linalg.davidson import Davidson
import numpy as np
from scipy.sparse.linalg import eigsh, LinearOperator
from threadpoolctl import threadpool_limits
from numpy import (
    ndarray,
    array,
    zeros,
    ascontiguousarray,
    float64,
    complex128,
    diagonal,

)
from numba import jit, set_num_threads, prange, types, float32, float64, complex64, complex128
from slothpy._general_utilities._constants import H_CM_1
from slothpy._general_utilities._system import (
    SharedMemoryArrayInfo,
    _load_shared_memory_arrays,
)
from slothpy._general_utilities._io import (
    _get_soc_magnetic_momenta_and_energies_from_hdf5,
)
from slothpy._general_utilities._math_expresions import _3d_dot
# import primme, scipy.sparse

@jit([
    types.Array(complex64, 2, 'C')(
        types.Array(complex64, 3, 'C', True), 
        types.Array(float32, 1, 'C', True), 
        float32, 
        types.Array(float32, 1, 'C', True)
    ),
    types.Array(complex128, 2, 'C')(
        types.Array(complex128, 3, 'C', True), 
        types.Array(float64, 1, 'C', True), 
        float64, 
        types.Array(float64, 1, 'C', True)
    )
],
nopython=True,
nogil=True,
cache=True,
fastmath=True,
parallel=True,
)
def _calculate_zeeman_matrix(
    magnetic_momenta, states_energies, field, orientation
):
    magnetic_momenta = _3d_dot(magnetic_momenta, -field * orientation)

    for k in prange(magnetic_momenta.shape[0]):
        magnetic_momenta[k, k] += states_energies[k]

    return magnetic_momenta


class KroneckerLinearOperator(LinearOperator):
    def __init__(self, magnetic_momenta, states_energies, magnetic_fields, orientations):
        self.magnetic_momenta = ascontiguousarray(magnetic_momenta)
        self.states_energies = states_energies
        self.magnetic_fields = magnetic_fields
        self.orientations = orientations
        self.dtype = self.magnetic_momenta.dtype
        self.shape = (self.magnetic_momenta.shape[1], self.magnetic_momenta.shape[2])

    def update_matrix(self, i):
        self._mat1 = _calculate_zeeman_matrix(
            self.magnetic_momenta,
            self.states_energies,
            self.magnetic_fields[i % self.magnetic_fields.shape[0]],
            self.orientations[i // self.magnetic_fields.shape[0], :3]
        )
    
    def _matvec(self, v):
        return kronecker_matvec_jit(v, self._mat1)
    
    def _mat(self):
        return self._mat1

    
def _zeeman_splitting(
    states_energies: ndarray,
    magnetic_momenta: ndarray,
    magnetic_fields: ndarray,
    orientations: ndarray,
    progress_array: ndarray,
    zeeman_array: ndarray,
    number_of_states: int,
    process_index: int,
    start: int,
    end: int,
):
    op = KroneckerLinearOperator(magnetic_momenta, states_energies, magnetic_fields, orientations)
    import primme

    for i in range(start, end):
        op.update_matrix(i)
        eigval, evecs = primme.eigsh(op, number_of_states, tol=1e-6, which='SA')
        # eigval, evecs = primme.eigsh(_calculate_zeeman_matrix(magnetic_momenta, states_energies, magnetic_fields[i%magnetic_fields.shape[0]], orientations[i//magnetic_fields.shape[0], :3]), number_of_states, tol=1e-6, which='SA')
        # success, eigval, eigen_vectors = Davidson(op, diagonal(op._mat1)).get_lowest_n(number_of_states)
        # eigval, _ = eigsh(op, k=number_of_states, which="SA")
        zeeman_array[i] = eigval * array(H_CM_1, dtype=states_energies.dtype) ######## moze z primme sa juz posortowane eigvals?
        progress_array[process_index] += 1

# @jit(["complex128[:,:](complex128[:,:], complex128[:,:])", "complex128[:,:](complex128[:], complex128[:,:])"], nopython=True, fastmath=True, cache=True)
###jit wolny tutaj
def kronecker_matvec_jit(v, mat1):
    # v = ascontiguousarray(v)
    # mat1 = ascontiguousarray(mat1)
    return (mat1@v).reshape(-1,1)

def _zeeman_splitting_average(
    states_energies: ndarray,
    magnetic_momenta: ndarray,
    magnetic_fields: ndarray,
    orientations: ndarray,
    progress_array: ndarray,
    number_of_states: int,
    process_index: int,
    start: int,
    end: int,
):
    pass


def _zeeman_splitting_proxy(sm_arrays_info_list: list[SharedMemoryArrayInfo], args_list, process_index, start: int, end: int, number_threads: int, returns: bool = False):
    sm, arrays = _load_shared_memory_arrays(sm_arrays_info_list)
    with threadpool_limits(limits=number_threads):
        set_num_threads(number_threads)
        if returns:
            return _zeeman_splitting_average(*arrays, *args_list, process_index, start, end)
        else:
            _zeeman_splitting(*arrays, *args_list, process_index, start, end)