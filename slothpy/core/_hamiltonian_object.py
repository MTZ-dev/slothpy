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


from numpy import ndarray, complex128, int64, ascontiguousarray, tensordot, empty, zeros_like, zeros

from slothpy._general_utilities._system import SharedMemoryArrayInfo, _load_shared_memory_arrays
from slothpy._general_utilities._math_expresions import _add_diagonal
from slothpy._general_utilities._direct_product_space import _kron_mult
from slothpy._general_utilities._math_expresions import _subtract_min_from_arrays_list

class Hamiltonian():

    def __init__(self, sm_arrays_info_list: list[SharedMemoryArrayInfo], mode, cutoff_info_list):
        self._cutoff_info_list = cutoff_info_list
        self._number_of_centers = len(self._cutoff_info_list)
        self._sm = []
        self._shared_memory_index = (len(mode))*self._number_of_centers + (1 if self._number_of_centers > 1 else 0)
        for index, attr in enumerate(mode):
            sm, arrays = _load_shared_memory_arrays(sm_arrays_info_list[index*self._number_of_centers:(index+1)*self._number_of_centers])
            self._sm.append(sm)
            setattr(self, attr, arrays)
        self._cutoff = 1
        for cutoff in cutoff_info_list:
            self._cutoff *= cutoff[2]
        if self._number_of_centers > 1:
            sm, interaction_matrix = _load_shared_memory_arrays([sm_arrays_info_list[self._shared_memory_index - 1]])
            self._sm.append(sm)
            self._interaction_matrix = interaction_matrix[0]
        self._magnetic_field = None
        self._electric_field = None
        if self.m[0].dtype == complex128: # Here I assume that m is always present, it may change in the future!
            from slothpy._general_utilities._lapack import _zheevr_lwork as _heevr_lwork, _zheevr as _heevr, _zutmud as _utmud, _zdot3d as _dot3d
        else:
            from slothpy._general_utilities._lapack import _cheevr_lwork as _heevr_lwork, _cheevr as _heevr, _cutmud as _utmud, _cdot3d as _dot3d
        self._heevr_lwork, self._heevr, self._utmud, self._dot3d = _heevr_lwork, _heevr, _utmud, _dot3d
        self._lwork = None

    def build_hamiltonian(self, index: int, cutoff: int = None):
        hamiltonian = self._dot3d(self.m[index] if cutoff is None else ascontiguousarray(self.m[index][:, :cutoff,:cutoff]), -self._magnetic_field)
        if self._electric_field is not None:
            hamiltonian += self._dot3d(self.p[index] if cutoff is None else ascontiguousarray(self.p[index][:, :cutoff,:cutoff]), -self._electric_field)
        _add_diagonal(hamiltonian, self.e[index])
        return hamiltonian
    
    def zeeman_energies(self, number_of_states = None):
        iu = self._cutoff if number_of_states is None else max(self._cutoff, number_of_states)
        if self._number_of_centers > 1:
            hamiltonian = self._interaction_matrix.copy()
            for i in range(self._number_of_centers):
                ops = [self.build_hamiltonian(i, self._cutoff_info_list[i][1]) if k == i else self._cutoff_info_list[k][1] for k in range(self._number_of_centers)]
                hamiltonian += _kron_mult(ops)
            if self._lwork is None:
                self._lwork = self._heevr_lwork(hamiltonian.shape[1], jobz='N', range='I', il=1, iu=iu)
            return self._heevr(hamiltonian.T, *self._lwork, jobz='N', range='I', il=1, iu=iu)[:number_of_states]
        else:
            if self._lwork is None:
                self._lwork = self._heevr_lwork(self.m[0].shape[1], jobz='N', range='I' if isinstance(iu, (int, int64)) else 'V', il=1, iu=iu, vu=iu)
            return self._heevr(self.build_hamiltonian(0).T, *self._lwork, jobz='N', range='I' if isinstance(iu, (int, int64)) else 'V', il=1, iu=iu, vu=iu)[:number_of_states]

    def slpjm_under_fields(self, mode, orientation):
        if self._number_of_centers > 1:
            result_list = []
            energies_list = []
            hamiltonian = self._interaction_matrix.copy()
            result_matrix = zeros_like(hamiltonian) if isinstance(orientation, ndarray) else zeros((3, hamiltonian.shape[0], hamiltonian.shape[1]))
            for i in range(self._number_of_centers):
                zeeman_matrix = self.build_hamiltonian(i)
                ops = [ascontiguousarray(zeeman_matrix[:self._cutoff_info_list[i][1], :self._cutoff_info_list[i][1]]) if k == i else self._cutoff_info_list[k][1] for k in range(self._number_of_centers)]
                hamiltonian += _kron_mult(ops)
                start_state = self._cutoff_info_list[i][2] + 1
                _lwork = self._heevr_lwork(zeeman_matrix.shape[1], jobz='V', range='I', il=start_state, iu=zeeman_matrix.shape[1])
                energies, eigenvectors = self._heevr(zeeman_matrix.T, *_lwork, jobz='V', range='I', il=start_state, iu=zeeman_matrix.shape[1])
                energies_list.append(energies)
                if not isinstance(orientation, ndarray):
                    result_local = empty((3, eigenvectors.shape[1]), dtype=energies.dtype, order='C')
                    for k in range(3):
                        matrix_local = getattr(self, mode)[i][k]
                        result_local[k,:] = self._utmud(eigenvectors, matrix_local.T)
                        ops_result = [ascontiguousarray(matrix_local[:self._cutoff_info_list[i][1], :self._cutoff_info_list[i][1]]) if l == i else self._cutoff_info_list[l][1] for l in range(self._number_of_centers)]
                        result_matrix[k,:,:] += _kron_mult(ops_result)
                else: 
                    matrix_local = tensordot(getattr(self, mode)[i], orientation, axes=(0,0))
                    result_local = self._utmud(eigenvectors, matrix_local.T)
                    ops_result = [ascontiguousarray(matrix_local[:self._cutoff_info_list[i][1], :self._cutoff_info_list[i][1]]) if l == i else self._cutoff_info_list[l][1] for l in range(self._number_of_centers)]
                    result_matrix += _kron_mult(ops_result)
                result_list.append(result_local)
            if self._lwork is None:
                self._lwork = self._heevr_lwork(hamiltonian.shape[1], jobz='V', range='I', il=1, iu=self._cutoff)
            energies, eigenvectors =  self._heevr(hamiltonian.T, *self._lwork, jobz='V', range='I', il=1, iu=self._cutoff)
            energies_list.append(energies)
            if not isinstance(orientation, ndarray):
                result = empty((3, eigenvectors.shape[1]), dtype=energies.dtype, order='C')
                for i in range(3):
                   result[i,:] = self._utmud(eigenvectors, result_matrix[i].T) 
            else:
                result = self._utmud(eigenvectors, result_matrix.T)
            result_list.append(result)
            _subtract_min_from_arrays_list(energies_list)
            return energies_list, result_list
        else:
            if self._lwork is None:
                self._lwork = self._heevr_lwork(self.m[0].shape[1], jobz='V', range='I' if isinstance(self._cutoff, (int, int64)) else 'V', il=1, iu=self._cutoff, vu=self._cutoff)
            energies, eigenvectors =  self._heevr(self.build_hamiltonian(0).T, *self._lwork, jobz='V', range='I' if isinstance(self._cutoff, (int, int64)) else 'V', il=1, iu=self._cutoff, vu=self._cutoff)
            if len(energies) < 1:
                raise ValueError("Auto-cutoff failed! Please set it manually and run the calculation using the reviewed settings once again.")
            if not isinstance(orientation, ndarray):
                result = empty((3, eigenvectors.shape[1]), dtype=energies.dtype, order='C')
                for i in range(3):
                    result[i,:] = self._utmud(eigenvectors, getattr(self, mode)[0][i].T)
            else:
                result = self._utmud(eigenvectors, tensordot(getattr(self, mode)[0], orientation, axes=(0,0)).T)
        return energies - energies[0], result


    def slpjm_matrix_under_fields(self, mode, orientation):
        pass

    





    




        
