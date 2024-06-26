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


from numpy import complex128

from slothpy._general_utilities._system import SharedMemoryArrayInfo, _load_shared_memory_arrays
from slothpy._general_utilities._math_expresions import _add_diagonal

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
        if self._number_of_centers > 1:
            sm, interaction_matrix = _load_shared_memory_arrays(sm_arrays_info_list[self._shared_memory_index - 1])
            self._sm.append(sm)
            setattr(self, "_interaction_matrix", interaction_matrix)
        self._magnetic_field = None
        if self.m[0].dtype == complex128: # Here I assume that m is always present, it may change in the future!
            from slothpy._general_utilities._lapack import _zheevr_lwork as _heevr_lwork, _zheevr as _heevr, _zutmud as _utmud, _zdot3d as _dot3d
        else:
            from slothpy._general_utilities._lapack import _cheevr_lwork as _heevr_lwork, _cheevr as _heevr, _cutmud as _utmud, _cdot3d as _dot3d
        self._heevr_lwork, self._heevr, self._utmud, self._dot3d = _heevr_lwork, _heevr, _utmud, _dot3d
        self._lwork = None

    def zeeman_hamiltonian(self, index):
        hamiltonian = self._dot3d(self.m[index], self._magnetic_field)
        _add_diagonal(hamiltonian, self.e[index])
        return hamiltonian
    
    def zeeman_energies(self):
        if self._number_of_centers > 1:
            for i in range(self._number_of_centers):
                pass
        else:
            if self._lwork is None:
                self._lwork = self._heevr_lwork(self.m[0].shape[1], jobz='N', range='I' if isinstance(self._cutoff_info_list[0][2], int) else 'V', il=1, iu=self._cutoff_info_list[0][2], vu=self._cutoff_info_list[0][2])
            return self._heevr(self.zeeman_hamiltonian(0).T, *self._lwork, jobz='N', range='I' if isinstance(self._cutoff_info_list[0][2], int) else 'V', il=1, iu=self._cutoff_info_list[0][2], vu=self._cutoff_info_list[0][2])



    





    




        
