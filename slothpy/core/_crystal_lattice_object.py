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


class CrystalLaticce():

    def __init__(self, sm_arrays_info_list: list[SharedMemoryArrayInfo], slt_hamiltonian_info):
        self._cutoff_info_list = slt_hamiltonian_info[1]
        self._local_states = slt_hamiltonian_info[2]
        self._number_of_centers = len(self._cutoff_info_list)
        self._sm = []
        self._shared_memory_index = (len(slt_hamiltonian_info[0]))*self._number_of_centers + (1 if self._number_of_centers > 1 else 0)
        for index, attr in enumerate(slt_hamiltonian_info[0]):
            sm, arrays = _load_shared_memory_arrays(sm_arrays_info_list[index*self._number_of_centers:(index+1)*self._number_of_centers])
            self._sm.append(sm)
            setattr(self, attr, arrays)
        self._cutoff = 1
        for cutoff in self._cutoff_info_list:
            self._cutoff *= cutoff[2]
        if self._number_of_centers > 1:
            sm, interaction_matrix = _load_shared_memory_arrays([sm_arrays_info_list[self._shared_memory_index - 1]])
            self._sm.append(sm)
            self._interaction_matrix = interaction_matrix[0]
        self._magnetic_field = None
        self._electric_field = None
        if self.m[0].dtype == complex128: # Here I assume that m is always present, it may change in the future!
            from slothpy._general_utilities._lapack import _zheevr_lwork as _heevr_lwork, _zheevr as _heevr, _zutmu as _utmu, _zutmud as _utmud, _zdot3d as _dot3d
        else:
            from slothpy._general_utilities._lapack import _cheevr_lwork as _heevr_lwork, _cheevr as _heevr, _cutmu as _utmu, _cutmud as _utmud, _cdot3d as _dot3d
        self._heevr_lwork, self._heevr, self._utmu, self._utmud, self._dot3d = _heevr_lwork, _heevr, _utmu, _utmud, _dot3d
        self._lwork = None