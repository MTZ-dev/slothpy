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

from __future__ import annotations
from multiprocessing.managers import SharedMemoryManager
from multiprocessing.synchronize import Event
from numpy import ndarray, zeros
from pandas import DataFrame
import matplotlib.pyplot as plt

from slothpy.core._config import settings
from slothpy.core._drivers import _SingleProcessed, _MultiProcessed
from slothpy._general_utilities._constants import H_CM_1
from slothpy._magnetism._zeeman_lanczos import _zeeman_splitting_proxy
################# __slots__ in every slt class!!
class SltStatesEnergiesCm1(_SingleProcessed):

    __slots__ = _SingleProcessed.__slots__ + ["_start_state", "_stop_state"]
     
    def __init__(self, slt_group, start_state: int = 0, stop_state: int = 0, slt_save: str = None) -> None:
        super().__init__(slt_group, slt_save)
        self._method_name = "States Energies in cm-1"
        self._method_type = "STATES_ENERGIES"
        self._start_state = start_state
        self._stop_state = stop_state

    def _executor(self):
        return self._slt_group.energies[self._start_state:self._stop_state] * H_CM_1
    
    def _save(self):
        self._metadata_dict = {
            "Type": self._method_type,
            "Kind": "CM_1",
            "States": self._result.shape[0],
            "Precision": settings.precision.upper(),
            "Description": f"States' energies in cm-1 from Group '{self._group_name}'."
        }
        self._data_dict = {"STATES_ENERGIES_CM_1": (self._result,  "States' energies in cm-1.")}

    def _load_from_file(self):
        self._result = self._slt_group["STATES_ENERGIES_CM_1"][:]

    #TODO: plot
    def _plot(self):
        fig, ax = plt.subplots()
        x_min = 0
        x_max = 1
        for energy in self._result:
            ax.hlines(y=energy, xmin=x_min, xmax=x_max, colors='skyblue', linestyles='solid', linewidth=2)
            ax.text(x_max + 0.1, energy, f'{energy:.1f}', va='center', ha='left')
        ax.set_ylabel('Energy (cm$^{-1}$)')
        ax.set_title('Energy Levels')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.xaxis.set_ticks_position('none')
        ax.set_xticks([])
        plt.tight_layout()
        plt.show()

    def _to_data_frame(self):
        self._df = DataFrame({'Energy (cm^-1)': self._result})
        self._df.index.name = 'State Number'
        return self._df


class SltStatesEnergiesAu(_SingleProcessed):

    __slots__ = _SingleProcessed.__slots__ + ["_start_state", "_stop_state"]
     
    def __init__(self, slt_group, start_state: int = 0, stop_state: int = 0, slt_save: str = None) -> None:
        super().__init__(slt_group, slt_save)
        self._method_name = "States Energies in a.u."
        self._method_type = "STATES_ENERGIES"
        self._start_state = start_state
        self._stop_state = stop_state

    def _executor(self):
        return self._slt_group.energies[self._start_state:self._stop_state]

    def _save(self):
        self._metadata_dict = {
            "Type": self._method_type,
            "Kind": "AU",
            "States": self._result.shape[0],
            "Precision": settings.precision.upper(),
            "Description": f"States' energies in a.u. from Group '{self._group_name}'."
        }
        self._data_dict = {"STATES_ENERGIES_AU": (self._result, "States' energies in cm-1.")}

    def _load_from_file(self):
        self._result = self._slt_group["STATES_ENERGIES_AU"][:]

    #TODO: plot
    def _plot(self):
        fig, ax = plt.subplots()
        x_min = 0
        x_max = 1
        for energy in self._result:
            ax.hlines(y=energy, xmin=x_min, xmax=x_max, colors='skyblue', linestyles='solid', linewidth=2)
            ax.text(x_max + 0.1, energy, f'{energy:.1f}', va='center', ha='left')
        ax.set_ylabel('Energy (cm$^{-1}$)')
        ax.set_title('Energy Levels')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.xaxis.set_ticks_position('none')
        ax.set_xticks([])
        plt.tight_layout()
        plt.show()

    def _to_data_frame(self):
        self._df = DataFrame({'Energy (a.u.)': self._result})
        self._df.index.name = 'State Number'
        return self._df


class SltSpinMatrices(_SingleProcessed):

    __slots__ = _SingleProcessed.__slots__ + ["_xyz", "_start_state", "_stop_state", "_rotation"]
     
    def __init__(self, slt_group, xyz='xyz', start_state=0, stop_state=0, rotation=None, slt_save=None) -> None:
        super().__init__(slt_group, slt_save)
        self._method_name = "Spin Matrices"
        self._method_type = "SPINS"
        self._xyz = xyz
        self._start_state = start_state
        self._stop_state = stop_state
        self._rotation = rotation
    ######### tutaj skonczyles
    def _executor(self):
        return self._slt_group.energies[self._start_state:self._stop_state]

    def _save(self):
        self._metadata_dict = {
            "Type": self._method_type,
            "Kind": "AU",
            "States": self._result.shape[0],
            "Precision": settings.precision.upper(),
            "Description": f"States' energies in a.u. from Group '{self._group_name}'."
        }
        self._data_dict = {"STATES_ENERGIES_AU": (self._result, "States' energies in cm-1.")}

    def _load_from_file(self):
        self._result = self._slt_group["STATES_ENERGIES_AU"][:]

    #TODO: plot
    def _plot(self):
        pass

    #TODO: df
    def _to_data_frame(self):
        pass
    

class SltZeemanSplitting(_MultiProcessed):

    __slots__ = _MultiProcessed.__slots__ + ["_magnetic_fields", "_orientations", "_states_cutoff"]
     
    def __init__(self, slt_group,
        magnetic_fields: ndarray,
        orientations: ndarray,
        states_cutoff: int,
        number_of_states: int,
        number_cpu: int,
        number_threads: int,
        autotune: bool,
        slt_save: str = None,
        smm: SharedMemoryManager = None,
        terminate_event: Event = None,
        ) -> None:
        super().__init__(slt_group, magnetic_fields.shape[0] * orientations.shape[0] , number_cpu, number_threads, autotune, smm, terminate_event, slt_save)
        self._method_name = "Zeeman Splitting"
        self._method_type = "ZEEMAN_SPLITTING"
        self._magnetic_fields = magnetic_fields
        self._orientations = orientations
        self._states_cutoff = states_cutoff
        self._args = (number_of_states,)
        self._executor_proxy = _zeeman_splitting_proxy
    
    def _load_args_arrays(self):
        self._args_arrays = (self._slt_group.energies[:self._states_cutoff], self._slt_group.magnetic_dipole_momenta[:, :self._states_cutoff, :self._states_cutoff], self._magnetic_fields, self._orientations)
        if self._orientations.shape[1] == 4:
            self._returns = True
        elif self._orientations.shape[1] == 3:
            self._result = zeros((self._magnetic_fields.shape[0] * self._orientations.shape[0], self._args[0]), dtype=self._magnetic_fields.dtype, order="C")
            self._result_shape = (self._orientations.shape[0], self._magnetic_fields.shape[0], self._args[0])
    
    def _gather_results(self, result_queue):
        zeeman_splitting_array = zeros((self._magnetic_fields.shape[0], self._args[0]), dtype=settings.float)
        while not result_queue.empty():
            start_field_index, end_field_index, zeeman_array = result_queue.get()
            for i, j in enumerate(range(start_field_index, end_field_index)):
                zeeman_splitting_array[j, :] += zeeman_array[i, :]
        return zeeman_splitting_array

    def _save(self):
        self._metadata_dict = {
            "Type": self._method_type,
            "Kind": "AVERAGE" if self._result.ndim == 2 else "DIRECTIONAL",
            "Precision": settings.precision.upper(),
            "Description": f"Group containing Zeeman splitting calculated from Group '{self._group_name}'."
        }
        self._data_dict = {
            "ZEEMAN_SPLITTING": (self._result, "Dataset containing Zeeman splitting in the form {}".format("[fields, energies]" if self._result.ndim == 2 else "[orientations, fields, energies]")),
            "MAGNETIC_FIELDS": (self._magnetic_fields, "Dataset containing magnetic field (T) values used in the simulation."),
            "ORIENTATIONS": (self._orientations, "Dataset containing magnetic fields' orientation grid used in the simulation."),
        }

    def _load_from_file(self):
        self._result = self._slt_group["ZEEMAN_SPLITTING"][:]
        self._magnetic_fields = self._slt_group["MAGNETIC_FIELDS"][:]
        self._orientations = self._slt_group["ORIENTATIONS"][:]

    def _plot(self):
        pass
 
    def _to_data_frame(self):
        pass