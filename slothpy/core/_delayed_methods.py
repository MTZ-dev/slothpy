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

from os.path import join

import matplotlib.pyplot as plt
from pandas import DataFrame

from slothpy.core._config import settings
from slothpy.core._slt_file import SltGroup
from slothpy.core._slothpy_exceptions import slothpy_exc
from slothpy.core._drivers import ensure_ready
from slothpy._general_utilities._constants import RED, GREEN, BLUE, PURPLE, YELLOW, RESET, H_CM_1

class SltStatesEnergiesCm1():
     
    def __init__(self, slt_group: SltGroup, start_state=0, stop_state=0, slt_save=None) -> None:
        self._slt_group = slt_group
        self._hdf5 = slt_group._hdf5
        self._group_path = slt_group._group_path
        self._ready = False
        self._start_state = start_state
        self._stop_state = stop_state
        self._slt_save = slt_save

        self._energies_cm_1 = None
        self._df = None
    
    @classmethod
    def _from_file(cls, hdf5, slt_group):
        instance = cls.__new__(cls)
        instance._slt_group = SltGroup(hdf5, slt_group)
        instance._hdf5 = hdf5
        instance._group_path = slt_group
        instance._ready = True
        instance._slt_save = None

        instance._energies_cm_1 = instance._slt_group["STATES_ENERGIES_CM_1"][:]
        instance._df = None

        return instance
        
    def __repr__(self) -> str:
        return f"<{RED}SltStatesEnergiesCm1{RESET} object from {BLUE}Group{RESET} '{self._group_path}' {GREEN}File{RESET} '{self._hdf5}'.>"
    
    def _prepare_memory(smm):
        pass
    
    @slothpy_exc("SltCompError")
    def run(self):
        if not self._ready:
            self._energies_cm_1 = self._slt_group.energies[self._start_state:self._stop_state] * H_CM_1
            self._ready = True
        if self._slt_save is not None:
            self.save()
        return self._energies_cm_1
    
    eval = run
    
    @ensure_ready
    def save(self, slt_save = None):
        new_group = SltGroup(self._hdf5, self._slt_save if slt_save is None else slt_save, exists=False)
        new_group["STATES_ENERGIES_CM_1"] = self._energies_cm_1
        new_group["STATES_ENERGIES_CM_1"].attributes["Description"] = "States' energies in cm-1."
        new_group.attributes["Type"] = "ENERGIES"
        new_group.attributes["Kind"] = "CM_1"
        new_group.attributes["States"] = self._energies_cm_1.shape[0]
        new_group.attributes["Precision"] = settings.precision.upper()
        new_group.attributes["Description"] = "States' energies in cm-1."

    @ensure_ready
    def load(self):
        return self._energies_cm_1

    @ensure_ready
    def plot(self):
        fig, ax = plt.subplots()
        x_min = 0
        x_max = 1
        for energy in self._energies_cm_1:
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

    @ensure_ready
    def to_numpy_array(self):
        return self._energies_cm_1
    
    @ensure_ready
    def to_data_frame(self):
        self._df = DataFrame({'Energy (cm^-1)': self._energies_cm_1})
        self._df.index.name = 'State Number'
        return self._df

    def to_csv(self, file_path=".", file_name="states_energies_cm_1.csv", separator=","):
        if self._df is None:
            self.to_data_frame()
        self._df.to_csv(join(file_path, file_name), sep=separator)

