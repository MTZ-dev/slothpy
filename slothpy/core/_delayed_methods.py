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

import matplotlib.pyplot as plt
from pandas import DataFrame

from slothpy.core._config import settings
from slothpy.core._slt_file import SltGroup
from slothpy.core._drivers import SingleProcessed, MulitProcessed, ensure_ready
from slothpy._general_utilities._constants import RED, GREEN, BLUE, PURPLE, YELLOW, RESET, H_CM_1

class SltStatesEnergiesCm1(SingleProcessed):

    __slots__ = SingleProcessed.__slots__ + ["_start_state", "_stop_state"]
     
    def __init__(self, slt_group: SltGroup, start_state: int = 0, stop_state: int = 0, slt_save: str = None) -> None:
        super().__init__(slt_group, slt_save)
        self._start_state = start_state
        self._stop_state = stop_state

    def __repr__(self) -> str:
        return f"<{RED}SltStatesEnergiesCm1{RESET} object from {BLUE}Group{RESET} '{self._group_name}' {GREEN}File{RESET} '{self._hdf5}'.>"

    def _executor(self):
        return self._slt_group.energies[self._start_state:self._stop_state] * H_CM_1
    
    @ensure_ready
    def save(self, slt_save = None):
        new_group = SltGroup(self._hdf5, self._slt_save if slt_save is None else slt_save)
        new_group["STATES_ENERGIES_CM_1"] = self._result
        new_group["STATES_ENERGIES_CM_1"].attributes["Description"] = "States' energies in cm-1."
        new_group.attributes["Type"] = "ENERGIES"
        new_group.attributes["Kind"] = "CM_1"
        new_group.attributes["States"] = self._result.shape[0]
        new_group.attributes["Precision"] = settings.precision.upper()
        new_group.attributes["Description"] = f"States' energies in cm-1 from Group '{self._group_name}'."

    def _load(self):
        self._result = self._slt_group["STATES_ENERGIES_CM_1"][:]

    @ensure_ready
    def plot(self):
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
    
    @ensure_ready
    def to_data_frame(self):
        self._df = DataFrame({'Energy (cm^-1)': self._result})
        self._df.index.name = 'State Number'
        return self._df

