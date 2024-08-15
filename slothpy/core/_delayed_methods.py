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
from typing import Literal, Union
from multiprocessing.managers import SharedMemoryManager
from multiprocessing.synchronize import Event

from numpy import ndarray, zeros
from pandas import DataFrame
import matplotlib.pyplot as plt

from slothpy.core._config import settings
from slothpy.core._drivers import _SingleProcessed, _MultiProcessed
from slothpy._general_utilities._constants import H_CM_1
from slothpy._general_utilities._utils import slpjm_components_driver
from slothpy._magnetism._zeeman import _zeeman_splitting_proxy
from slothpy._magnetism._magnetisation import _magnetisation_proxy

class SltStatesEnergiesCm1(_SingleProcessed):

    __slots__ = _SingleProcessed.__slots__ + ["_start_state", "_stop_state"]
     
    def __init__(self, slt_group, start_state: int = 0, stop_state: int = 0, slt_save: str = None) -> None:
        super().__init__(slt_group, slt_save)
        self._method_name = "States' Energies in cm-1"
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
        self._method_name = "States' Energies in a.u."
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
        self._method_name = "Spin matrices"
        self._method_type = "SPINS"
        self._xyz = xyz
        self._start_state = start_state
        self._stop_state = stop_state
        self._rotation = rotation

    def _executor(self):
        return slpjm_components_driver(self._slt_group, "full", "s", self._xyz, self._start_state, self._stop_state, self._rotation)

    def _save(self):
        self._metadata_dict = {
            "Type": self._method_type,
            "Kind": f"{self._xyz.upper() if isinstance(self._xyz, str) else 'ORIENTATIONAL'}",
            "States": self._result.shape[1],
            "Precision": settings.precision.upper(),
            "Description": f"Spin matrices from Group '{self._group_name}'."
        }
        self._data_dict = {"SPIN_MATRICES": (self._result, f"{str(self._xyz).upper()}{' [(x-0, y-1, z-2), :, :]' if isinstance(self._xyz, str) and self._xyz == 'xyz' else ''} component{'s' if isinstance(self._xyz, str) and self._xyz == 'xyz' else ''} of the spin.")}
        if self._rotation is not None:
            self._data_dict["ROTATION"] = (self._rotation, "Rotation used to rotate the spin components.")
    
    def _load_from_file(self):
        self._result = self._slt_group["SPIN_MATRICES"][:]

    #TODO: plot
    def _plot(self):
        pass

    #TODO: df
    def _to_data_frame(self):
        pass


class SltStatesSpins(_SingleProcessed):

    __slots__ = _SingleProcessed.__slots__ + ["_xyz", "_start_state", "_stop_state", "_rotation"]
     
    def __init__(self, slt_group, xyz='xyz', start_state=0, stop_state=0, rotation=None, slt_save=None) -> None:
        super().__init__(slt_group, slt_save)
        self._method_name = "States' spins"
        self._method_type = "STATES_SPINS"
        self._xyz = xyz
        self._start_state = start_state
        self._stop_state = stop_state
        self._rotation = rotation

    def _executor(self):
        return slpjm_components_driver(self._slt_group, "diagonal", "s", self._xyz, self._start_state, self._stop_state, self._rotation)

    def _save(self):
        self._metadata_dict = {
            "Type": self._method_type,
            "Kind": f"{self._xyz.upper() if isinstance(self._xyz, str) else 'ORIENTATIONAL'}",
            "States": self._result.shape[1] if self._xyz == "xyz" or isinstance(self._xyz, ndarray) else self._result.shape[0],
            "Precision": settings.precision.upper(),
            "Description": f"States' expectation values of the spin from Group '{self._group_name}'."
        }
        self._data_dict = {"STATES_SPINS": (self._result, f"{str(self._xyz).upper()}{' [(x-0, y-1, z-2), :]' if isinstance(self._xyz, str) and self._xyz == 'xyz' else ''} component{'s' if isinstance(self._xyz, str) and self._xyz == 'xyz' else ''} of the states's spins.")}
        if self._rotation is not None:
            self._data_dict["ROTATION"] = (self._rotation, "Rotation used to rotate the spin components.")
    
    def _load_from_file(self):
        self._result = self._slt_group["STATES_SPINS"][:]

    #TODO: plot
    def _plot(self):
        pass

    #TODO: df
    def _to_data_frame(self):
        pass


class SltAngularMomentumMatrices(_SingleProcessed):

    __slots__ = _SingleProcessed.__slots__ + ["_xyz", "_start_state", "_stop_state", "_rotation"]
     
    def __init__(self, slt_group, xyz='xyz', start_state=0, stop_state=0, rotation=None, slt_save=None) -> None:
        super().__init__(slt_group, slt_save)
        self._method_name = "Angular momentum matrices"
        self._method_type = "ANGULAR_MOMENTA"
        self._xyz = xyz
        self._start_state = start_state
        self._stop_state = stop_state
        self._rotation = rotation

    def _executor(self):
        return slpjm_components_driver(self._slt_group, "full", "l", self._xyz, self._start_state, self._stop_state, self._rotation)

    def _save(self):
        self._metadata_dict = {
            "Type": self._method_type,
            "Kind": f"{self._xyz.upper() if isinstance(self._xyz, str) else 'ORIENTATIONAL'}",
            "States": self._result.shape[1],
            "Precision": settings.precision.upper(),
            "Description": f"Angular momentum matrices from Group '{self._group_name}'."
        }
        self._data_dict = {"ANGULAR_MOMENTUM_MATRICES": (self._result, f"{str(self._xyz).upper()}{' [(x-0, y-1, z-2), :, :]' if isinstance(self._xyz, str) and self._xyz == 'xyz' else ''} component{'s' if isinstance(self._xyz, str) and self._xyz == 'xyz' else ''} of the angular momentum.")}
        if self._rotation is not None:
            self._data_dict["ROTATION"] = (self._rotation, "Rotation used to rotate the angular momentum components.")
    
    def _load_from_file(self):
        self._result = self._slt_group["ANGULAR_MOMENTUM_MATRICES"][:]

    #TODO: plot
    def _plot(self):
        pass

    #TODO: df
    def _to_data_frame(self):
        pass


class SltStatesAngularMomenta(_SingleProcessed):

    __slots__ = _SingleProcessed.__slots__ + ["_xyz", "_start_state", "_stop_state", "_rotation"]
     
    def __init__(self, slt_group, xyz='xyz', start_state=0, stop_state=0, rotation=None, slt_save=None) -> None:
        super().__init__(slt_group, slt_save)
        self._method_name = "States' angular momenta"
        self._method_type = "STATES_ANGULAR_MOMENTA"
        self._xyz = xyz
        self._start_state = start_state
        self._stop_state = stop_state
        self._rotation = rotation

    def _executor(self):
        return slpjm_components_driver(self._slt_group, "diagonal", "l", self._xyz, self._start_state, self._stop_state, self._rotation)

    def _save(self):
        self._metadata_dict = {
            "Type": self._method_type,
            "Kind": f"{self._xyz.upper() if isinstance(self._xyz, str) else 'ORIENTATIONAL'}",
            "States": self._result.shape[1] if self._xyz == "xyz" or isinstance(self._xyz, ndarray) else self._result.shape[0],
            "Precision": settings.precision.upper(),
            "Description": f"States' expectation values of the angular momentum from Group '{self._group_name}'."
        }
        self._data_dict = {"STATES_ANGULAR_MOMENTA": (self._result, f"{str(self._xyz).upper()}{' [(x-0, y-1, z-2), :]' if isinstance(self._xyz, str) and self._xyz == 'xyz' else ''} component{'s' if isinstance(self._xyz, str) and self._xyz == 'xyz' else ''} of the states's angular momenta.")}
        if self._rotation is not None:
            self._data_dict["ROTATION"] = (self._rotation, "Rotation used to rotate the angular momentum components.")
    
    def _load_from_file(self):
        self._result = self._slt_group["STATES_ANGULAR_MOMENTA"][:]

    #TODO: plot
    def _plot(self):
        pass

    #TODO: df
    def _to_data_frame(self):
        pass


class SltElectricDipoleMomentumMatrices(_SingleProcessed):

    __slots__ = _SingleProcessed.__slots__ + ["_xyz", "_start_state", "_stop_state", "_rotation"]
     
    def __init__(self, slt_group, xyz='xyz', start_state=0, stop_state=0, rotation=None, slt_save=None) -> None:
        super().__init__(slt_group, slt_save)
        self._method_name = "Electric dipole momentum matrices"
        self._method_type = "ELECTRIC_DIPOLE_MOMENTA"
        self._xyz = xyz
        self._start_state = start_state
        self._stop_state = stop_state
        self._rotation = rotation

    def _executor(self):
        return slpjm_components_driver(self._slt_group, "full", "p", self._xyz, self._start_state, self._stop_state, self._rotation)

    def _save(self):
        self._metadata_dict = {
            "Type": self._method_type,
            "Kind": f"{self._xyz.upper() if isinstance(self._xyz, str) else 'ORIENTATIONAL'}",
            "States": self._result.shape[1],
            "Precision": settings.precision.upper(),
            "Description": f"Electric dipole momentum matrices from Group '{self._group_name}'."
        }
        self._data_dict = {"ELECTRIC_DIPOLE_MOMENTUM_MATRICES": (self._result, f"{str(self._xyz).upper()}{' [(x-0, y-1, z-2), :, :]' if isinstance(self._xyz, str) and self._xyz == 'xyz' else ''} component{'s' if isinstance(self._xyz, str) and self._xyz == 'xyz' else ''} of the electric dipole momentum.")}
        if self._rotation is not None:
            self._data_dict["ROTATION"] = (self._rotation, "Rotation used to rotate the electric dipole momentum components.")
    
    def _load_from_file(self):
        self._result = self._slt_group["ELECTRIC_DIPOLE_MOMENTUM_MATRICES"][:]

    #TODO: plot
    def _plot(self):
        pass

    #TODO: df
    def _to_data_frame(self):
        pass


class SltStatesElectricDipoleMomenta(_SingleProcessed):

    __slots__ = _SingleProcessed.__slots__ + ["_xyz", "_start_state", "_stop_state", "_rotation"]
     
    def __init__(self, slt_group, xyz='xyz', start_state=0, stop_state=0, rotation=None, slt_save=None) -> None:
        super().__init__(slt_group, slt_save)
        self._method_name = "States' electric dipole momenta"
        self._method_type = "STATES_ELECTRIC_DIPOLE_MOMENTA"
        self._xyz = xyz
        self._start_state = start_state
        self._stop_state = stop_state
        self._rotation = rotation

    def _executor(self):
        return slpjm_components_driver(self._slt_group, "diagonal", "p", self._xyz, self._start_state, self._stop_state, self._rotation)

    def _save(self):
        self._metadata_dict = {
            "Type": self._method_type,
            "Kind": f"{self._xyz.upper() if isinstance(self._xyz, str) else 'ORIENTATIONAL'}",
            "States": self._result.shape[1] if self._xyz == "xyz" or isinstance(self._xyz, ndarray) else self._result.shape[0],
            "Precision": settings.precision.upper(),
            "Description": f"States' expectation values of the electric dipole momentum from Group '{self._group_name}'."
        }
        self._data_dict = {"STATES_ELECTRIC_DIPOLE_MOMENTA": (self._result, f"{str(self._xyz).upper()}{' [(x-0, y-1, z-2), :]' if isinstance(self._xyz, str) and self._xyz == 'xyz' else ''} component{'s' if isinstance(self._xyz, str) and self._xyz == 'xyz' else ''} of the states's electric dipole momenta.")}
        if self._rotation is not None:
            self._data_dict["ROTATION"] = (self._rotation, "Rotation used to rotate the electric dipole momentum components.")
    
    def _load_from_file(self):
        self._result = self._slt_group["STATES_ELECTRIC_DIPOLE_MOMENTA"][:]

    #TODO: plot
    def _plot(self):
        pass

    #TODO: df
    def _to_data_frame(self):
        pass


class SltTotalAngularMomentumMatrices(_SingleProcessed):

    __slots__ = _SingleProcessed.__slots__ + ["_xyz", "_start_state", "_stop_state", "_rotation"]
     
    def __init__(self, slt_group, xyz='xyz', start_state=0, stop_state=0, rotation=None, slt_save=None) -> None:
        super().__init__(slt_group, slt_save)
        self._method_name = "Total angular momentum matrices"
        self._method_type = "TOTAL_ANGULAR_MOMENTA"
        self._xyz = xyz
        self._start_state = start_state
        self._stop_state = stop_state
        self._rotation = rotation

    def _executor(self):
        return slpjm_components_driver(self._slt_group, "full", "j", self._xyz, self._start_state, self._stop_state, self._rotation)

    def _save(self):
        self._metadata_dict = {
            "Type": self._method_type,
            "Kind": f"{self._xyz.upper() if isinstance(self._xyz, str) else 'ORIENTATIONAL'}",
            "States": self._result.shape[1],
            "Precision": settings.precision.upper(),
            "Description": f"Total angular momentum matrices from Group '{self._group_name}'."
        }
        self._data_dict = {"TOTAL_ANGULAR_MOMENTUM_MATRICES": (self._result, f"{str(self._xyz).upper()}{' [(x-0, y-1, z-2), :, :]' if isinstance(self._xyz, str) and self._xyz == 'xyz' else ''} component{'s' if isinstance(self._xyz, str) and self._xyz == 'xyz' else ''} of the total angular momentum.")}
        if self._rotation is not None:
            self._data_dict["ROTATION"] = (self._rotation, "Rotation used to rotate the total angular momentum components.")
    
    def _load_from_file(self):
        self._result = self._slt_group["TOTAL_ANGULAR_MOMENTUM_MATRICES"][:]

    #TODO: plot
    def _plot(self):
        pass

    #TODO: df
    def _to_data_frame(self):
        pass


class SltStatesTotalAngularMomenta(_SingleProcessed):

    __slots__ = _SingleProcessed.__slots__ + ["_xyz", "_start_state", "_stop_state", "_rotation"]
     
    def __init__(self, slt_group, xyz='xyz', start_state=0, stop_state=0, rotation=None, slt_save=None) -> None:
        super().__init__(slt_group, slt_save)
        self._method_name = "States' total angular momenta"
        self._method_type = "STATES_TOTAL_ANGULAR_MOMENTA"
        self._xyz = xyz
        self._start_state = start_state
        self._stop_state = stop_state
        self._rotation = rotation

    def _executor(self):
        return slpjm_components_driver(self._slt_group, "diagonal", "j", self._xyz, self._start_state, self._stop_state, self._rotation)

    def _save(self):
        self._metadata_dict = {
            "Type": self._method_type,
            "Kind": f"{self._xyz.upper() if isinstance(self._xyz, str) else 'ORIENTATIONAL'}",
            "States": self._result.shape[1] if self._xyz == "xyz" or isinstance(self._xyz, ndarray) else self._result.shape[0],
            "Precision": settings.precision.upper(),
            "Description": f"States' expectation values of the total angular momentum from Group '{self._group_name}'."
        }
        self._data_dict = {"STATES_TOTAL_ANGULAR_MOMENTA": (self._result, f"{str(self._xyz).upper()}{' [(x-0, y-1, z-2), :]' if isinstance(self._xyz, str) and self._xyz == 'xyz' else ''} component{'s' if isinstance(self._xyz, str) and self._xyz == 'xyz' else ''} of the states's total angular momenta.")}
        if self._rotation is not None:
            self._data_dict["ROTATION"] = (self._rotation, "Rotation used to rotate the total angular momentum components.")
    
    def _load_from_file(self):
        self._result = self._slt_group["STATES_TOTAL_ANGULAR_MOMENTA"][:]

    #TODO: plot
    def _plot(self):
        pass

    #TODO: df
    def _to_data_frame(self):
        pass


class SltMagneticDipoleMomentumMatrices(_SingleProcessed):

    __slots__ = _SingleProcessed.__slots__ + ["_xyz", "_start_state", "_stop_state", "_rotation"]
     
    def __init__(self, slt_group, xyz='xyz', start_state=0, stop_state=0, rotation=None, slt_save=None) -> None:
        super().__init__(slt_group, slt_save)
        self._method_name = "Magnetic dipole momentum matrices"
        self._method_type = "MAGNETIC_DIPOLE_MOMENTA"
        self._xyz = xyz
        self._start_state = start_state
        self._stop_state = stop_state
        self._rotation = rotation

    def _executor(self):
        return slpjm_components_driver(self._slt_group, "full", "m", self._xyz, self._start_state, self._stop_state, self._rotation)

    def _save(self):
        self._metadata_dict = {
            "Type": self._method_type,
            "Kind": f"{self._xyz.upper() if isinstance(self._xyz, str) else 'ORIENTATIONAL'}",
            "States": self._result.shape[1],
            "Precision": settings.precision.upper(),
            "Description": f"Magnetic dipole momentum matrices from Group '{self._group_name}'."
        }
        self._data_dict = {"MAGNETIC_DIPOLE_MOMENTUM_MATRICES": (self._result, f"{str(self._xyz).upper()}{' [(x-0, y-1, z-2), :, :]' if isinstance(self._xyz, str) and self._xyz == 'xyz' else ''} component{'s' if isinstance(self._xyz, str) and self._xyz == 'xyz' else ''} of the magnetic dipole momentum.")}
        if self._rotation is not None:
            self._data_dict["ROTATION"] = (self._rotation, "Rotation used to rotate the magnetic dipole momentum components.")
    
    def _load_from_file(self):
        self._result = self._slt_group["MAGNETIC_DIPOLE_MOMENTUM_MATRICES"][:]

    #TODO: plot
    def _plot(self):
        pass

    #TODO: df
    def _to_data_frame(self):
        pass


class SltStatesMagneticDipoleMomenta(_SingleProcessed):

    __slots__ = _SingleProcessed.__slots__ + ["_xyz", "_start_state", "_stop_state", "_rotation"]
     
    def __init__(self, slt_group, xyz='xyz', start_state=0, stop_state=0, rotation=None, slt_save=None) -> None:
        super().__init__(slt_group, slt_save)
        self._method_name = "States' magnetic dipole momenta"
        self._method_type = "STATES_MAGNETIC_DIPOLE_MOMENTA"
        self._xyz = xyz
        self._start_state = start_state
        self._stop_state = stop_state
        self._rotation = rotation

    def _executor(self):
        return slpjm_components_driver(self._slt_group, "diagonal", "m", self._xyz, self._start_state, self._stop_state, self._rotation)

    def _save(self):
        self._metadata_dict = {
            "Type": self._method_type,
            "Kind": f"{self._xyz.upper() if isinstance(self._xyz, str) else 'ORIENTATIONAL'}",
            "States": self._result.shape[1] if self._xyz == "xyz" or isinstance(self._xyz, ndarray) else self._result.shape[0],
            "Precision": settings.precision.upper(),
            "Description": f"States' expectation values of the magnetic dipole momentum from Group '{self._group_name}'."
        }
        self._data_dict = {"STATES_MAGNETIC_DIPOLE_MOMENTA": (self._result, f"{str(self._xyz).upper()}{' [(x-0, y-1, z-2), :]' if isinstance(self._xyz, str) and self._xyz == 'xyz' else ''} component{'s' if isinstance(self._xyz, str) and self._xyz == 'xyz' else ''} of the states's magnetic dipole momenta.")}
        if self._rotation is not None:
            self._data_dict["ROTATION"] = (self._rotation, "Rotation used to rotate the magnetic dipole momentum components.")
    
    def _load_from_file(self):
        self._result = self._slt_group["STATES_MAGNETIC_DIPOLE_MOMENTA"][:]

    #TODO: plot
    def _plot(self):
        pass

    #TODO: df
    def _to_data_frame(self):
        pass


class SltPropertyUnderMagneticField(_MultiProcessed):

    __slots__ = _MultiProcessed.__slots__ + ["_mode", "_matrix", "_return_energies", "_energies", "_direction", "_magnetic_fields", "_orientations", "_states_cutoff", "_rotation", "_electric_field_vector", "_hyperfine"]

    def __repr__():
        pass # tutaj jak z SLPGroup różne rperezentacje

    def __init__(self, slt_group,
        mode: Literal["s", "l", "p", "j", "m"],
        full_matrix: bool,
        return_energies: bool,
        direction: Union[ndarray, Literal["xyz"]],
        magnetic_fields: ndarray,
        orientations: ndarray,
        number_of_states: int,
        states_cutoff: list = [0,0],
        rotation: ndarray = None,
        electric_field_vector: ndarray = None,
        hyperfine: dict = None,
        number_cpu: int = 1,
        number_threads: int = 1,
        autotune: bool = False,
        slt_save: str = None,
        smm: SharedMemoryManager = None,
        terminate_event: Event = None,
        ) -> None:
        super().__init__(slt_group, magnetic_fields.shape[0] * orientations.shape[0], number_cpu, number_threads, autotune, smm, terminate_event, slt_save)
        self._method_name = "Zeeman Splitting"
        self._method_type = "ZEEMAN_SPLITTING"
        self._magnetic_fields = magnetic_fields
        self._orientations = orientations
        self._states_cutoff = states_cutoff
        self._rotation = rotation
        self._electric_field_vector = electric_field_vector
        self._hyperfine = hyperfine
        self._args = (number_of_states, electric_field_vector)
        self._executor_proxy = _zeeman_splitting_proxy
        self._slt_hamiltonian = self._slt_group._hamiltonian_from_slt_group(self._states_cutoff, self._rotation, self._hyperfine, False)
        self._slt_hamiltonian._mode = "em" if electric_field_vector is None else "emp"
    
    def _load_args_arrays(self):
        self._args_arrays = (*self._slt_hamiltonian.arrays_to_shared_memory, self._magnetic_fields, self._orientations)
        if self._orientations.shape[1] == 4:
            self._returns = True
        elif self._orientations.shape[1] == 3:
            self._result = zeros((self._magnetic_fields.shape[0] * self._orientations.shape[0], self._args[0]), dtype=self._magnetic_fields.dtype, order="C")
            self._result_shape = (self._orientations.shape[0], self._magnetic_fields.shape[0], self._args[0])
    
    def _gather_results(self, result_queue):
        zeeman_splitting_array = zeros((self._magnetic_fields.shape[0], self._args[0]), dtype=self._magnetic_fields.dtype)
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

    def _plot(self, **kwargs):
        pass
 
    def _to_data_frame(self):
        pass

    #matrix, states, etc under field with energies returned also in input parser add to error with direct acces to properties to use property_under_magnetic_field instead!! for slothpy hamiltonians with field [0,0,0]

class SltZeemanSplitting(_MultiProcessed):

    __slots__ = _MultiProcessed.__slots__ + ["_magnetic_fields", "_orientations", "_states_cutoff", "_rotation", "_electric_field_vector", "_hyperfine"]
     
    def __init__(self, slt_group,
        magnetic_fields: ndarray,
        orientations: ndarray,
        number_of_states: int,
        states_cutoff: list = [0,0],
        rotation: ndarray = None,
        electric_field_vector: ndarray = None,
        hyperfine: dict = None,
        number_cpu: int = 1,
        number_threads: int = 1,
        autotune: bool = False,
        slt_save: str = None,
        smm: SharedMemoryManager = None,
        terminate_event: Event = None,
        ) -> None:
        super().__init__(slt_group, magnetic_fields.shape[0] * orientations.shape[0], number_cpu, number_threads, autotune, smm, terminate_event, slt_save)
        self._method_name = "Zeeman Splitting"
        self._method_type = "ZEEMAN_SPLITTING"
        self._magnetic_fields = magnetic_fields
        self._orientations = orientations
        self._states_cutoff = states_cutoff
        self._rotation = rotation
        self._electric_field_vector = electric_field_vector
        self._hyperfine = hyperfine
        self._args = (number_of_states, electric_field_vector)
        self._executor_proxy = _zeeman_splitting_proxy
        self._slt_hamiltonian = self._slt_group._hamiltonian_from_slt_group(self._states_cutoff, self._rotation, self._hyperfine, False)
        self._slt_hamiltonian._mode = "em" if electric_field_vector is None else "emp"
    
    def _load_args_arrays(self):
        self._args_arrays = (*self._slt_hamiltonian.arrays_to_shared_memory, self._magnetic_fields, self._orientations)
        if self._orientations.shape[1] == 4:
            self._returns = True
        elif self._orientations.shape[1] == 3:
            self._result = zeros((self._magnetic_fields.shape[0] * self._orientations.shape[0], self._args[0]), dtype=self._magnetic_fields.dtype, order="C")
            self._result_shape = (self._orientations.shape[0], self._magnetic_fields.shape[0], self._args[0])
    
    def _gather_results(self, result_queue):
        zeeman_splitting_array = zeros((self._magnetic_fields.shape[0], self._args[0]), dtype=self._magnetic_fields.dtype)
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

    def _plot(self, **kwargs):
        from slothpy._general_utilities._ploting_utilities import _plot_zeeman_splitting
        _plot_zeeman_splitting(self._, self._result, self._magnetic_fields, **kwargs)
 
    def _to_data_frame(self):
        pass


class SltMagnetisation(_MultiProcessed):

    __slots__ = _MultiProcessed.__slots__ + ["_magnetic_fields", "_orientations", "_temperatures", "_states_cutoff", "_rotation", "_electric_field_vector", "_hyperfine"]
     
    def __init__(self, slt_group,
        magnetic_fields: ndarray,
        orientations: ndarray,
        temperatures: ndarray,
        states_cutoff: list = [0,0],
        rotation: ndarray = None,
        electric_field_vector: ndarray = None,
        hyperfine: dict = None,
        number_cpu: int = 1,
        number_threads: int = 1,
        autotune: bool = False,
        slt_save: str = None,
        smm: SharedMemoryManager = None,
        terminate_event: Event = None,
        ) -> None:
        super().__init__(slt_group, magnetic_fields.shape[0] * orientations.shape[0] , number_cpu, number_threads, autotune, smm, terminate_event, slt_save)
        self._method_name = "Magnetisation"
        self._method_type = "MAGNETISATION"
        self._magnetic_fields = magnetic_fields
        self._orientations = orientations
        self._temperatures = temperatures
        self._states_cutoff = states_cutoff
        self._rotation = rotation
        self._electric_field_vector = electric_field_vector
        self._hyperfine = hyperfine
        self._args = (electric_field_vector,)
        self._executor_proxy = _magnetisation_proxy
        self._slt_hamiltonian = self._slt_group._hamiltonian_from_slt_group(self._states_cutoff, self._rotation, self._hyperfine, True)
        self._slt_hamiltonian._mode = "em" if electric_field_vector is None else "emp"
    
    def _load_args_arrays(self):
        self._args_arrays = (*self._slt_hamiltonian.arrays_to_shared_memory, self._magnetic_fields, self._orientations, self._temperatures)
        if self._orientations.shape[1] == 4:
            self._returns = True
        elif self._orientations.shape[1] == 3:
            self._result = zeros((self._magnetic_fields.shape[0] * self._orientations.shape[0], self._temperatures.shape[0]), dtype=self._magnetic_fields.dtype, order="C")
            self._result_shape = (self._orientations.shape[0], self._magnetic_fields.shape[0], self._temperatures.shape[0])
            self._transpose_result = (0, 2, 1)
    
    def _gather_results(self, result_queue):
        result_magnetisation_array = zeros((self._magnetic_fields.shape[0], self._temperatures.shape[0]), dtype=self._temperatures.dtype)
        while not result_queue.empty():
            start_field_index, end_field_index, magnetisation_array = result_queue.get()
            for i, j in enumerate(range(start_field_index, end_field_index)):
                result_magnetisation_array[j, :] += magnetisation_array[i, :]
        return result_magnetisation_array.T

    def _save(self):
        self._metadata_dict = {
            "Type": self._method_type,
            "Kind": "AVERAGE" if self._result.ndim == 2 else "DIRECTIONAL",
            "Precision": settings.precision.upper(),
            "Description": f"Group containing magnetisation calculated from Group '{self._group_name}'."
        }
        self._data_dict = {
            "MAGNETISATION": (self._result, "Dataset containing magnetisation in the form {}".format("[temperatures, fields]" if self._result.ndim == 2 else "[orientations, temperatures, fields]")),
            "MAGNETIC_FIELDS": (self._magnetic_fields, "Dataset containing magnetic field (T) values used in the simulation."),
            "ORIENTATIONS": (self._orientations, "Dataset containing magnetic fields' orientation grid used in the simulation."),
            "TEMPERATURES": (self._temperatures, "Dataset containing temperature (K) values used in the simulation.")
        }

    def _load_from_file(self):
        self._result = self._slt_group["MAGNETISATION"][:]
        self._magnetic_fields = self._slt_group["MAGNETIC_FIELDS"][:]
        self._orientations = self._slt_group["ORIENTATIONS"][:]

    def _plot(self):
        pass
 
    def _to_data_frame(self):
        pass