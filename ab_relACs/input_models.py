# SlothPy
# Copyright (C) 2025 Mikolaj Tadeusz Zychowicz (MTZ)

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

from typing import Any, List

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator, field_validator, field_serializer

from utils import eval_numpy_expr

def _to_ndarray(v: Any, *, dtype=float) -> np.ndarray:
    if isinstance(v, np.ndarray):
        return v.astype(dtype, copy=False)
    if hasattr(v, "__iter__") and not isinstance(v, (str, bytes)):
        return np.asarray(list(v), dtype=dtype)
    if isinstance(v, (int, float, np.integer, np.floating)):
        return np.asarray([v], dtype=dtype)
    raise TypeError(f"Cannot coerce {type(v)!r} to numpy array")

class InpRelacs(BaseModel):
    model_config = ConfigDict(extra='forbid', arbitrary_types_allowed=True)
    slt_filepath: str = ""
    number_cpus: int = 1

    orientations: np.ndarray | List[list] | str | int = 0
    fields: np.ndarray | List[float] | str = Field(default_factory=list)
    temperatures: np.ndarray | List[float] | str = Field(default_factory=list)
    frequencies: np.ndarray | List[float] | str = "np.logspace(0, 6, 100)"

    states_number: int = 0
    degeneracy_tolerance: float = 1e-5
    psi_frequency_shift: bool = False
    initial_correlation: bool = False
    omega_loop: bool = False
    chi_s: bool | np.ndarray | List[float] = False
    symm: bool = True

    n_points: np.ndarray | List[int] = [1]
    q_ranges: List[float] = [0.5]
    broadening: str = "gaussian" # "lorentzian"
    fwhm: np.ndarray | List[float] = [0.1]
    adaptive_fwhm: bool = False
    modes_low: float = 0.0
    modes_high: float = 0.1
    cutoff_fwhm: float = 1000
    direct: bool = True

    chi_csv_path: str = ""
    tau_21_csv_path: str = ""
    tau_41_csv_path: str= ""
    show_plot: bool = True

    @field_validator("fields", "temperatures", "frequencies", "fwhm", mode="before")
    @classmethod
    def _coerce_1d_float_arrays(cls, v):
        if isinstance(v, np.ndarray):
            return v.astype(float, copy=False)
        if isinstance(v, str) and v.strip().startswith("np."):
            arr = eval_numpy_expr(v)
            return _to_ndarray(arr, dtype=float)
        return _to_ndarray(v, dtype=float)

    @field_validator("orientations", mode="before")
    @classmethod
    def _coerce_orientations(cls, v):
        if isinstance(v, int):
            return v
        if isinstance(v, str) and v.strip().startswith("np."):
            v = eval_numpy_expr(v)
        arr = _to_ndarray(v, dtype=float)
        return arr
    
    @field_validator("n_points", mode="before")
    @classmethod
    def _coerce_n_points(cls, v):
        if isinstance(v, str) and v.strip().startswith("np."):
            v = eval_numpy_expr(v)
        arr = _to_ndarray(v, dtype=int)
        return arr

    @field_validator("chi_s", mode="before")
    @classmethod
    def _coerce_chi_s(cls, v):
        if isinstance(v, bool):
            return v
        if isinstance(v, str) and v.strip().startswith("np."):
            v = eval_numpy_expr(v)
        return _to_ndarray(v, dtype=float)

    @field_serializer("fields", "temperatures", "frequencies", "orientations", "chi_s", "n_points", "fwhm")
    def _serialize_arrays(self, v):
        if isinstance(v, np.ndarray):
            return v.tolist()
        return v

class InpSupercell(BaseModel):
    model_config = ConfigDict(extra='forbid')
    xyz_path: str = ""
    group_name: str = ""
    cell_params: List[float] = Field(default_factory=list)
    nx: int = 0
    ny: int = 0
    nz: int = 0
    replace_atoms: List[int] = Field(default_factory=list)
    new_atoms:List[str] = Field(default_factory=list)

class InpHessian(BaseModel):
    model_config = ConfigDict(extra='forbid')
    path: str = ""
    group_name: str = ""
    displacement_number: int = 0
    step: float = 0.0
    accoustic_sum_rule: str = ""
    dos: str = ""

class InpSpinPhonon(BaseModel):
    model_config = ConfigDict(extra='forbid')
    path: str = ""
    group_name: str = ""
    orca_fragovl_path: str = ""
    displacement_number: int = 0
    step: float = 0.0

from typing import Optional

class InpWeightedDOS(BaseModel):
    model_config = ConfigDict(extra='forbid', arbitrary_types_allowed=True)

    # Histogram / DOS construction
    resolution: int = 2000
    density: bool = False

    # Convolution of histogram to smooth curve
    # Use None to disable convolution
    convolution: Optional[str] = "gaussian"   # None / "gaussian" / "lorentzian"
    fwhm: float = 10.0                        # cm^-1 (for convolution curve)

    # Spin–phonon coupling weight definition
    weight_mode: str = "fro_offdiag"          # "fro", "fro_offdiag", "thermal_sym", "thermal_offdiag"
    temperature: Optional[float] = None       # required for thermal_* modes

    # Frequency handling on the DOS axis
    dos_freq: str = "raw"                     # "raw" or "abs"
    eps_freq_cm1: float = 1e-12               # protects 1/sqrt(0) in Y construction

    # Plot controls
    save_path: Optional[str] = None
    show: bool = True
    title: str = "Spin--phonon weighted phonon DOS"
    energy_lines: np.ndarray | List[float] | str | float | int = Field(default_factory=list)

    # ---------------- validators ----------------

    @field_validator("resolution")
    @classmethod
    def _check_resolution(cls, v):
        if v < 10:
            raise ValueError("resolution must be >= 10")
        return int(v)

    @field_validator("convolution")
    @classmethod
    def _check_convolution(cls, v):
        if v is None:
            return None
        v = str(v).strip().lower()
        if v in ("none", "null"):
            return None
        if v not in ("gaussian", "lorentzian"):
            raise ValueError("convolution must be None, 'gaussian', or 'lorentzian'")
        return v

    @field_validator("dos_freq")
    @classmethod
    def _check_dos_freq(cls, v):
        v = str(v).strip().lower()
        if v not in ("raw", "abs"):
            raise ValueError("dos_freq must be 'raw' or 'abs'")
        return v

    @field_validator("weight_mode")
    @classmethod
    def _check_weight_mode(cls, v):
        v = str(v).strip().lower()
        allowed = ("fro", "fro_offdiag", "thermal_sym", "thermal_offdiag")
        if v not in allowed:
            raise ValueError(f"weight_mode must be one of {allowed}")
        return v

    @model_validator(mode="after")
    def _check_thermal_requires_temperature(self):
        if self.weight_mode.startswith("thermal") and self.temperature is None:
            raise ValueError("temperature must be set when weight_mode is thermal_*")
        return self

    @field_validator("fwhm", "eps_freq_cm1")
    @classmethod
    def _check_positive(cls, v, info):
        v = float(v)
        if v <= 0.0:
            raise ValueError(f"{info.field_name} must be > 0")
        return v

    @field_validator("energy_lines", mode="before")
    @classmethod
    def _coerce_energy_lines(cls, v):
        # Accept: [], scalar, list, np.array, or "np.linspace(...)" etc.
        if v is None:
            return np.asarray([], dtype=float)
        if isinstance(v, (int, float, np.integer, np.floating)):
            return np.asarray([v], dtype=float)
        if isinstance(v, np.ndarray):
            return v.astype(float, copy=False).ravel()
        if isinstance(v, str) and v.strip().startswith("np."):
            arr = eval_numpy_expr(v)
            return _to_ndarray(arr, dtype=float).ravel()
        return _to_ndarray(v, dtype=float).ravel()

    @field_serializer("energy_lines")
    def _serialize_energy_lines(self, v):
        if isinstance(v, np.ndarray):
            return v.tolist()
        return v

        
class AppConfig(BaseModel):
    relacs: InpRelacs = Field(default_factory=InpRelacs)
    supercell: InpSupercell = Field(default_factory=InpSupercell)
    hessian: InpHessian = Field(default_factory=InpHessian)
    spin_phonon: InpSpinPhonon = Field(default_factory=InpSpinPhonon)
    weighted_dos: InpWeightedDOS = Field(default_factory=InpWeightedDOS)

    @model_validator(mode="after")
    def _post(self):
        #TODO: implement validation and custom dataclasses
        return self