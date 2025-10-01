#!/usr/bin/env python3

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

import csv
from pathlib import Path
from typing import Sequence, Union

import numpy as np
ArrayLike = Union[Sequence, np.ndarray]

def export_susceptibility_csv(
    temperatures: ArrayLike,
    fields: ArrayLike,
    freqs: ArrayLike,
    chi: np.ndarray,
    filename: str | Path = "susceptibility_data.csv",
) -> None:

    T  = np.asarray(temperatures, dtype=np.float64)
    H  = np.asarray(fields, dtype=np.float64)
    f  = np.asarray(freqs, dtype=np.float64)
    chi = np.asarray(chi, dtype=np.complex128)

    if chi.shape != (H.size, T.size, f.size):
        raise ValueError(
            f"`chi` must have shape (n_fields, n_temps, n_freqs) = "
            f"({H.size}, {T.size}, {f.size}); got {chi.shape!r}."
        )

    H_grid, T_grid, f_grid = np.meshgrid(H, T, f, indexing="ij")
    rows = np.column_stack([
        T_grid.ravel(),        
        H_grid.ravel(),
        f_grid.ravel(),
        np.abs(chi.real).ravel(),
        np.abs(chi.imag).ravel(),
    ])

    filename = Path(filename)
    with filename.open(mode="w", newline="") as fp:
        fp.write("[Data]\n")
        fp.write("Temperature (K),Magnetic Field (Oe),AC Frequency (Hz),AC X'  (emu/Oe),AC X\" (emu/Oe)\n")
        writer = csv.writer(fp, quoting=csv.QUOTE_NONE, escapechar="\\")
        writer.writerows(rows)

def export_tau_csv(
    temperatures: ArrayLike,
    fields: ArrayLike,
    tau: np.ndarray,
    filename: str | Path = "tau_data.csv",
) -> None:

    T  = np.asarray(temperatures, dtype=np.float64)
    H  = np.asarray(fields, dtype=np.float64)
    tau = np.asarray(tau, dtype=np.float64)

    if tau.shape != (H.size, T.size):
        raise ValueError(
            f"`tau` must have shape (n_fields, n_temps) = "
            f"({H.size}, {T.size}); got {tau.shape!r}."
        )

    H_grid, T_grid = np.meshgrid(H, T, indexing="ij")
    rows = np.column_stack([
        T_grid.ravel(),
        H_grid.ravel(),
        tau.ravel(),
    ])

    filename = Path(filename)
    with filename.open(mode="w", newline="") as fp:
        fp.write("[Data]\n")
        fp.write("T,H,tau\n")
        writer = csv.writer(fp, quoting=csv.QUOTE_NONE, escapechar="\\")
        writer.writerows(rows)