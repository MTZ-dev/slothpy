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
    fields: ArrayLike,
    temperatures: ArrayLike,
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
        fp.write("Temperature (K),Field (Oe),Wave Frequency (Hz),AC X' (cm3/mol),AC X\" (cm3/mol)\n")
        writer = csv.writer(fp, quoting=csv.QUOTE_NONE, escapechar="\\")
        writer.writerows(rows)

def export_tau_csv(
    fields: ArrayLike,
    temperatures: ArrayLike,
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


def export_dos_csv(frequency: Sequence[float], convolution: Sequence[float], filepath: Path | str) -> None:
    """
    Save phonon DOS convolution vs frequency as a simple CSV.
    Header lines keep the same style as your AC files.
    """
    p = Path(filepath)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8", newline="") as f:
        f.write("[DOS]\n")
        w = csv.writer(f)
        w.writerow(["Frequency", "Convolution"])
        for x, y in zip(frequency, convolution):
            w.writerow([x, y])