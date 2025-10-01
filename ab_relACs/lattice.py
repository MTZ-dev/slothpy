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

from typing import Tuple

import numpy as np

import slothpy as slt
slt.set_default_error_reporting_mode()
from slothpy.core._slt_file import SltHessian
from slothpy._general_utilities._io import _hamiltonian_derivatives_from_dir_to_slt
from input_models import AppConfig
from utils import h5_has_group

def get_hessian_recip_axes_masses_inv_sqrt_spin_phonon(cfg: AppConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    if not h5_has_group(cfg.relacs.slt_filepath, cfg.supercell.group_name):
        slt_file = slt.supercell(cfg.supercell.xyz_path, cfg.relacs.slt_filepath,
                                cfg.supercell.group_name, cfg.supercell.nx,
                                cfg.supercell.ny, cfg.supercell.nz,
                                supercell_params=cfg.supercell.cell_params)
    else:
        slt_file = slt.slt_file(cfg.relacs.slt_filepath)
    
    supercell = slt_file[cfg.supercell.group_name]
    supercell.replace_atoms(cfg.supercell.replace_atoms, cfg.supercell.new_atoms)

    if not h5_has_group(cfg.relacs.slt_filepath, cfg.hessian.group_name):
        hessian = supercell.hessian_from_finite_displacements(cfg.hessian.path, "CP2K",
                                cfg.hessian.group_name, cfg.hessian.displacement_number,
                                cfg.hessian.step)
    else:
        hessian = slt.slt_file(cfg.relacs.slt_filepath)[cfg.hessian.group_name]
    
    slt_hessian = SltHessian(hessian)
    masses_inv_sqrt = slt_hessian._masses_inv_sqrt
    recip_axes = slt_hessian.atoms_object().cell.reciprocal().cellpar()[:3]
    
    if not h5_has_group(cfg.relacs.slt_filepath, cfg.spin_phonon.group_name):
        _hamiltonian_derivatives_from_dir_to_slt(cfg.spin_phonon.path,
                            cfg.relacs.slt_filepath, cfg.spin_phonon.group_name,
                            cfg.spin_phonon.displacement_number, cfg.spin_phonon.step,
                            cfg.relacs.number_cpus, 1, "ORCA",
                            _orca_fragovl_path = cfg.spin_phonon.orca_fragovl_path)
    
    return slt_hessian.hessian()[:], recip_axes, masses_inv_sqrt
