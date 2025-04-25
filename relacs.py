import os
import re
import posixpath

import numpy as np
import h5py

import slothpy as slt
from slothpy._general_utilities._constants import A_BOHR, H_CM_1
from slothpy._general_utilities._io import _hamiltonian_derivatives_from_dir_to_slt
from slothpy._general_utilities._lapack import _zdot3d
from slothpy._general_utilities._math_expresions import _central_finite_difference_stencil

import matplotlib.pyplot as plt

def plot_complex_matrix(
    M: np.ndarray,
    *,
    title: str = "Complex Matrix",
    cmap: str = "bwr",
    vmin: float = None,
    vmax: float = None,
    figsize: tuple = (10, 4)
):
    """
    Plot real and imaginary parts of a 2D complex matrix as heatmaps.

    Parameters
    ----------
    M : (N,N) ndarray
        Complex matrix to visualize.
    title : str, optional
        Global title for the figure.
    cmap : str, optional
        Colormap to use for heatmaps (default 'bwr' — blue-white-red).
    vmin, vmax : float, optional
        Fixed limits for color scaling; if None, autoscale separately for Re and Im.
    figsize : (width, height) tuple, optional
        Size of the figure.
    """
    if M.ndim != 2 or not np.iscomplexobj(M):
        raise ValueError("Input must be a 2D complex matrix.")

    real_part = M.real
    imag_part = M.imag

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    im0 = axes[0].imshow(real_part, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[0].set_title("Real part")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(imag_part, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[1].set_title("Imaginary part")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    fig.suptitle(title, fontsize=16)
    for ax in axes:
        ax.set_xlabel('j')
        ax.set_ylabel('i')

    plt.tight_layout()
    plt.show()


def dofs_with_complete_displacements(h5_group: h5py.Group, displacement_number: int):
    
    base_regex = re.compile(r'^(-?\d+)_(-?\d+)_(-?\d+)_(-?\d+)$')

    complete = []

    def visitor(rel_name, obj):
        if not isinstance(obj, h5py.Dataset):
            return

        leaf_name = posixpath.basename(rel_name)
        m = base_regex.fullmatch(leaf_name)
        if not m:
            return

        dof, nx, ny, nz = map(int, m.groups())
        base = leaf_name
        parent = obj.parent

        missing = [d for d in range(-displacement_number, displacement_number + 1) if d != 0 and f"{base}_{d}" not in parent]

        if not missing:
            complete.append([dof, nx, ny, nz])

    h5_group.visititems(visitor)
    complete = np.asarray(complete, dtype=np.int64)

    return complete


def k_mch(dof_array: np.ndarray, group: h5py.Group, U_R0: np.ndarray):
    k_mch = []
    U_R0_T = U_R0.T
    U_R0_conj = U_R0.conj()

    for dof in dof_array:
        k_mch.append(U_R0_T @ group[f"{dof[0]}_{dof[1]}_{dof[2]}_{dof[3]}"][:] @ U_R0_conj)

    return np.asarray(k_mch, dtype=np.complex128)


def E_grad_k_U(dof_array: np.ndarray, group: h5py.Group, U_R0: np.ndarray, magnetic_field_vector: np.ndarray, displacement_number: int, degeneracy_tolerance: float = 1e-8):
    E_grad = []
    k_U = []
    finite_difference_stencil = _central_finite_difference_stencil(1, displacement_number, step * A_BOHR)

    for dof in dof_array:
        E_grad_component = np.zeros(U_R0.shape[0], dtype=np.float64)
        k_U_component = np.zeros_like(U_R0)
        stencil_index = -1
        for displacement in range(-displacement_number, displacement_number + 1):
            stencil_index += 1
            if displacement == 0:
                continue
            group_name = f"{dof[0]}_{dof[1]}_{dof[2]}_{dof[3]}_{displacement}"
            hamiltonian = group[f"{group_name}/HAMILTONIAN_MATRIX"][:]+_zdot3d(group[f"{group_name}/MAGNETIC_DIPOLE_MOMENTA"][:], -magnetic_field_vector)
            E, U_R0_delta = np.linalg.eigh(hamiltonian)

            S = U_R0_delta.conj().T @ U_R0

            projection_mask = np.abs(E[:, None] - E[None, :]) < degeneracy_tolerance
            M = np.where(projection_mask, S, 0.0) 
            u, s, vt = np.linalg.svd(M)
            U_R0_delta = U_R0_delta @ u @ vt

            E_grad_component += E * finite_difference_stencil[stencil_index]
            k_U_component += U_R0_delta * finite_difference_stencil[stencil_index]
        
        E_grad.append(E_grad_component)
        k_U.append(k_U_component.conj().T @ U_R0)
    
    return np.asarray(E_grad, dtype=np.float64), np.asarray(k_U, dtype=np.complex128)



if __name__ == "__main__":
    os.remove("./seminarium/import.slt")
    slt.set_default_error_reporting_mode()

    orca_fragovl_path = "/home/mikolaj/orca_6_0_1_avx2/orca_fragovl"
    dirpath = "/home/mikolaj/Data/Displacements_small/CeCo_displ"
    slt_filepath = "./seminarium/import.slt"
    group_name = "xxx"
    displacement_number = 1
    step = 0.025

    _hamiltonian_derivatives_from_dir_to_slt(dirpath, slt_filepath, group_name, displacement_number, step, 64, 1, "ORCA", False, False, False, orca_fragovl_path)
    
    magnetic_field = 0.1
    orientation = np.asarray([1,1,1], dtype=np.float64)
    magnetic_field_vector = magnetic_field / H_CM_1 * orientation / np.linalg.vector_norm(orientation)

    with h5py.File("./seminarium/import.slt", "r") as file:
        group = file[group_name]
        dof_array = dofs_with_complete_displacements(group, displacement_number)

        hamiltonian_total = group["0/HAMILTONIAN_MATRIX"][:]+_zdot3d(group["0/MAGNETIC_DIPOLE_MOMENTA"][:], -magnetic_field_vector)
        E, U_R0 = np.linalg.eigh(hamiltonian_total)

        k_mch_array = k_mch(dof_array, group, U_R0)
        E_grad, k_U_array = E_grad_k_U(dof_array, group, U_R0, magnetic_field_vector, displacement_number, 1e-8)
        
        H_grad = np.empty_like(k_mch_array)

        for i in range(H_grad.shape[0]):
            E_k_mch_k_U = (E[None, :]-E[:, None])*(k_mch_array[i] + k_U_array[i])
            np.fill_diagonal(E_k_mch_k_U, E_grad[i])
            H_grad[i] = E_k_mch_k_U
    
    plot_complex_matrix(H_grad[2])
