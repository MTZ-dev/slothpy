import os
os.environ['OMP_NUM_THREADS'] = '2'
import multiprocessing
import re
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib import cm
from numba import jit, cfunc
import timeit
from mpl_toolkits.mplot3d import Axes3D
import math


#TO DO - print the elipsoid of main magnetic axes


@jit('float64(complex128[:,:,:], float64[:], float64, float64[:], float64)', nopython=True, cache=True, nogil=True)
def calculate_mth_grid(magnetic_moment: np.ndarray, soc_energies: np.ndarray, field: np.float64, grid: np.ndarray, temperature: np.float64) -> np.float64:

    bohr_magneton = 2.127191078656686e-06 # Bohr magneton in a.u./T

    # Initialize arrays as contiguous
    magnetic_moment = np.ascontiguousarray(magnetic_moment)
    soc_energies = np.ascontiguousarray(soc_energies)


    # Construct Zeeman matrix
    orient = -field * bohr_magneton * grid[:3]
    zeeman_matrix = magnetic_moment[0] * orient[0] + magnetic_moment[1] * orient[1] + magnetic_moment[2] * orient[2]

    # Add SOC energy to diagonal of Hamiltonian(Zeeman) matrix
    for k in range(zeeman_matrix.shape[0]):
        zeeman_matrix[k, k] += soc_energies[k]

    # Diagonalize full Hamiltonian matrix
    eigenvalues, eigenvectors = np.linalg.eigh(zeeman_matrix)
    eigenvalues = np.ascontiguousarray(eigenvalues)
    eigenvectors = np.ascontiguousarray(eigenvectors)

    # Transform momenta according to the new eigenvectors
    states_momenta = eigenvectors.conj().T @ (
        grid[0] * magnetic_moment[0]
        + grid[1] * magnetic_moment[1]
        + grid[2] * magnetic_moment[2]
    ) @ eigenvectors

    # Get diagonal momenta of the new states
    states_momenta = np.diag(states_momenta).real.astype(np.float64)

    # Compute partition function and magnetization
    mth = calculate_magnetization(eigenvalues, states_momenta, temperature)

    return mth


def calculate_mth_grid_wrapper(args):

    # Unpack arguments and call the function
    mth = calculate_mth_grid(*args)

    return mth


def arg_iter_mag_3d(magnetic_moment, soc_energies, field, theta, phi, temperature):
    
    for i in range(phi.shape[0]):
        for j in range(phi.shape[1]):
            yield (magnetic_moment, soc_energies, field, np.array([np.sin(phi[i, j]) * np.cos(theta[i, j]), np.sin(phi[i, j]) * np.sin(theta[i, j]), np.cos(phi[i, j])]), temperature)


def mag_3d(path: str, hdf5_file: str, states_cutoff: int, field: np.ndarray, grid: int, temperature: np.float64, num_cpu: int) -> np.ndarray:

    # Get number of parallel proceses to be used
    num_process = get_num_of_processes(num_cpu)

    # Create a gird
    theta = np.linspace(0, 2*np.pi, grid)
    phi = np.linspace(0, np.pi, grid)
    theta, phi = np.meshgrid(theta, phi)

    # Initialize the result array
    mag_3d_array = np.zeros_like(phi, dtype=np.float64)

    # Read data from HDF5 file
    magnetic_moment, soc_energies = get_soc_moment_energies_from_hdf5_orca(path, hdf5_file, states_cutoff)

    # Parallel M(T,H) calculation over different grid points
    with multiprocessing.Pool(num_process) as p:
        mth = p.map(calculate_mth_grid_wrapper, arg_iter_mag_3d(magnetic_moment, soc_energies, field, theta, phi, temperature))

    index = 0

    # Collecting results
    for i in range(phi.shape[0]):
        for j in range(phi.shape[1]):
            mag_3d_array[i, j] = mth[index]
            index += 1

    x = np.sin(phi) * np.cos(theta) * mag_3d_array
    y = np.sin(phi) * np.sin(theta) * mag_3d_array
    z = np.cos(phi) * mag_3d_array

    return x, y, z

# zastanów się czy numerycznie to robić też
def chi_exp_3d(path: str, hdf5_file: str, states_cutoff: int, field: np.ndarray, grid: int, temperature: np.float64, num_cpu: int) -> np.ndarray:

    x, y, z = mag_3d(path, hdf5_file, states_cutoff, field, grid, temperature, num_cpu)

    x /= field
    y /= field
    z /= field

    return x, y, z



