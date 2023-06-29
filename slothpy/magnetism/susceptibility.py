import numpy as np
from slothpy.magnetism.magnetisation import mth
from slothpy.general_utilities.math_expresions import finite_diff_stencil

def chitht(filename: str, group: str, fields: np.ndarray, states_cutoff: int, temperatures: np.ndarray, num_cpu: int, num_of_points: int, delta_h: np.float64, grid: np.ndarray = None, exp: bool = False) -> np.ndarray:
    """
    Calculates chiT(H,T) using data from a HDF5 file for given field, states cutoff, temperatures, and optional grid (XYZ if not present).

    Args:
        path (str): Path to the file.
        hdf5_file (str): Name of the HDF5 file.
        field (np.ndarray[np.float64]): Array of fields.
        states_cutoff (int): Number of states cutoff value.
        temperatures (np.ndarray[np.float64]): Array of temperatures.
        num_cpu (int): Number of CPU used for to call mth function for M(T,H) calculation
        grid (np.ndarray[np.float64], optional): Grid array for direction averaging. Defaults to XYZ.

    Returns:
        np.ndarray[np.float64]: Array of chit values.

    """

    if num_of_points < 0 or (not isinstance(num_of_points, int)):

        raise ValueError(f'Number of points for finite difference method has to be a possitive integer!')
    
    bohr_magneton_to_cm3 = 0.5584938904 # Conversion factor for chi in cm3
    
    # Comments here modyfied!!!!
    chitht_array = np.zeros((fields.shape[0], temperatures.shape[0]))

    # Default XYZ grid
    if grid is None or grid == None:
        grid = np.array([[1., 0., 0., 0.3333333333333333], [0., 1., 0., 0.3333333333333333], [0., 0., 1., 0.3333333333333333]], dtype=np.float64)

    # Experimentalist model
    if (exp == True) or (num_of_points == 0):

        for index_field, field in enumerate(fields):

            mth_array = mth(filename, group, states_cutoff, np.array([field]), grid, temperatures, num_cpu)

            for index, temp in enumerate(temperatures):
                chit[index] = temp * mth_array[index] / field
            
            chitht_array[index_field, :] = chit * bohr_magneton_to_cm3

    else:

        for index_field, field in enumerate(fields):

            # Set fields for finite difference method
            fields_diff = np.arange(-num_of_points, num_of_points + 1).astype(np.int64) * delta_h + field
            fields_diff = fields_diff.astype(np.float64)

            # Initialize result array
            chit = np.zeros_like(temperatures)

            # Get M(t,H) for two adjacent values of field
            mth_array = mth(filename, group, states_cutoff, fields_diff, grid, temperatures, num_cpu)

            stencil_coeff = finite_diff_stencil(1, num_of_points, delta_h)

            # Numerical derivative of M(T,H) around given field value 
            for index, temp in enumerate(temperatures):
                chit[index] = temp * np.dot(mth_array[index], stencil_coeff)
            
            chitht_array[index_field, :] = chit * bohr_magneton_to_cm3

    return chitht_array