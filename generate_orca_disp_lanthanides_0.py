import slothpy as slt
import os

if __name__ == "__main__":

    file_path = "./seminarium/Ln.slt"

    if os.path.exists(file_path):
        os.remove(file_path)

    Ln = slt.unit_cell("/home/mikolaj/Sloth/Sloth/seminarium/YCo_opt_cell.xyz", "./seminarium/Ln.slt", "Y_unit_cell", cell_vectors=[[7.5543780496917456E+000,  0.0000000000000000E+000,  0.0000000000000000E+000], [-3.7771890248458728E+000,  6.5422833008245931E+000,  0.0000000000000000E+000], [0.0000000000000000E+000,  0.0000000000000000E+000,  1.2574257142334060E+001]])
    
    lanthanide_data = {
    'Ce': {'nel':1, 'S_list':[0.5], 'NDoubGtensor':3},
    'Pr': {'nel':2, 'S_list':[1, 0], 'NDoubGtensor':5},
    'Nd': {'nel':3, 'S_list':[1.5, 0.5], 'NDoubGtensor':5},
    'Pm': {'nel':4, 'S_list':[2, 1, 0], 'NDoubGtensor':5},
    'Sm': {'nel':5, 'S_list':[2.5, 1.5, 0.5], 'NDoubGtensor':3},
    'Eu': {'nel':6, 'S_list':[3, 2, 1, 0], 'NDoubGtensor':1},
    'Gd': {'nel':7, 'S_list':[3.5, 2.5, 1.5, 0.5], 'NDoubGtensor':4},
    'Tb': {'nel':8, 'S_list':[3, 2, 1, 0], 'NDoubGtensor':7},
    'Dy': {'nel':9, 'S_list':[2.5, 1.5, 0.5], 'NDoubGtensor':8},
    'Ho': {'nel':10, 'S_list':[2, 1, 0], 'NDoubGtensor':9},
    'Er': {'nel':11, 'S_list':[1.5, 0.5], 'NDoubGtensor':8},
    'Tm': {'nel':12, 'S_list':[1, 0], 'NDoubGtensor':7},
    'Yb': {'nel':13, 'S_list':[0.5], 'NDoubGtensor':4}
    }

    lanthanide_data_single = {
    'Ce': {'nel':1, 'S_list':[0.5], 'NDoubGtensor':3},
    'Pr': {'nel':2, 'S_list':[1], 'NDoubGtensor':5},
    'Nd': {'nel':3, 'S_list':[1.5], 'NDoubGtensor':5},
    'Pm': {'nel':4, 'S_list':[2], 'NDoubGtensor':5},
    'Sm': {'nel':5, 'S_list':[2.5], 'NDoubGtensor':3},
    'Eu': {'nel':6, 'S_list':[3], 'NDoubGtensor':1},
    'Gd': {'nel':7, 'S_list':[3.5], 'NDoubGtensor':4},
    'Tb': {'nel':8, 'S_list':[3], 'NDoubGtensor':7},
    'Dy': {'nel':9, 'S_list':[2.5], 'NDoubGtensor':8},
    'Ho': {'nel':10, 'S_list':[2], 'NDoubGtensor':9},
    'Er': {'nel':11, 'S_list':[1.5], 'NDoubGtensor':8},
    'Tm': {'nel':12, 'S_list':[1], 'NDoubGtensor':7},
    'Yb': {'nel':13, 'S_list':[0.5], 'NDoubGtensor':4}
    }

    for lanthanide in lanthanide_data_single.keys():
        # Choose small or big cluster
        Ln = slt.xyz("/home/mikolaj/Sloth/Sloth/seminarium/YCo_1.xyz", "./seminarium/Ln.slt", f"{lanthanide}.xyz", -3, int(2*lanthanide_data[lanthanide]["S_list"][0]+1))
        # Ln = slt.xyz("/home/mikolaj/Sloth/Sloth/seminarium/YCo_3.xyz", "./seminarium/Ln.slt", f"{lanthanide}.xyz", 21, int(2*lanthanide_data[lanthanide]["S_list"][0]+1))
        Ln[f"{lanthanide}.xyz"].replace_atoms([0], [lanthanide])
        Ln[f"{lanthanide}.xyz"].generate_finite_stencil_displacements_reduced_to_unit_cell("Y_unit_cell", [7.554490, 8.722928, 9.430707], 1, 0.01, custom_directory=f"/home/mikolaj/Ac_relacs_publication/derivatives_displacements/unit_cell_01_single/{lanthanide}Co")
        # Ln[f"{lanthanide}.xyz"].generate_finite_stencil_displacements_across_unit_cells("Y_unit_cell", [7.554490, 8.722928, 9.430707], 1, 0.0001, custom_directory=f"/home/mikolaj/Data/Displacements_cluster_0001/{lanthanide}Co_displ_cluster")