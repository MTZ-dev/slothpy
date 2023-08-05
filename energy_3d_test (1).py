import time
import slothpy as slt
import numpy as np

if __name__ == '__main__':
    fields = np.linspace(0.01, 14, 60)
    temperatures = np.linspace(0.1,700, 60)

    NdCoNO2 = slt.compound_from_slt(".", "NdCoNO2")


    start_time = time.perf_counter()

    NdCoNO2.animate_3d("twoja_stara", 'hemholtz_energy', 'temperature', fps=10, dpi=100)

    end_time = time.perf_counter()

    print(f'{end_time - start_time} s')


