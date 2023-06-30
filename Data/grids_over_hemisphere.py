import numpy as np

with open("grids_numpy", 'w') as f:
    for i in range(12):
        data = np.loadtxt(f'{i}.txt', dtype=np.float64, usecols=(1,2,3,4))
        f.write(f'if grid == {i}:\n')
        f.write(f'  grid_array = np.array([\n')
        for row in data:
            f.write(f'  [{row[0]}, {row[1]}, {row[2]}, {row[3]}],\n')
        f.write(f'  ])\n')
        f.write('\n')



