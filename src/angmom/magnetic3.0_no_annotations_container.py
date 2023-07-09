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



