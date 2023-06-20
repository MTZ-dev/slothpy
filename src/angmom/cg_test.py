import numpy as np
from numba import jit, njit
import timeit
#from scipy.special import binom
from sympy.physics.quantum.cg import CG

@jit('float64(float64, float64)', nopython=True, cache=True, nogil=True)
def binom(n, k):
    if k > n - k:
        k = n - k
    res = 1
    for i in range(k):
        res *= (n - i)
        res /= (i + 1)
    return res

#@jit('float64(float64, float64, float64, float64, float64, float64)', nopython=True, cache=True, nogil=True)
def Clebsh_Gordan(j1, m1, j2, m2, j3, m3):
    wp = np.float64(0.0)
    coeffCG = np.float64(0.0)
    fct = np.math.factorial
    
    if (m1 + m2 != m3) or (j1 < 0.0) or (j2 < 0.0) or (j3 < 0.0) or (np.abs(m1) > j1) or (np.abs(m2) > j2) or (np.abs(m3) > j3) or (np.abs(j1 - j2) > j3) or ((j1 + j2) < j3) or (np.abs(j2 - j3) > j1) or ((j2 + j3) < j1) or (np.abs(j3 - j1) > j2) or ((j3 + j1) < j2) or (np.mod(int(2.0 * j1), 2) != np.mod(int(2.0 * np.abs(m1)), 2)) or (np.mod(int(2.0 * j2), 2) != np.mod(int(2.0 * np.abs(m2)), 2)) or (np.mod(int(2.0 * j3), 2) != np.mod(int(2.0 * np.abs(m3)), 2)):
        return coeffCG
    
    u = np.float64(0.0)
    lb1 = int(min(j3 - j2 + m1, j3 - j1 - m2))
    lb2 = int(min(j1 + j2 - j3, j1 - m1, j2 + m2))
    
    if lb1 < 0:
        if -lb1 > lb2:
            return coeffCG
        else:
            for i in range(-lb1, lb2 + 1):
                u += np.power(-1, i) / (fct(i) * fct(int(j1 - m1 - i)) * fct(int(j2 + m2 - i)) * fct(int(j1 + j2 - j3 - i)) * fct(int(j3 - j2 + m1 + i)) * fct(int(j3 - j1 - m2 + i)))
    else:
        for i in range(lb2 + 1):
            u += np.power(-1, i) / (fct(i) * fct(int(j1 - m1 - i)) * fct(int(j2 + m2 - i)) * fct(int(j1 + j2 - j3 - i)) * fct(int(j3 - j2 + m1 + i)) * fct(int(j3 - j1 - m2 + i)))
    
    s1 = np.float64(0.0)
    s2 = np.float64(0.0)
    
    s1 = np.sqrt(float(fct(int(j1 + j2 - j3)) * fct(int(j1 - j2 + j3)) * fct(int(-j1 + j2 + j3)) / fct(int(j1 + j2 + j3 + 1)))) 
    s2 = np.sqrt(float(fct(int(j1 + m1)) * fct(int(j1 - m1)) * fct(int(j2 + m2)) * fct(int(j2 - m2)) * fct(int(j3 + m3)) * fct(int(j3 - m3)) * (2 * j3 + 1)))
    
    coeffCG = u * s1 * s2
    
    return coeffCG



@jit('float64(float64, float64, float64, float64, float64, float64)', nopython=True, cache=True, nogil=True)
def Clebsh_Gordan2(j1,m1,j2,m2,j3,m3):

    cg_coeff = 0

    if (m1 + m2 != m3) or (j1 < 0.0) or (j2 < 0.0) or (j3 < 0.0) or np.abs(m1) > j1 or np.abs(m2) > j2 or np.abs(m3) > j3 or (np.abs(j1 - j2) > j3) or ((j1 + j2) < j3) or (np.abs(j2 - j3) > j1) or ((j2 + j3) < j1) or (np.abs(j3 - j1) > j2) or ((j3 + j1) < j2) or (np.mod(int(2.0 * j1), 2) != np.mod(int(2.0 * np.abs(m1)), 2)) or (np.mod(int(2.0 * j2), 2) != np.mod(int(2.0 * np.abs(m2)), 2)) or (np.mod(int(2.0 * j3), 2) != np.mod(int(2.0 * np.abs(m3)), 2)):
        return cg_coeff

    J = j1 + j2 + j3
    C = np.sqrt(binom(2*j1,J-2*j2)*binom(2*j2,J-2*j3)/(binom(J+1,J-2*j3)*binom(2*j1,j1-m1)*binom(2*j2,j2-m2)*binom(2*j3,j3-m3)))
    z_min = np.max(np.array([0,j1-m1-J+2*j2,j2+m2-J+2*j1]))
    z_max = np.min(np.array([J-2*j3,j1-m1,j2+m2]))
    for z in range(z_min,z_max+1):
        cg_coeff  += (-1)**z * binom(J-2*j3,z) * binom(J-2*j2,j1-m1-z) * binom(J-2*j1,j2+m2-z)
    
    return cg_coeff * C

j1,m1,j2,m2,j3,m3 = 29/2,7/2,10,-5,27/2,-3/2 

repetitions = 10000

a = Clebsh_Gordan(j1,m1,j2,m2,j3,m3)
print(a)
b = Clebsh_Gordan2(j1,m1,j2,m2,j3,m3)
print(b)
c = CG(j1, m1, j2, m2, j3, m3).doit().evalf()
print(c)


def Clebsh_Gordan_wrapper():
    z = Clebsh_Gordan(j1,m1,j2,m2,j3,m3)


#Measure execution time
execution_times = timeit.repeat(stmt=Clebsh_Gordan_wrapper, repeat=5, number=repetitions)

print("Execution times Clebsh_Gordan:", str(np.array(execution_times)/repetitions), "seconds")


def Clebsh_Gordan2_wrapper():
    z = Clebsh_Gordan2(j1,m1,j2,m2,j3,m3)


#Measure execution time
execution_times = timeit.repeat(stmt=Clebsh_Gordan2_wrapper, repeat=5, number=repetitions)

print("Execution times Clebsh_Gordan2:", str(np.array(execution_times)/repetitions), "seconds")



def Clebsh_Gordan_sympy_wrapper():
    z = CG(j1, m1, j2, m2, j3, m3).doit().evalf()


#Measure execution time
execution_times = timeit.repeat(stmt=Clebsh_Gordan_sympy_wrapper, repeat=5, number=repetitions)

print("Execution times Clebsh_Gordan_sympy:", str(np.array(execution_times)/repetitions), "seconds")
