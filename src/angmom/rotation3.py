import numpy as np
from numba import jit

class Rotation:

    @staticmethod
    @jit(nopython=True, cache=False)
    def from_matrix_to_angles(matrix: "np.ndarray") -> "np.ndarray":
        if (not np.allclose(matrix.T@matrix, np.identity(3), rtol=1e-05, atol=1e-08, equal_nan=False)) or (not np.isclose(np.linalg.det(matrix), 1, rtol=1e-05)):
            raise ValueError("Rotation matrix needs to be orthogonal.")
        return np.array([np.arctan2(matrix[1][2], matrix[0][2]), np.arccos(matrix[2][2]), np.arctan2(-matrix[2][1],matrix[2][0])])

    @staticmethod
    @jit(nopython=True, cache=False)
    def from_matrix_to_axis_angle(matrix: "np.ndarray") -> "np.ndarray":
        if (not np.allclose(matrix.T@matrix, np.identity(3), rtol=1e-05, atol=1e-08, equal_nan=False)) or (not np.isclose(np.linalg.det(matrix), 1, rtol=1e-05)):
            raise ValueError("Rotation matrix needs to be orthogonal.")
        eigenvalues, eigenvectors = np.linalg.eig(matrix.astype(np.complex128))
        index = np.isclose(eigenvalues.real.copy().astype(np.float64), np.float64(1.0), rtol=1e-05)
        axis_angle = np.empty(4, dtype=np.float64)
        axis_angle[:3] = np.reshape(np.ascontiguousarray(eigenvectors[:, index].real),3)
        beta_sph = np.arccos(axis_angle[2])
        axis_angle[3] = np.array(np.arccos(np.divide(matrix[2][2] - np.power(np.cos(beta_sph),2), np.power(np.sin(beta_sph),2))))
        return axis_angle
    
    @staticmethod
    @jit(nopython=True, cache=False)
    def from_angles_to_matrix(angles: "np.ndarray") -> "np.ndarray":
        return np.array([[-np.sin(angles[0])*np.sin(angles[2]) + np.cos(angles[0])*np.cos(angles[1])*np.cos(angles[2]), -np.sin(angles[0])*np.cos(angles[2]) - np.sin(angles[2])*np.cos(angles[0])*np.cos(angles[1]), np.sin(angles[1])*np.cos(angles[0])], [np.sin(angles[0])*np.cos(angles[1])*np.cos(angles[2]) + np.sin(angles[2])*np.cos(angles[0]), -np.sin(angles[0])*np.sin(angles[2])*np.cos(angles[1]) + np.cos(angles[0])*np.cos(angles[2]), np.sin(angles[0])*np.sin(angles[1])], [-np.sin(angles[1])*np.cos(angles[2]), np.sin(angles[1])*np.sin(angles[2]), np.cos(angles[1])]])

    @staticmethod
    @jit(nopython=True, cache=False)
    def from_angles_to_axis_angle(angles: "np.ndarray") -> "np.ndarray": #jak sie te tuple oznaczalo
        return Rotation.from_matrix_to_axis_angle(Rotation.from_angles_to_matrix(angles))

    @staticmethod
    @jit(nopython=True, cache=False)
    def from_axis_angle_to_matrix(axis_angle: "np.NDArray") -> "np.NDArray":
        axis_angle[:3] = axis_angle[:3] / np.linalg.norm(axis_angle[:3]) 
        alpha_sph, beta_sph = np.arctan2(axis_angle[1], axis_angle[0]), np.arccos(axis_angle[2])
        matrix_sph = np.array([[np.cos(alpha_sph)*np.cos(beta_sph), -np.sin(alpha_sph), np.sin(beta_sph)*np.cos(alpha_sph)], [np.sin(alpha_sph)*np.cos(beta_sph), np.cos(alpha_sph), np.sin(alpha_sph)*np.sin(beta_sph)], [-np.sin(beta_sph), 0, np.cos(beta_sph)]])
        matrix = matrix_sph@np.array([[np.cos(axis_angle[3]), -np.sin(axis_angle[3]), 0],[np.sin(axis_angle[3]), np.cos(axis_angle[3]), 0], [0, 0, 1]])@matrix_sph.T
        return matrix.T

    @staticmethod
    @jit(nopython=True, cache=False)
    def from_axis_angle_to_angles(axis_angle: "np.ndarray") -> "np.ndarray":
        return Rotation.from_matrix_to_angles(Rotation.from_axis_angle_to_matrix(axis_angle))

    @classmethod
    def from_angles(cls, angles):
        return cls._new(matrix=None, angles = angles)

    @classmethod
    def from_matrix(cls, matrix):
        return cls._new(matrix=matrix)
    
    @classmethod
    def from_axis_angle(cls, axis_angle):
        return cls._new(axis_angle=axis_angle)

    @classmethod
    def _new(cls, matrix=None, axis_angle=None, angles=None):

        obj = super().__new__(cls)

        if matrix is not None:
            obj._matrix = matrix
            obj._angles = Rotation.from_matrix_to_angles(matrix)
            obj._axis_angle = Rotation.from_matrix_to_axis_angle(matrix)
            return obj
        elif axis_angle is not None:
            matrix = Rotation.from_axis_angle_to_matrix(axis_angle)
            obj._matrix = matrix
            obj._angles = Rotation.from_matrix_to_angles(matrix)
            obj._axis_angle = axis_angle
            return obj
        else:
            matrix = Rotation.from_angles_to_matrix(angles)
            obj._matrix = matrix
            obj._angles = angles
            obj._axis_angle = Rotation.from_matrix_to_axis_angle(matrix)
            return obj

    def __new__(cls, *args, **kwargs):
        raise TypeError("The Rotation object should not be instantiated directly. Use a Rotation creation function, such as from_angles(), from_matrix() or from_axis_angle.")

    
    @property
    def angles(self):
        return self._angles 
    
    @property
    def matrix(self):
        return self._matrix
    
    @property
    def axis_angle(self):
        return self._axis_angle    


if __name__ == "__main__":

    import timeit

    #test
    
    def test():

        test = Rotation.from_angles(np.array([1,3.0,1]))

        a = test.angles
        b = test.matrix
        c = test.axis_angle

        #print(test.angles)
        #print(test.matrix)
        #print(test.axis_angle)
        
        test2 = Rotation.from_matrix(test.matrix)

        a = test2.angles
        b = test2.matrix
        c = test2.axis_angle

        #print(test2.angles)
        #print(test2.matrix)
        #print(test2.axis_angle)

        test3 = Rotation.from_axis_angle(test.axis_angle)

        a = test3.angles
        b = test3.matrix
        c = test3.axis_angle

        #print(test3.angles)
        #print(test3.matrix)
        #print(test3.axis_angle)

    test()

    execution_time = timeit.repeat(test, repeat=5, number=1000)

    print(f"Execution time: {execution_time} seconds")
    


   














#to do definicji rotacji j mj za pomoca potem moze i nawet macierzy D ale z parametrów macierzy rotacji klasycznych między dwoma klasami
#mozna przemyslec czty chcesz wszystkie inne klasy dziedziczyć po jednej super klasie angmom i wtedy odwoływać sie do jej konstrukora
class ClassA:
    def __init__(self, value):
        self.value = value
    
    def __mul__(self, other):
        if isinstance(other, ClassB):
            return self.value * other.value
        else:
            raise TypeError("Unsupported operand type: {} and {}".format(type(self).__name__, type(other).__name__))


class ClassB:
    def __init__(self, value):
        self.value = value

a = ClassA(5)
b = ClassB(10)

#result = a * b
#print(result)  
