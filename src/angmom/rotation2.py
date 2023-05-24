import numpy as np

class Rotation:
    
    @staticmethod
    def from_matrix_to_angles():
        pass
    
    @staticmethod
    def from_matrix_to_axis_angle():
        pass
    
    @staticmethod
    def from_angles_to_matrix():
        pass

    @staticmethod
    def from_angles_to_axis_angle():
        pass

    @staticmethod
    def from_axis_angle_to_matrix():
        pass

    @staticmethod
    def from_axis_angle_to_angles():
        pass

    @classmethod
    def from_angles(cls, alpha, beta, gamma):
        return cls._new(matrix=None, alpha=alpha, beta=beta, gamma=gamma)

    @classmethod
    def from_matrix(cls, matrix):
        return cls._new(matrix=matrix)
    
    @classmethod
    def from_axis_angle(cls, axis, angle):
        return cls._new(axis=axis, angle=angle)

    @classmethod
    def _new(cls, matrix=None, axis=None, angle=None, alpha=None, beta=None, gamma=None):

        obj = super().__new__(cls)

        if matrix is not None:
            matrix = np.array(matrix)
            if (not np.all(np.isclose(matrix.T@matrix, np.identity(3), rtol=1e-05))) or (not np.isclose(np.linalg.det(matrix), 1, rtol=1e-05)):
                raise ValueError("Rotation matrix needs to be orthogonal.")
            alpha, beta, gamma = np.arctan2(matrix[1][2], matrix[0][2]), np.arccos(matrix[2][2]), np.arctan2(-matrix[2][1],matrix[2][0])
            eigenvalues, eigenvectors = np.linalg.eig(matrix)
            index = np.isclose(eigenvalues, 1.0, rtol=1e-05)
            eigenvector = np.reshape(eigenvectors[:, index].real,3)
            alpha_sph, beta_sph = np.arctan2(eigenvector[1], eigenvector[0]), np.arccos(eigenvector[2])
            angle = np.arccos(np.divide(matrix[2][2] - np.power(np.cos(beta_sph),2), np.power(np.sin(beta_sph),2)))
            obj._matrix = matrix
            obj._angles = np.array([alpha, beta, gamma])
            obj._axis_angle = (eigenvector, angle)
            return obj
        elif (axis is not None) and (angle is not None):
            axis = np.divide(np.array(axis),np.sqrt(np.power(axis[0], 2) + np.power(axis[1], 2) + np.power(axis[2], 2)))
            alpha_sph, beta_sph = np.arctan2(axis[1], axis[0]), np.arccos(axis[2])
            matrix_sph = np.array([[np.cos(alpha_sph)*np.cos(beta_sph), -np.sin(alpha_sph), np.sin(beta_sph)*np.cos(alpha_sph)], [np.sin(alpha_sph)*np.cos(beta_sph), np.cos(alpha_sph), np.sin(alpha_sph)*np.sin(beta_sph)], [-np.sin(beta_sph), 0, np.cos(beta_sph)]])
            matrix = matrix_sph@np.array([[np.cos(angle), -np.sin(angle), 0],[np.sin(angle), np.cos(angle), 0], [0, 0, 1]])@matrix_sph.T
            matrix = matrix.T
            alpha, beta, gamma = np.arctan2(matrix[1][2], matrix[0][2]), np.arccos(matrix[2][2]), np.arctan2(-matrix[2][1],matrix[2][0])
            obj._matrix = matrix
            obj._angles = np.array([alpha,beta,gamma])
            obj._axis_angle = (np.array([axis]), angle)
            return obj
        else:
            matrix = np.array([[-np.sin(alpha)*np.sin(gamma) + np.cos(alpha)*np.cos(beta)*np.cos(gamma), -np.sin(alpha)*np.cos(gamma) - np.sin(gamma)*np.cos(alpha)*np.cos(beta), np.sin(beta)*np.cos(alpha)], [np.sin(alpha)*np.cos(beta)*np.cos(gamma) + np.sin(gamma)*np.cos(alpha), -np.sin(alpha)*np.sin(gamma)*np.cos(beta) + np.cos(alpha)*np.cos(gamma), np.sin(alpha)*np.sin(beta)], [-np.sin(beta)*np.cos(gamma), np.sin(beta)*np.sin(gamma), np.cos(beta)]])
            eigenvalues, eigenvectors = np.linalg.eig(matrix)
            index = np.isclose(eigenvalues, 1.0, rtol=1e-05)
            eigenvector = np.reshape(eigenvectors[:, index].real,3)
            alpha_sph, beta_sph = np.arctan2(eigenvector[1], eigenvector[0]), np.arccos(eigenvector[2])
            angle = np.arccos(np.divide(matrix[2][2] - np.power(np.cos(beta_sph),2), np.power(np.sin(beta_sph),2)))
            obj._matrix = matrix
            obj._angles = np.array([alpha, beta, gamma])
            obj._axis_angle = (eigenvector, angle)
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
        test = Rotation.from_angles(1,3.14,1)

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

        test3 = Rotation.from_axis_angle(test.axis_angle[0], test.axis_angle[1])

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
