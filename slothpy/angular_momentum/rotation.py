import numpy as np


# eventually incorporate it into class as classmethod probably
def _rotate_vector_operator(vect_oper: np.ndarray, rotation: np.ndarray):
    # rotation = Rotation.matrix (from class)

    rotated_oper = np.zeros_like(vect_oper)

    rotated_oper[0] = (
        rotation[0, 0] * vect_oper[0]
        + rotation[0, 1] * vect_oper[1]
        + rotation[0, 2] * vect_oper[2]
    )
    rotated_oper[1] = (
        rotation[1, 0] * vect_oper[0]
        + rotation[1, 1] * vect_oper[1]
        + rotation[1, 2] * vect_oper[2]
    )
    rotated_oper[2] = (
        rotation[2, 0] * vect_oper[0]
        + rotation[2, 1] * vect_oper[1]
        + rotation[2, 2] * vect_oper[2]
    )

    return rotated_oper


class Rotation:
    pass
