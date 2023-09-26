from numpy import ndarray, allclose, identity, zeros_like


# eventually incorporate it into class as classmethod probably
def _rotate_vector_operator(vect_oper: ndarray, rotation: ndarray):
    # rotation = Rotation.matrix (from class)

    if rotation.shape != (3, 3):
        raise ValueError("Input rotation matrix must be a 3x3 matrix.")

    product = rotation.T @ rotation

    if not allclose(product, identity(3), atol=1e-2, rtol=0):
        raise ValueError("Input rotation matrix must be orthogonal.")

    rotated_oper = zeros_like(vect_oper)

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
