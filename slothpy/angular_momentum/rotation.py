# eventually incorporate it into class as classmethod probably
def rotate_vector_operator(x, y, z, rotation):

    #rotation = Rotation.matrix (from class)

    x_rot = rotation[0,0] * x + rotation[0,1] * y + rotation[0,2] * z
    y_rot = rotation[1,0] * x + rotation[1,1] * y + rotation[1,2] * z
    z_rot = rotation[2,0] * x + rotation[2,1] * y + rotation[2,2] * z

    return x_rot, y_rot, z_rot

class Rotation:
    pass

