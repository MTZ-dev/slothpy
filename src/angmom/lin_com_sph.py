import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.special import sph_harm

def plot_linear_combination(coefficients, degrees, orders):
    # Generate theta and phi values
    theta = np.linspace(0, np.pi, 100)
    phi = np.linspace(0, 2 * np.pi, 200)
    theta, phi = np.meshgrid(theta, phi)

    # Initialize the linear combination
    lin_comb = np.zeros_like(theta).astype(np.complex128)

    # Compute the linear combination of spherical harmonics
    for coeff, l, m in zip(coefficients, degrees, orders):
        lin_comb += coeff * sph_harm(m, l, theta, phi)
    
    # Take abs
    lin_comb = np.abs(lin_comb)

    # Convert spherical coordinates to Cartesian coordinates
    x = lin_comb * np.sin(theta) * np.cos(phi)
    y = lin_comb * np.sin(theta) * np.sin(phi)
    z = lin_comb * np.cos(theta)

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the wireframe of the linear combination
    ax.plot_wireframe(x, y, z, color='b')

    # Set plot title and labels
    ax.set_title('Linear Combination of Spherical Harmonics')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Set plot limits
    ax.set_xlim([-0.6, 0.6])
    ax.set_ylim([-0.6,0.6])
    ax.set_zlim([-0.6, 0.6])

    # Show the plot
    plt.show()

# Example usage: Plot a linear combination of spherical harmonics
coefficients = [1,-1]  # Coefficients of the linear combination
degrees = [1,1]  # Degrees of the spherical harmonics
orders = [-1,1]  # Orders of the spherical harmonics

plot_linear_combination(coefficients, degrees, orders)
