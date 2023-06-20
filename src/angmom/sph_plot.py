import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.special import sph_harm

def plot_spherical_harmonic(l, m):
    # Generate theta and phi values
    theta = np.linspace(0, 2 * np.pi, 100)
    phi = np.linspace(0, np.pi, 50)

    # Create a grid of theta and phi values
    theta, phi = np.meshgrid(theta, phi)

    # Compute the spherical harmonic values
    r = np.abs(sph_harm(m, l, theta, phi))

    # Convert spherical coordinates to Cartesian coordinates
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the wireframe of the spherical harmonic
    ax.plot_wireframe(x, y, z, rstride=5, cstride=5, color='b')

    # Set plot title and labels
    ax.set_title(f"Spherical Harmonic Y({l}, {m})")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Set plot limits
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])

    # Show the plot
    plt.show()

# Example usage: Plot the spherical harmonic Y(2, 1)
plot_spherical_harmonic(5, 1)
