import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
# The following import configures Matplotlib for 3D plotting.
from mpl_toolkits.mplot3d import Axes3D
from scipy.special import sph_harm
#plt.rc('text', usetex=True)



def plot_Y(ax, sph_lin_comb: np.ndarray):
    """Plot the spherical harmonic of degree el and order m on Axes ax."""

    # Grids of polar and azimuthal angles
    theta = np.linspace(0, np.pi, 100)
    phi = np.linspace(0, 2*np.pi, 100)
    # Create a 2-D meshgrid of (theta, phi) angles.
    theta, phi = np.meshgrid(theta, phi)
    # Calculate the Cartesian coordinates of each point in the mesh.
    xyz = np.array([np.sin(theta) * np.sin(phi),
                    np.sin(theta) * np.cos(phi),
                    np.cos(theta)])
    
    Y_sum = np.zeros_like(phi)

    for coefficients in sph_lin_comb:

        # NB In SciPy's sph_harm function the azimuthal coordinate, theta,
        # comes before the polar coordinate, phi.
        Y = sph_harm(int(abs(coefficients[1])), int(coefficients[0]), phi, theta)

        # Linear combination of Y_l,m and Y_l,-m to create the real form.
        if m < 0:
            Y_sum += coefficients[2] * np.sqrt(2) * (-1)**m * Y.imag
        elif m > 0:
            Y_sum += coefficients[2] * np.sqrt(2) * (-1)**m * Y.real
    Yx, Yy, Yz = np.abs(Y_sum) * xyz

    # Colour the plotted surface according to the sign of Y.
    cmap = plt.cm.ScalarMappable(cmap=plt.get_cmap('PRGn'))
    cmap.set_clim(-0.5, 0.5)

    ax.plot_surface(Yx, Yy, Yz,
                    facecolors=cmap.to_rgba(Y_sum.real),
                    rstride=2, cstride=2)

    # Draw a set of x, y, z axes for reference.
    ax_lim = 7
    ax.plot([-ax_lim, ax_lim], [0,0], [0,0], c='0.5', lw=1, zorder=10)
    ax.plot([0,0], [-ax_lim, ax_lim], [0,0], c='0.5', lw=1, zorder=10)
    ax.plot([0,0], [0,0], [-ax_lim, ax_lim], c='0.5', lw=1, zorder=10)
    # Set the Axes limits and title, turn off the Axes frame.
    #ax.set_title(r'$Y_{{{},{}}}$'.format(el, m))
    ax_lim = 7
    ax.set_xlim(-ax_lim, ax_lim)
    ax.set_ylim(-ax_lim, ax_lim)
    ax.set_zlim(-ax_lim, ax_lim)
    ax.axis('off')

fig = plt.figure(figsize=plt.figaspect(1.))
ax = fig.add_subplot(projection='3d')
l, m = 16, 2
n = 1 #5.1478150705
sph = np.array([          [2,          -2, -0.26453899417567145],     
           [2,          -1,   1.1127903844066721E-003],
           [2,           0,  -8.1400556536813955],
           [2,           1,  -1.9113964390770812E-003],
           [2,           2,  0.14511680553334527]    ,
           [4,          -4,   1.1637454775672636E-003],
           [4,          -3,   4.6742698371375724E-005],
           [4,          -2,  -8.2337806808672131E-004],
           [4,          -1,  -7.5033294419675202E-005],
           [4,          0,  0.15409863852710773 ],    
           [4,          1,  -3.2094957407899496E-005],
           [4,           2,   3.9579493944930430E-004],
           [4,           3,  -9.2456201877817493E-005],
           [4,           4,   1.6053789000875070E-002],
           [6,          -6,   1.1721244014668160E-005],
           [6,          -5,   6.9111639127215382E-009],
           [6,          -4,  -7.0692308854051675E-005],
           [6,          -3,  -5.6550081844011115E-007],
           [6,          -2,  -3.8903923800249307E-005],
           [6,          -1,  -2.1800395591621855E-006],
           [6,           0,   3.8916335870857000E-004],
           [6,           1,   7.3014529920415559E-007],
           [6,           2,   2.0522438875492040E-005],
           [6,           3, -2.7555584451815448E-006],
           [6,           4,  -4.4596770444745190E-005],
           [6,          5,  1.6805238466743968E-006],
           [6,           6,  -4.5810098814287549E-004]])
plot_Y(ax, sph)
#plt.savefig('Y{}_{}.png'.format(l, m))
plt.show()