# SlothPy
# Copyright (C) 2023 Mikolaj Tadeusz Zychowicz (MTZ)

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
from sys import argv
from typing import Union, Literal
from multiprocessing import current_process
from pkg_resources import resource_filename
from numpy import linspace
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure
from matplotlib.pyplot import close
from PyQt6.QtWidgets import (
    QApplication,
    QWidget,
    QMainWindow,
    QVBoxLayout,
    QVBoxLayout,
    QFileDialog,
)
from PyQt6.QtGui import QIcon, QCloseEvent, QFont, QAction
from cycler import cycler

from slothpy._general_utilities._system import _is_notebook


class CustomNavigationToolbar(NavigationToolbar):
    def __init__(self, canvas, parent, fig):
        super(CustomNavigationToolbar, self).__init__(canvas, parent)

        # Add a custom save button to the toolbar
        self.addSeparator()
        self.save_custom_action = QAction("Save 600 DPI", self)
        self.save_custom_action.triggered.connect(self.save_custom_figure)
        self.addAction(self.save_custom_action)
        self.fig = fig

    def save_custom_figure(self):
        # Get the save path and filename using a file dialog
        save_path, _ = QFileDialog.getSaveFileName(
            self, "Save Figure", "", "PNG Files (*.png)"
        )
        if not save_path:
            return  # User canceled the operation

        if not save_path.lower().endswith(".png"):
            save_path += ".png"

        # Set the desired DPI
        dpi = 600

        # Save the figure with the specified parameters
        self.fig.savefig(save_path, transparent=True, dpi=dpi)
        print(f"Figure saved to {save_path} with DPI={dpi}")


class MainView(QMainWindow):
    def __init__(self, fig, onclose=None):
        super().__init__()
        self.setWindowTitle("SlothPy")
        self.setObjectName("MainWindow")
        self.onClose = onclose
        self.fig = fig

    def closeEvent(self, a0: QCloseEvent | None) -> None:
        if self.onClose is not None:
            self.onClose()
        close(self.fig)
        return super().closeEvent(a0)

    def set_fig(self, fig):
        mpl_canvas = FigureCanvasQTAgg(fig)
        toolbar = CustomNavigationToolbar(mpl_canvas, self, fig)
        layout = QVBoxLayout()
        layout.addWidget(toolbar)
        layout.addWidget(mpl_canvas)
        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)


class SlothGui(QApplication):
    def __init__(
        self, sys_argv: list[str], fig=None, onClose: callable = None
    ):
        super(SlothGui, self).__init__(sys_argv)
        self.main_view = MainView(fig, onclose=onClose)
        image_path = resource_filename("slothpy", "static/slothpy_3.ico")
        app_icon = QIcon(image_path)
        SlothGui.setWindowIcon(app_icon)
        SlothGui.setFont(QFont("Helvetica", 12))

    def show(self, fig):
        self.main_view.set_fig(fig)
        self.main_view.show()


# If we are in Jupyter notebook prepare blank gui page
if current_process().name == "MainProcess":
    if _is_notebook():
        app = SlothGui(sys_argv=argv)
        app.setFont(QFont("Helvetica", 12))


def _display_plot(fig: Figure = None, onClose: callable = None):
    if current_process().name == "MainProcess":
        global app
        if _is_notebook():
            app.show(fig)
        else:
            tmp_app = SlothGui(sys_argv=argv)
            tmp_app.setFont(QFont("Helvetica", 12))
            tmp_app.main_view.set_fig(fig)
            tmp_app.main_view.show()
            tmp_app.exec_()


def color_map(name: Union[str, list[str]]):
    """
    Creates matplotlib color map object.

    Parameters
    ----------
    name: Union["BuPi", "rainbow", "dark_rainbow", "light_rainbow",
        "light_rainbow_alt", "BuOr", "BuYl", "BuRd", "GnYl", "PrOr", "GnRd",
        "funmat", "NdCoN322bpdo", "NdCoNO222bpdo", "NdCoI22bpdo", "viridis",
        "plasma", "inferno", "magma", "cividis"] or list[str]

        One of the defined names for color maps: BuPi, rainbow, dark_rainbow,
        light_rainbow,light_rainbow_alt, BuOr, BuYl, BuRd, GnYl, PrOr, GnRd,
        funmat, NdCoN322bpdo, NdCoNO222bpdo, NdCoI22bpdo, viridis, plasma,
        inferno, magma, cividis or list of HTML color codes from which the
        color map will be created by interpolation of colors between ones on
        the list. The predefined names modifiers can be applied: _l loops
        the list in a way that it starts and ends with the same color, _r
        reverses the list.
    Returns
    -------
    matplotlib.colors.LinearSegmentedColormap
        The Matplotlib's color map object used for plotting.
    Raises
    ------
    ValueError
        If the input is not acceptable for creating a color map from the list
        of color codes or the name of predefined color map was incorrectly
        written.
    """
    cmap_list = []
    reverse = False
    loop = False
    if name[-2:] == "_l":
        name = name[:-2]
        loop = True

    if name[-2:] == "_r":
        reverse = True
        name = name[:-2]
    if type(name) == list:
        cmap_list = name
    elif name == "BuPi":
        cmap_list = [
            "#0091ad",
            "#1780a1",
            "#2e6f95",
            "#455e89",
            "#5c4d7d",
            "#723c70",
            "#a01a58",
            "#b7094c",
        ]
    elif name == "rainbow":
        cmap_list = [
            "#ff0000",
            "#ff8700",
            "#ffd300",
            "#deff0a",
            "#a1ff0a",
            "#0aff99",
            "#0aefff",
            "#147df5",
            "#580aff",
            "#be0aff",
        ]
    elif name == "dark_rainbow":
        cmap_list = [
            "#F94144",
            "#F3722C",
            "#F8961E",
            "#F9844A",
            "#F9C74F",
            "#90BE6D",
            "#43AA8B",
            "#4D908E",
            "#577590",
            "#277DA1",
        ]
    elif name == "light_rainbow":
        cmap_list = [
            "#FFADAD",
            "#FFD6A5",
            "#FDFFB6",
            "#CAFFBF",
            "#9BF6FF",
            "#A0C4FF",
            "#BDB2FF",
            "#FFC6FF",
        ]
    elif name == "light_rainbow_alt":
        cmap_list = [
            "#FBF8CC",
            "#FDE4CF",
            "#FFCFD2",
            "#F1C0E8",
            "#CFBAF0",
            "#A3C4F3",
            "#90DBF4",
            "#8EECF5",
            "#98F5E1",
            "#B9FBC0",
        ]
    elif name == "BuOr":
        cmap_list = [
            "#03045e",
            "#023e8a",
            "#0077b6",
            "#0096c7",
            "#00b4d8",
            "#ff9e00",
            "#ff9100",
            "#ff8500",
            "#ff6d00",
            "#ff5400",
        ]
    elif name == "BuRd":
        cmap_list = [
            "#033270",
            "#1368aa",
            "#4091c9",
            "#9dcee2",
            "#fedfd4",
            "#f29479",
            "#ef3c2d",
            "#cb1b16",
            "#65010c",
        ]
    elif name == "BuYl":
        cmap_list = [
            "#184e77",
            "#1e6091",
            "#1a759f",
            "#168aad",
            "#34a0a4",
            "#52b69a",
            "#76c893",
            "#99d98c",
            "#b5e48c",
            "#d9ed92",
        ]
    elif name == "GnYl":
        cmap_list = [
            "#007f5f",
            "#2b9348",
            "#55a630",
            "#80b918",
            "#aacc00",
            "#bfd200",
            "#d4d700",
            "#dddf00",
            "#eeef20",
            "#ffff3f",
        ]
    elif name == "PrOr":
        cmap_list = [
            "#240046",
            "#3c096c",
            "#5a189a",
            "#7b2cbf",
            "#9d4edd",
            "#ff9e00",
            "#ff9100",
            "#ff8500",
            "#ff7900",
            "#ff6d00",
        ]
    elif name == "GnRd":
        cmap_list = [
            "#005C00",
            "#2D661B",
            "#2A850E",
            "#27A300",
            "#A9FFA5",
            "#FFA5A5",
            "#FF0000",
            "#BA0C0C",
            "#751717",
            "#5C0000",
        ]
    elif name == "funmat":
        cmap_list = [
            "#1f6284",
            "#277ba5",
            "#2f94c6",
            "#49a6d4",
            "#6ab6dc",
            "#ffe570",
            "#ffe15c",
            "#ffda33",
            "#ffd20a",
            "#e0b700",
        ]
    elif name == "NdCoN322bpdo":
        cmap_list = [
            "#00268f",
            "#0046ff",
            "#009cf4",
            "#E5E4E2",
            "#ede76d",
            "#ffb900",
            "#b88700",
        ]
    elif name == "NdCoNO222bpdo":
        cmap_list = [
            "#A90F97",
            "#E114C9",
            "#f9bbf2",
            "#77f285",
            "#11BB25",
            "#0C831A",
        ]
    elif name == "NdCoI22bpdo":
        cmap_list = [
            "#075F5F",
            "#0B9898",
            "#0fd1d1",
            "#FAB3B3",
            "#d10f0f",
            "#720808",
        ]
    if cmap_list:
        if reverse:
            cmap_list.reverse()
        if loop:
            new_cmap_list = cmap_list.copy()
            for i in range(len(cmap_list)):
                new_cmap_list.append(cmap_list[-(i + 1)])
            cmap_list = new_cmap_list
        cmap = LinearSegmentedColormap.from_list("", cmap_list)
    elif name == "viridis":
        cmap = matplotlib.cm.viridis
        if reverse:
            cmap = matplotlib.cm.viridis_r
    elif name == "plasma":
        cmap = matplotlib.cm.plasma
        if reverse:
            cmap = matplotlib.cm.plasma_r
    elif name == "inferno":
        cmap = matplotlib.cm.inferno
        if reverse:
            cmap = matplotlib.cm.inferno_r
    elif name == "magma":
        cmap = matplotlib.cm.magma
        if reverse:
            cmap = matplotlib.cm.magma_r
    elif name == "cividis":
        cmap = matplotlib.cm.cividis
        if reverse:
            cmap = matplotlib.cm.cividis_r
    else:
        raise ValueError(
            f"""There is no such color map as {name} use one of those: BuPi, rainbow, dark_rainbow, light_rainbow, 
                light_rainbow_alt, BuOr, BuYl, BuRd, GnYl, PrOr, GnRd, funmat, NdCoN322bpdo, NdCoNO222bpdo,
                NdCoI22bpdo, viridis, plasma, inferno, magma, cividis or enter list of HTML color codes"""
        ) from None

    return cmap


def _custom_color_cycler(number_of_colors: int, cmap1: str, cmap2: str):
    """
    Creates a custom color cycler from two color maps in alternating pattern,
    suitable for use for matplotlib plots.

    Parameters
    ----------
    number_of_colors: int
        Number of colors in cycle.
    cmap1: str or list[str]
        Input of the color_map function.
    cmap2: str or list[str]
        Input of the color_map function.

    Returns
    -------
    cycler.cycler
        Cycler object created based on two input color maps.

    Raises
    ------
    ValueError
        If unable to use the given inputs. It should not be possible
        to trigger this error :).
    """
    try:
        if number_of_colors % 2 == 0:
            increment = 0
            lst1 = color_map(cmap1)(linspace(0, 1, int(number_of_colors / 2)))
            lst2 = color_map(cmap2)(linspace(0, 1, int(number_of_colors / 2)))
            color_cycler_list = []
            while increment < number_of_colors:
                if increment % 2 == 0:
                    color_cycler_list.append(lst1[int(increment / 2)])
                else:
                    color_cycler_list.append(lst2[int((increment - 1) / 2)])
                increment += 1
        else:
            increment = 0
            lst1 = color_map(cmap1)(
                linspace(0, 1, int((number_of_colors / 2) + 1))
            )
            lst2 = color_map(cmap2)(linspace(0, 1, int(number_of_colors / 2)))
            color_cycler_list = []
            while increment < number_of_colors:
                if increment % 2 == 0:
                    color_cycler_list.append(lst1[int(increment / 2)])
                else:
                    color_cycler_list.append(lst2[int((increment - 1) / 2)])
                increment += 1
        return cycler(color=color_cycler_list)
    except Exception as exc:
        raise SystemError(
            "Sloths are a Neotropical group of xenarthran mammals constituting"
            " the suborder Folivora, including the extant arboreal tree sloths"
            " and extinct terrestrial ground sloths. Noted for their slowness"
            " of movement, tree sloths spend most of their lives hanging"
            " upside down in the trees of the tropical rainforests of South"
            " America and Central America. Sloths are considered to be most"
            " closely related to anteaters, together making up the xenarthran"
            " order Pilosa. There are six extant sloth species in two genera –"
            " Bradypus (three–toed sloths) and Choloepus (two–toed sloths)."
            " Despite this traditional naming, all sloths have three toes on"
            " each rear limb, although two-toed sloths have only two digits on"
            " each forelimb. The two groups of sloths are from different,"
            " distantly related families, and are thought to have evolved"
            " their morphology via parallel evolution from terrestrial"
            " ancestors. Besides the extant species, many species of ground"
            " sloths ranging up to the size of elephants (like Megatherium)"
            " inhabited both North and South America during the Pleistocene"
            " Epoch. However, they became extinct during the Quaternary"
            " extinction event around 12,000 years ago, along with most"
            " large-bodied animals in the New World. The extinction correlates"
            " in time with the arrival of humans, but climate change has also"
            " been suggested to have contributed. Members of an endemic"
            " radiation of Caribbean sloths also formerly lived in the Greater"
            " Antilles but became extinct after humans settled the archipelago"
            " in the mid-Holocene, around 6,000 years ago. Sloths are so named"
            " because of their very low metabolism and deliberate movements."
            " Sloth, related to slow, literally means laziness, and their"
            " common names in several other languages (e.g. French: paresseux,"
            " Spanish: perezoso) also mean lazy or similar. Their slowness"
            " permits their low-energy diet of leaves and avoids detection by"
            " predatory hawks and cats that hunt by sight. Sloths are almost"
            " helpless on the ground but are able to swim. The shaggy coat has"
            " grooved hair that is host to symbiotic green algae which"
            " camouflage the animal in the trees and provide it nutrients. The"
            " algae also nourish sloth moths, some species of which exist"
            " solely on sloths."
            + "Source: https://en.wikipedia.org/wiki/Sloth"
        )
    

def _plot_zeeman_splitting(zeeman_array, magnetic_fields, orientations, ldfkgjdflgkdjf, dklfjgdfkgj):
    pass

def energy_units(unit: Literal['kj/mol', 'eh', 'hartree', 'au', 'ev', 'kcal/mol', 'wavenumber']) -> tuple[float, str]:
    """
    Returns tuple of (float, str) which are conversion from cm^-1 to chosen unit
    and latex-type string of the name of this unit.

    Parameters
    ----------
    unit : str
    Name of the unit - one of the following: kj/mol, eh or hartree or au, ev, kcal/mol, wavenumber.

    Returns
    ----------
    Tuple[float, str] of conversion scalar and latex-type unit name
    """
    unit = unit.lower()
    unit_conversion = {'kj/mol': 0.0119627,
                       'eh': 0.000124,
                       'hartree': 0.000124,
                       'au': 0.000124,
                       'ev': 0.000123,
                       'kcal/mol': 0.0028591,
                       'wavenumber': 1.0,
                       }
    unit_label = {'kjmol': 'kJ·mol$^{-1}$',
                        'eh': 'Hartree',
                        'hartree': 'Hartree',
                        'au': 'Hartree',
                        'ev': 'eV',
                        'kcal': 'kcal·mol$^{-1}$',
                        'wavenumber': 'cm$^{-1}$'
                        }
    return (unit_conversion[unit], unit_label[unit])