import os
from sys import argv

from PyQt6.QtWidgets import QCheckBox, QDialog, QSpinBox, QDialogButtonBox, QLabel, QMessageBox, QWidget, QVBoxLayout, \
    QApplication, QMainWindow, QFileDialog, QHBoxLayout, QGridLayout
from PyQt6.QtGui import QAction, QCloseEvent, QIcon, QFont
from PyQt6.QtCore import QSize, Qt

import matplotlib as mpl  # import matplotlib after PyQt6
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.backends.backend_qtagg import (
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.pyplot import close

from multiprocessing import current_process
from pkg_resources import resource_filename

from slothpy._general_utilities._system import _is_notebook

mpl.use("QtAgg")


class CustomDialog(QDialog):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.setWindowTitle('Custom File Dialog')
        self.setFixedSize(QSize(200, 125))

        # Get dpi
        self.int_box = QSpinBox(self)
        self.int_box.setRange(0, 100000)
        self.int_box.setValue(300)

        # Create a checkbox
        self.checkbox = QCheckBox("Transparency", self)

        # Dialog buttons
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

        # Layout for the custom dialog
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel('Enter resolution (dpi):'))
        layout.addWidget(self.int_box)
        layout.addWidget(self.checkbox)
        layout.addWidget(self.buttons)

        self.setLayout(layout)

    def dialog_state(self):
        return [self.checkbox.isChecked(), self.int_box.value()]


class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, fig, ax, parent=None):
        self.axes = ax
        super().__init__(fig)


class CustomNavigationToolbar(NavigationToolbar):
    def __init__(self, canvas, parent):
        super().__init__(canvas, parent)

    def save_figure(self, *args):
        filetypes = self.canvas.get_supported_filetypes_grouped()
        sorted_filetypes = sorted(filetypes.items())
        default_filetype = self.canvas.get_default_filetype()

        startpath = os.path.expanduser(mpl.rcParams['savefig.directory'])
        start = os.path.join(startpath, self.canvas.get_default_filename())
        filters = []
        selectedFilter = None
        for name, exts in sorted_filetypes:
            exts_list = " ".join(['*.%s' % ext for ext in exts])
            filter = f'{name} ({exts_list})'
            if default_filetype in exts:
                selectedFilter = filter
            filters.append(filter)
        filters = ';;'.join(filters)

        custom_dialog = CustomDialog(self)
        if custom_dialog.exec() == QDialog.DialogCode.Accepted:
            ftransp, fdpi = custom_dialog.dialog_state()
            fname, filter = QFileDialog.getSaveFileName(
                self.canvas.parent(), "Choose a filename to save to", start,
                filters, selectedFilter)
            if fname:
                # Save dir for next time, unless empty str (i.e., use cwd).
                if startpath != "":
                    mpl.rcParams['savefig.directory'] = os.path.dirname(fname)
                try:
                    self.canvas.figure.savefig(fname, transparent=ftransp, dpi=fdpi)
                except Exception as e:
                    QMessageBox.critical(
                        self, "Error saving file", str(e),
                        QMessageBox.StandardButton.Ok,
                        QMessageBox.StandardButton.NoButton)


#TODO: Ribbon as a grid, for energy plot current functionalities to be implemented: removal of energy values, setting number of states shown, 
# customization of appearance.


class PlotView(QMainWindow, ):
    def __init__(self, data, plot_type, onclose=None):
        super().__init__()
        self.setWindowTitle("SlothPy")
        self.setObjectName("MainWindow")
        self.onClose = onclose
        self.ribbon = None
        self.data = data
        self.fig = None
        
        if plot_type == 'energy_levels':
            self.cutoff = 0
            self.energy_unit = 'wavenumber'
            self.energy_levels()

    def closeEvent(self, a0: QCloseEvent | None) -> None:
        if self.onClose is not None:
            self.onClose()
        close(self.fig)
        return super().closeEvent(a0)

    def set_fig(self):
        # Close previous figure to free up memory
        if self.fig is not None:
            close(self.fig)
        
        fig = self.fig
        ax = self.ax

        mpl_canvas = MplCanvas(fig, ax, self)

        # Create toolbar, passing canvas as first parameter, parent (self, the MainWindow) as second.
        toolbar = CustomNavigationToolbar(mpl_canvas, self)

        layout_main = QVBoxLayout()
        layout_main.addWidget(toolbar)
        layout_ribbons = QHBoxLayout()
        self.ribbon.setParent(None)
        layout_ribbons.addLayout(self.ribbon)
        layout_ribbons.addWidget(mpl_canvas)
        layout_main.addLayout(layout_ribbons)

        # Create a placeholder widget to hold our toolbar and canvas.
        widget = QWidget()
        widget.setLayout(layout_main)
        self.setCentralWidget(widget)

    def energy_levels(self):

        # Create energy levels plot
        from slothpy._general_utilities._plot import _plot_energy_levels
        fig, ax = _plot_energy_levels(self.data, self.cutoff, self.energy_unit)

        # Create ribbon suitable for plot type
        ribbon_layout = QGridLayout()
        ribbon_layout.addWidget(QLabel('Number of states:'), 0, 0, alignment=Qt.AlignmentFlag.AlignCenter)
        num_states_widged = QSpinBox(minimum=1, maximum=len(self.data))
        num_states_widged.valueChanged.connect(self.energy_levels_cutoff)
        ribbon_layout.addWidget(num_states_widged, 0, 1, alignment=Qt.AlignmentFlag.AlignCenter)

        # Store changes
        self.fig = fig
        self.ax = ax
        self.ribbon = ribbon_layout

    def energy_levels_cutoff(self, cutoff):
        self.cutoff = cutoff

        from slothpy._general_utilities._plot import _plot_energy_levels
        fig, ax = _plot_energy_levels(self.data, self.cutoff, self.energy_unit)
        self.fig = fig
        self.ax = ax

        self.set_fig()


class SlothGui(QApplication):
    def __init__(
        self, sys_argv: list[str], data = None, plot_type: str = None , onClose: callable = None
    ):
        super(SlothGui, self).__init__(sys_argv)
        self.plot_view = PlotView(data, plot_type, onclose=onClose)
        image_path = resource_filename("slothpy", "static/slothpy_3.ico")
        app_icon = QIcon(image_path)
        SlothGui.setWindowIcon(app_icon)
        SlothGui.setFont(QFont("Helvetica", 12))

    def show(self):
        self.plot_view.set_fig()
        self.plot_view.show()


# If we are in Jupyter notebook prepare blank gui page
if current_process().name == "MainProcess":
    if _is_notebook():
        app = SlothGui(sys_argv=argv)
        app.setFont(QFont("Helvetica", 12))


def _display_plot(data, plot_type, onClose: callable = None):
    if current_process().name == "MainProcess":
        global app
        if _is_notebook():
            app.show()
        else:
            tmp_app = SlothGui(sys_argv=argv, data=data, plot_type=plot_type)
            tmp_app.setFont(QFont("Helvetica", 12))
            tmp_app.plot_view.set_fig()
            tmp_app.plot_view.show()
            tmp_app.exec()




