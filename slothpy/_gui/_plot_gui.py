import os
from sys import argv

from PyQt6.QtWidgets import QCheckBox, QDialog, QSpinBox, QDialogButtonBox, QLabel, QMessageBox, QWidget, QVBoxLayout, \
    QApplication, QMainWindow, QFileDialog, QHBoxLayout, QGridLayout, QAbstractSpinBox, QComboBox
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
        self.canvas = None
        
        if plot_type == 'states_energy_cm_1':
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
        fig.set_dpi(100)
        fig.set_size_inches(6, 4.8)
        fig.tight_layout()

        if self.canvas is None:
            mpl_canvas = MplCanvas(fig, ax, self)
            self.canvas = mpl_canvas

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
        else:
            self.canvas.figure = fig
            self.canvas.axes = ax
            self.canvas.draw()
            #TODO: Better resize, figure should have consistent size

    def energy_levels(self):

        # Create energy levels plot
        from slothpy._general_utilities._plot import _plot_energy_levels
        fig, ax = _plot_energy_levels(self.data, self.cutoff, self.energy_unit)

        # Create ribbon suitable for plot type
        ribbon_layout = QVBoxLayout()
        ribbon_layout.setSpacing(0)
        ribbon_layout.setContentsMargins(0, 0, 0, 0)

        # Widget that changes how many states are visible
        number_states_layout = QHBoxLayout()
        number_states_text = QLabel('Number of states:')
        number_states_text.setFont(QFont('Helvetica', 12))
        number_states_layout.addWidget(number_states_text, alignment=Qt.AlignmentFlag.AlignCenter)
        number_states_widget = QSpinBox(minimum=1, maximum=len(self.data), value=len(self.data))
        number_states_widget.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        number_states_widget.setMinimumWidth(55)
        number_states_widget.setKeyboardTracking(False)
        number_states_widget.valueChanged.connect(self.energy_levels_cutoff)
        number_states_layout.addWidget(number_states_widget, alignment=Qt.AlignmentFlag.AlignCenter)
        ribbon_layout.addLayout(number_states_layout)

        # Widget that changes energy unit
        energy_unit_layout = QHBoxLayout()
        energy_unit_text = QLabel('Energy unit:')
        energy_unit_text.setFont(QFont('Helvetica', 12))
        energy_unit_layout.addWidget(energy_unit_text, alignment=Qt.AlignmentFlag.AlignCenter)
        energy_unit_widget = QComboBox()
        energy_unit_widget.addItems(['Wavenumber', r'Kj/mol', 'Hartree', 'eV', r'Kcal/mol'])
        energy_unit_widget.setMouseTracking(False)
        energy_unit_widget.currentTextChanged.connect(self.energy_levels_unit)
        energy_unit_layout.addWidget(energy_unit_widget, alignment=Qt.AlignmentFlag.AlignCenter)
        ribbon_layout.addLayout(energy_unit_layout)

        # Store changes
        self.fig = fig
        self.ax = ax
        self.ribbon = ribbon_layout

    def energy_levels_cutoff(self, cutoff):
        self.cutoff = cutoff
        self.update_energy_levels()

    def energy_levels_unit(self, unit):
        self.energy_unit = unit.lower().replace(r'/', '_')
        self.update_energy_levels()

    def update_energy_levels(self):
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




