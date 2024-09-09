import os
from sys import argv

from PyQt6.QtWidgets import QCheckBox, QDialog, QSpinBox, QDialogButtonBox, QLabel, QMessageBox, QWidget, QVBoxLayout, \
    QApplication, QMainWindow, QFileDialog, QHBoxLayout, QGridLayout, QAbstractSpinBox, QComboBox, QTabWidget, QFormLayout, \
    QColorDialog
from PyQt6.QtGui import QAction, QCloseEvent, QIcon, QFont, QColor
from PyQt6.QtCore import QSize, Qt, pyqtSignal

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

class QColorComboBox(QComboBox):
    ''' A drop down menu for selecting colors '''

    # signal emitted if a color has been selected
    selectedColor = pyqtSignal(QColor)

    def __init__(self, parent = None, enableUserDefColors = True):
        ''' if the user shall not be able to define colors on its own, then set enableUserDefColors=False '''
        # init QComboBox
        super(QColorComboBox, self).__init__(parent)

        # enable the line edit to display the currently selected color
        self.setEditable(True)
        # read only so that there is no blinking cursor or sth editable
        self.lineEdit().setReadOnly(True)

        # text that shall be displayed for the option to pop up the QColorDialog for user defined colors
        self._userDefEntryText = 'Custom'
        # add the option for user defined colors
        if (enableUserDefColors):
            self.addItem(self._userDefEntryText)

        self._currentColor = None

        self.activated.connect(self._color_selected)
        
    # ------------------------------------------------------------------------
    def addColors(self, colors):
        ''' Adds colors to the QComboBox '''
        for a_color in colors:
            # if input is not a QColor, try to make it one
            if (not (isinstance(a_color, QColor))):
                a_color = QColor(a_color)
            # avoid dublicates
            if (self.findData(a_color) == -1):
                # add the new color and set the background color of that item
                self.addItem('', userData = a_color)
                self.setItemData(self.count()-1, QColor(a_color), Qt.ItemDataRole.BackgroundRole)
            
    # ------------------------------------------------------------------------
    def addColor(self, color):
        ''' Adds the color to the QComboBox '''
        self.addColors([color])

    # ------------------------------------------------------------------------
    def setColor(self, color):
        ''' Adds the color to the QComboBox and selects it'''
        self.addColor(color)
        self._color_selected(self.findData(color), False)

    # ------------------------------------------------------------------------
    def getCurrentColor(self):
        ''' Returns the currently selected QColor
            Returns None if non has been selected yet
        '''
        return self._currentColor

    # ------------------------------------------------------------------------
    def _color_selected(self, index, emitSignal = True):
        ''' Processes the selection of the QComboBox '''
        # if a color is selected, emit the selectedColor signal
        if (self.itemText(index) == ''):            
            self._currentColor = self.itemData(index)
            if (emitSignal):
                self.selectedColor.emit(self._currentColor)
                
        # if the user wants to define a custom color
        elif(self.itemText(index) == self._userDefEntryText):
            # get the user defined color
            new_color = QColorDialog.getColor(self._currentColor if self._currentColor else QColor.white)
            if (new_color.isValid()):
                # add the color to the QComboBox and emit the signal
                self.addColor(new_color)
                self._currentColor = new_color
                if (emitSignal):
                    self.selectedColor.emit(self._currentColor)
        
        # make sure that current color is displayed
        if (self._currentColor):
            self.setCurrentIndex(self.findData(self._currentColor))
            self.lineEdit().setStyleSheet("background-color: "+self._currentColor.name())

class PlotView(QMainWindow, ):
    def __init__(self, data, plot_type, onclose=None):
        super().__init__()
        self.setWindowTitle("SlothPy")
        self.setObjectName("MainWindow")
        self.plot_type = plot_type
        self.onClose = onclose
        self.ribbon = None
        self.data = data
        self.fig = None
        self.canvas = None
        
        if plot_type == 'states_energy_cm_1':
            self.cutoff = 0
            self.energy_unit = 'wavenumber'
            self.marker_size = 500
            self.marker_width = 2
            self.marker_color = '#000000'
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
        self.initialize_energy_levels()

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
        energy_unit_widget.setMinimumWidth(65)
        energy_unit_widget.currentTextChanged.connect(self.energy_levels_unit)
        energy_unit_layout.addWidget(energy_unit_widget, alignment=Qt.AlignmentFlag.AlignCenter)
        ribbon_layout.addLayout(energy_unit_layout)

        # Widget that allows customization of a plot
        tabs = QTabWidget()
        
        # Tab for customising the marker
        markers_tab = QWidget()
        markers_layout = QFormLayout()
        markers_tab.setLayout(markers_layout)

        marker_size_widget = QSpinBox(minimum=1, maximum=9999999, value=500)
        marker_size_widget.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        marker_size_widget.setKeyboardTracking(False)
        marker_size_widget.valueChanged.connect(self.marker_size_general)
        markers_layout.addRow(QLabel('Marker size:'), marker_size_widget)

        marker_width_widget = QSpinBox(minimum=0, maximum=9999999, value=2)
        marker_width_widget.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        marker_width_widget.setKeyboardTracking(False)
        marker_width_widget.valueChanged.connect(self.marker_width_general)
        markers_layout.addRow(QLabel('Marker width:'), marker_width_widget)

        marker_color_widget = QColorComboBox()
        marker_color_widget.setColor(self.marker_color)
        marker_color_widget.selectedColor.connect(self.marker_color_general)
        markers_layout.addRow(QLabel('Marker color:'), marker_color_widget)

        tabs.addTab(markers_tab, 'Markers')
        ribbon_layout.addWidget(tabs)

        # Store changes
        self.ribbon = ribbon_layout

    def marker_color_general(self, color):
        self.marker_color = color.name()
        if self.plot_type == 'states_energy_cm_1':
            self.update_energy_levels()

    def marker_size_general(self, value):
        self.marker_size = value
        if self.plot_type == 'states_energy_cm_1':
            self.update_energy_levels()

    def marker_width_general(self, value):
        self.marker_width = value
        if self.plot_type == 'states_energy_cm_1':
            self.update_energy_levels()

    def energy_levels_cutoff(self, cutoff):
        self.cutoff = cutoff
        self.update_energy_levels()

    def energy_levels_unit(self, unit):
        self.energy_unit = unit.lower().replace(r'/', '_')
        self.update_energy_levels()

    def initialize_energy_levels(self):
        from slothpy._general_utilities._plot import _plot_energy_levels
        fig, ax = _plot_energy_levels(array=self.data, cutoff=self.cutoff, energy_unit=self.energy_unit, marker_size=self.marker_size, marker_color=self.marker_color)
        self.fig = fig
        self.ax = ax

    def update_energy_levels(self):
        self.initialize_energy_levels()
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




