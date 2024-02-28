import sys
import h5py
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QFileDialog, QTreeWidget, QTreeWidgetItem
from PyQt6.QtCore import Qt

class HDF5Viewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HDF5 Viewer")
        self.setGeometry(100, 100, 600, 400)
        self.currentFileName = ''  # To store the current HDF5 file path

        # Main layout
        layout = QVBoxLayout()

        # Button to load HDF5 file
        self.loadButton = QPushButton("Load HDF5 File")
        self.loadButton.clicked.connect(self.loadFile)
        layout.addWidget(self.loadButton)

        # Tree widget to display HDF5 file contents
        self.treeWidget = QTreeWidget()
        self.treeWidget.setHeaderLabels(["Name", "Type", "Attributes"])
        self.treeWidget.itemClicked.connect(self.onItemClicked)  # Connect item clicked signal
        layout.addWidget(self.treeWidget)

        # Setting the central widget
        centralWidget = QWidget()
        centralWidget.setLayout(layout)
        self.setCentralWidget(centralWidget)

        self.applyDarkTheme()

        
    def loadFile(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "Open HDF5 File", "", "HDF5 Files (*.hdf5);;All Files (*)")
        if fileName:
            self.currentFileName = fileName  # Save the current file name
            self.loadHDF5Contents(fileName)

    def loadHDF5Contents(self, fileName):
        self.treeWidget.clear()  # Clear existing items
        with h5py.File(fileName, 'r') as file:
            self.populateTreeWidget(file, self.treeWidget.invisibleRootItem())

    def populateTreeWidget(self, hdf5obj, parentItem):
        # Recursive function to populate tree with groups/datasets
        try:
            for name, item in hdf5obj.items():
                itemType = "Group" if isinstance(item, h5py.Group) else "Dataset"
                newItem = QTreeWidgetItem(parentItem, [name, itemType, str(dict(item.attrs))])
                newItem.setData(0, Qt.ItemDataRole.UserRole, (item.name, itemType))  # Store HDF5 path and type
                if isinstance(item, h5py.Group):
                    self.populateTreeWidget(item, newItem)
        except AttributeError:
            pass  # Handles the case where hdf5obj is not iterable

    def onItemClicked(self, item, column):
        # Retrieve stored HDF5 path and type
        hdf5Path, itemType = item.data(0, Qt.ItemDataRole.UserRole)
        print(f"Clicked on: {hdf5Path} (Type: {itemType})")  # Confirm click handling

        # Optionally, handle dataset/group specific actions here
        if itemType == 'Dataset':
            with h5py.File(self.currentFileName, 'r') as file:
                dataset = file[hdf5Path]
                print(f"Dataset shape: {dataset.shape}, Data type: {dataset.dtype}")

    def applyDarkTheme(self):
        self.setStyleSheet("""
        QWidget {
            color: #b1b1b1;
            background-color: #323232;
        }
        QTreeWidget, QTableWidget {
            border: none;
            background-color: #242424;
        }
        QTreeWidget::item:hover, QTableWidget::item:hover {
            background-color: #3d3d3d;
        }
        QTreeWidget::item:selected, QTableWidget::item:selected {
            background-color: #2a82da;
        }
        QPushButton {
            border: 2px solid #525252;
            border-radius: 5px;
            padding: 5px;
            background-color: #393939;
        }
        QPushButton:hover {
            border: 2px solid #2a82da;
        }
        QPushButton:pressed {
            background-color: #2a82da;
        }
        QDialog {
            background-color: #323232;
        }
        """)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = HDF5Viewer()
    viewer.show()
    sys.exit(app.exec())
