from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QProgressBar, QLabel, QScrollArea, QFrame
from PyQt6.QtCore import QTimer, QDateTime, QTime
from PyQt6.QtGui import QFont
import sys
import numpy as np
from multiprocessing.shared_memory import SharedMemory
from numpy import ndarray, int64
import psutil

class WorkerMonitorApp(QMainWindow):
    def __init__(self, progress_array_name, progress_array_shape, number_tasks, calling_function_name):
        super().__init__()
        self.progress_array_name = progress_array_name
        self.progress_array_shape = progress_array_shape
        self.number_tasks = number_tasks
        self.calling_function_name = calling_function_name
        self.startTime = QTime(0, 0, 0)
        self.initUI()
    
    def initUI(self):
        self.setWindowTitle('Worker Progress Monitor')
        self.setGeometry(100, 100, 400, 400)  # Adjust size as needed

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Elapsed Time and Function Name Labels
        self.elapsedTimeLabel = QLabel("Elapsed Time: 00:00:00")
        self.elapsedTimeLabel.setFont(QFont("Arial", 10))
        main_layout.addWidget(self.elapsedTimeLabel)

        self.functionNameLabel = QLabel(f"Monitoring: {self.calling_function_name}")
        self.functionNameLabel.setFont(QFont("Arial", 10))
        main_layout.addWidget(self.functionNameLabel)

        # Overall Progress Bar and Label
        self.overall_label = QLabel("Overall Progress")
        self.overall_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        main_layout.addWidget(self.overall_label)

        self.overall_progress_bar = QProgressBar()
        self.overall_progress_bar.setMaximum(sum(self.number_tasks))
        main_layout.addWidget(self.overall_progress_bar)

        # Scroll Area for Individual Progress Bars
        scrollArea = QScrollArea(self)
        scrollArea.setWidgetResizable(True)
        scrollWidget = QWidget()
        scrollLayout = QVBoxLayout(scrollWidget)
        scrollArea.setWidget(scrollWidget)
        main_layout.addWidget(scrollArea)

        # Individual Progress Bars
        self.progress_bars = [QProgressBar() for _ in range(self.progress_array_shape[0])]
        for i, bar in enumerate(self.progress_bars):
            bar.setMaximum(self.number_tasks[i])
            scrollLayout.addWidget(bar)

        # CPU and Memory Usage Labels
        self.cpu_label = QLabel("CPU Usage: 0%")
        self.cpu_label.setFont(QFont("Arial", 10))
        main_layout.addWidget(self.cpu_label)

        self.memory_label = QLabel("Memory Usage: 0%")
        self.memory_label.setFont(QFont("Arial", 10))
        main_layout.addWidget(self.memory_label)

        # Timer for Updates and Elapsed Time
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.updateProgress)
        self.timer.timeout.connect(self.updateSystemResources)
        self.timer.start(190)

        # Separate timer for updating elapsed time every second
        self.elapsedTimeTimer = QTimer(self)
        self.elapsedTimeTimer.timeout.connect(self.updateElapsedTime)
        self.elapsedTimeTimer.start(1000)

    def updateProgress(self):
        progress_shared = SharedMemory(self.progress_array_name)
        progress_array = ndarray(self.progress_array_shape, dtype=int64, buffer=progress_shared.buf)
        overall_progress = np.sum(progress_array)
        self.overall_progress_bar.setValue(overall_progress)
        for i, bar in enumerate(self.progress_bars):
            bar.setValue(progress_array[i])
        if overall_progress >= np.sum(self.number_tasks):
            self.timer.stop()
            QTimer.singleShot(1000, self.close)

    def updateSystemResources(self):
        self.cpu_label.setText(f"CPU Usage: {psutil.cpu_percent()}%")
        self.memory_label.setText(f"Memory Usage: {psutil.virtual_memory().percent}%")

    def updateElapsedTime(self):
        # elapsed = self.startDateTime.secsTo(QDateTime.currentDateTime())
        # Format elapsed time as H:M:S
        # elapsedString = str(int(elapsed / 3600)).zfill(2) + ":" + str(int((elapsed % 3600) / 60)).zfill(2) + ":" + str(elapsed % 60).zfill(2)
        # self.elapsedTimeLabel.setText(f"Elapsed Time: {elapsedString}")
        pass

def run_gui(progress_array_name, progress_array_shape, number_tasks, calling_function_name):
    app = QApplication(sys.argv)
    main_window = WorkerMonitorApp(progress_array_name, progress_array_shape, number_tasks, calling_function_name)
    main_window.show()
    sys.exit(app.exec())
