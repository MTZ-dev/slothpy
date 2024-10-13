from sys import exit, argv
from signal import SIGINT, SIGTERM

from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QProgressBar, QLabel, QScrollArea, QFrame
from PyQt6.QtCore import QTimer, QDateTime, QTime
from PyQt6.QtGui import QFont
from numpy import array, sum
from multiprocessing.shared_memory import SharedMemory
import psutil

from slothpy._general_utilities._system import SltTemporarySignalHandler, exit_handler, _from_shared_memory, _distribute_chunks

class WorkerMonitorApp(QMainWindow):
    def __init__(self, progress_array_info, number_tasks_per_process, calling_function_name):
        super().__init__()
        self.progress_array_info = progress_array_info
        self.number_tasks_per_process = number_tasks_per_process
        self.overall_number_tasks = sum(number_tasks_per_process)
        self.calling_function_name = calling_function_name
        self.startTime = QTime.currentTime()
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
        self.overall_progress_bar.setMaximum(self.overall_number_tasks)
        main_layout.addWidget(self.overall_progress_bar)

        # Scroll Area for Individual Progress Bars
        scrollArea = QScrollArea(self)
        scrollArea.setWidgetResizable(True)
        scrollWidget = QWidget()
        scrollLayout = QVBoxLayout(scrollWidget)
        scrollArea.setWidget(scrollWidget)
        main_layout.addWidget(scrollArea)

        # Individual Progress Bars
        self.progress_bars = [QProgressBar() for _ in range(self.progress_array_info.shape[0])]
        for i, bar in enumerate(self.progress_bars):
            bar.setMaximum(self.number_tasks_per_process[i])
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
        sm_progress, progress_array = _from_shared_memory(self.progress_array_info)
        overall_progress = sum(progress_array)
        self.overall_progress_bar.setValue(overall_progress)
        for i, bar in enumerate(self.progress_bars):
            bar.setValue(progress_array[i])
        if overall_progress >= self.overall_number_tasks:
            self.timer.stop()
            QTimer.singleShot(1000, self.close)

    def updateSystemResources(self):
        self.cpu_label.setText(f"CPU Usage: {psutil.cpu_percent()}%")
        self.memory_label.setText(f"Memory Usage: {psutil.virtual_memory().percent}%")

    def updateElapsedTime(self):
        currentTime = QTime.currentTime()
        elapsed = self.startTime.secsTo(currentTime)
        hours, remainder = divmod(elapsed, 3600)
        minutes, seconds = divmod(remainder, 60)
        elapsedString = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        self.elapsedTimeLabel.setText(f"Elapsed Time: {elapsedString}")


def _run_monitor_gui(progress_array_info, number_to_parallelize, number_processes, calling_function_name):

    with SltTemporarySignalHandler([SIGTERM, SIGINT], exit_handler):
        number_tasks_per_process = [(chunk.end - chunk.start) for chunk in _distribute_chunks(number_to_parallelize, number_processes)]
        app = QApplication(argv)
        main_window = WorkerMonitorApp(progress_array_info, number_tasks_per_process, calling_function_name)
        main_window.show()
        exit(app.exec())
