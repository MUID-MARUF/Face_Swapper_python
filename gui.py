from PyQt5.QtWidgets import (QWidget, QLabel, QPushButton, QVBoxLayout,
                             QFileDialog, QProgressBar, QApplication, QMessageBox, QSlider)
from PyQt5.QtGui import QPalette, QColor, QFont
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from logic import swap_faces
import sys
import cv2

# Worker thread
class SwapThread(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, source_path, target_path, model_path):
        super().__init__()
        self.source_path = source_path
        self.target_path = target_path
        self.model_path = model_path

    def run(self):
        try:
            result = swap_faces(self.source_path, self.target_path, self.model_path,
                                progress_callback=lambda val: self.progress.emit(val))
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))

class FaceSwapGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face Swap App")
        self.setFixedSize(500, 450)
        self.model_path = "inswapper_128.onnx"

        # Light pink background
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(255, 228, 240))
        self.setPalette(palette)

        # Font
        font = QFont("Arial", 10)
        self.setFont(font)

        # UI elements
        self.label_source = QLabel("Source Image: None")
        self.label_target = QLabel("Target Image: None")
        self.btn_source = QPushButton("Select Source Image")
        self.btn_target = QPushButton("Select Target Image")
        self.btn_swap = QPushButton("Swap Faces")
        self.progress = QProgressBar()

        # Slider for source strength
        self.label_strength = QLabel("Source Strength: 100%")
        self.slider_strength = QSlider(Qt.Horizontal)
        self.slider_strength.setMinimum(0)
        self.slider_strength.setMaximum(100)
        self.slider_strength.setValue(100)
        self.slider_strength.valueChanged.connect(self.update_strength_label)

        # Black text for readability
        self.label_source.setStyleSheet("color: black;")
        self.label_target.setStyleSheet("color: black;")
        self.btn_source.setStyleSheet("color: black;")
        self.btn_target.setStyleSheet("color: black;")
        self.btn_swap.setStyleSheet("color: black;")
        self.label_strength.setStyleSheet("color: black;")

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.label_source)
        layout.addWidget(self.btn_source)
        layout.addWidget(self.label_target)
        layout.addWidget(self.btn_target)
        layout.addWidget(self.label_strength)
        layout.addWidget(self.slider_strength)
        layout.addWidget(self.btn_swap)
        layout.addWidget(self.progress)
        self.setLayout(layout)

        # Connect buttons
        self.btn_source.clicked.connect(self.select_source)
        self.btn_target.clicked.connect(self.select_target)
        self.btn_swap.clicked.connect(self.start_swap)

        self.source_path = None
        self.target_path = None

    def update_strength_label(self):
        val = self.slider_strength.value()
        self.label_strength.setText(f"Source Strength: {val}%")

    def select_source(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Source Image")
        if path:
            self.source_path = path
            self.label_source.setText(f"Source Image: {path}")

    def select_target(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Target Image")
        if path:
            self.target_path = path
            self.label_target.setText(f"Target Image: {path}")

    def start_swap(self):
        if not self.source_path or not self.target_path:
            QMessageBox.warning(self, "Warning", "Please select both source and target images!")
            return

        self.btn_swap.setEnabled(False)
        self.progress.setValue(0)
        self.thread = SwapThread(self.source_path, self.target_path, self.model_path)
        self.thread.progress.connect(self.progress.setValue)
        self.thread.finished.connect(self.save_result)
        self.thread.error.connect(self.show_error)
        self.thread.start()

    def save_result(self, result):
        # Blend with target based on slider value
        alpha = self.slider_strength.value() / 100
        blended = cv2.addWeighted(result, alpha, cv2.imread(self.target_path), 1 - alpha, 0)

        self.btn_swap.setEnabled(True)
        save_path, _ = QFileDialog.getSaveFileName(self, "Save Swapped Image", "", "Images (*.jpg *.png)")
        if save_path:
            cv2.imwrite(save_path, blended)
            QMessageBox.information(self, "Success", f"Image saved to {save_path}")

    def show_error(self, message):
        QMessageBox.critical(self, "Error", message)
        self.btn_swap.setEnabled(True)
