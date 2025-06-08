from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QSlider
from PyQt6.QtCore import Qt, QSettings
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QApplication

class SettingsWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Settings")
        layout = QVBoxLayout()

        label = QLabel("Font Size")
        layout.addWidget(label)

        self.font_slider = QSlider(Qt.Orientation.Horizontal)
        self.font_slider.setRange(8, 13)
        current_size = QApplication.instance().font().pointSize()
        self.font_slider.setValue(current_size)
        self.font_slider.valueChanged.connect(self.change_font_size)
        layout.addWidget(self.font_slider)

        self.setLayout(layout)

    def change_font_size(self, size: int):
        font = QFont()
        font.setPointSize(size)
        QApplication.instance().setFont(font)

        settings = QSettings("YourCompany", "YourApp")
        settings.setValue("fontSize", size)