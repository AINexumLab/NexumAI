from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QSlider, QApplication, QGraphicsDropShadowEffect, QFrame
from PyQt6.QtCore import Qt, QSettings
from PyQt6.QtGui import QPainter, QColor, QFont

class SettingsWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Settings")
        layout = QVBoxLayout()

        label = QLabel("Font Size")
        layout.addWidget(label)

        # Create the slider
        self.font_slider = QSlider(Qt.Orientation.Horizontal)
        self.font_slider.setRange(8, 13)
        current_size = QApplication.instance().font().pointSize()
        self.font_slider.setValue(current_size)
        self.font_slider.valueChanged.connect(self.change_font_size)

        slider_frame = QFrame()
        slider_layout = QVBoxLayout(slider_frame)
        slider_layout.setContentsMargins(0, 0, 0, 0)
        slider_layout.addWidget(self.font_slider)

        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(20)
        shadow.setColor(QColor(0, 0, 0, 160))
        shadow.setOffset(0, 0)

        slider_frame.setGraphicsEffect(shadow)

        layout.addWidget(slider_frame)

        self.setLayout(layout)

    def change_font_size(self, size: int):
        font = QFont()
        font.setPointSize(size)
        QApplication.instance().setFont(font)

        settings = QSettings("YourCompany", "YourApp")
        settings.setValue("fontSize", size)
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(0, 0, 0, 0))
        color = QColor(40, 40, 80, int(0.6 * 255))
        painter.fillRect(self.rect(), color)