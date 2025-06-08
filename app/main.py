import sys
import os
from PyQt6.QtWidgets import QApplication
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from presentation.widgets.trainer_widget import TrainerWidget

def main():
    app = QApplication(sys.argv)
    window = TrainerWidget()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()