import sys
from PyQt6.QtWidgets import QApplication
from widgets.trainer_widget import TrainerWidget

def main():
    app = QApplication(sys.argv)
    window = TrainerWidget()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
