from PyQt6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QComboBox, QLabel, QFileDialog
from core.dataset_loader import load_csv_dataset
from core.model_trainer import train_keras_model, train_random_forest_model
from core.model_saver import save_model_to_file

class TrainerWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.dataset = None
        self.model = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        self.model_label = QLabel("Choose Algorithm:")
        layout.addWidget(self.model_label)

        self.model_combobox = QComboBox()
        self.model_combobox.addItems(["Keras", "Scikit-learn Random Forest"])
        layout.addWidget(self.model_combobox)

        self.load_button = QPushButton("Load Dataset")
        self.load_button.clicked.connect(self.load_dataset)
        layout.addWidget(self.load_button)

        self.train_button = QPushButton("Train Model")
        self.train_button.clicked.connect(self.train_model)
        layout.addWidget(self.train_button)

        self.save_button = QPushButton("Save Model")
        self.save_button.clicked.connect(self.save_model)
        layout.addWidget(self.save_button)

        self.setLayout(layout)
        self.setWindowTitle("ML Model Trainer")

    def load_dataset(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Dataset", "", "CSV Files (*.csv);;All Files (*)")
        if file_path:
            self.dataset = load_csv_dataset(file_path)
            print(f"Dataset loaded from {file_path}")

    def train_model(self):
        if self.dataset is None:
            print("Please load a dataset first.")
            return

        algorithm = self.model_combobox.currentText()
        if algorithm == "Keras":
            self.model = train_keras_model(self.dataset)
        elif algorithm == "Scikit-learn Random Forest":
            self.model = train_random_forest_model(self.dataset)

    def save_model(self):
        if self.model:
            file_path, _ = QFileDialog.getSaveFileName(self, "Save Model", "", "All Files (*)")
            if file_path:
                save_model_to_file(self.model, file_path)
