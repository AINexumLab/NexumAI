from PyQt6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QComboBox, QLabel, QFileDialog, QLayout
from core.dataset_loader import DatasetLoader
from core.model_trainer import train_keras_model, train_random_forest_model
from core.model_saver import save_model_to_file

class TrainerWidget(QWidget):
    model_types = [
        "Image Classification",
        "Image Segmentation",
        "Voice Classification"
    ]
    
    image_classification_algorithms = [
        "CNN (Recommended)",
        "SVM"
    ]
    
    image_segmentation_algorithms = [
        "U-Net (Recommended)",
        "CNN"
    ]
        
    def __init__(self):
        super().__init__()
        self.dataset = None
        self.model = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        
        self.model_label = QLabel("Choose Task Type:")
        layout.addWidget(self.model_label)

        self.model_combobox = QComboBox()
        self.model_combobox.addItem("Select a model...")
        self.model_combobox.addItems(self.model_types)
        self.model_combobox.model().item(0).setEnabled(False)
        self.model_combobox.currentIndexChanged.connect(self.update_algorithm_section)
        self.model_combobox.currentIndexChanged.connect(self.style_combobox_on_selection)
        layout.addWidget(self.model_combobox)
        
        self.algorithm_label = QLabel("Choose Algorithm:")
        self.algorithm_label.setVisible(False)
        layout.addWidget(self.algorithm_label)
        
        self.algorithm_combobox = QComboBox()
        self.algorithm_combobox.setVisible(False)
        self.algorithm_combobox.currentIndexChanged.connect(self.style_combobox_on_selection)
        layout.addWidget(self.algorithm_combobox)

        # Load dataset button
        self.load_button = QPushButton("Load Dataset")
        self.load_button.clicked.connect(self.load_dataset)
        layout.addWidget(self.load_button)

        # Train button
        self.train_button = QPushButton("Train Model")
        self.train_button.clicked.connect(self.train_model)
        layout.addWidget(self.train_button)

        # Save model button
        self.save_button = QPushButton("Save Model")
        self.save_button.clicked.connect(self.save_model)
        layout.addWidget(self.save_button)

        layout.addStretch()
        self.setLayout(layout)
        
        self.setWindowTitle("ML Model Trainer")

    def update_algorithm_section(self, index):
        selected_model = self.model_combobox.currentText()

        if selected_model == "Image Classification":
            algorithms = self.image_classification_algorithms
        elif selected_model == "Image Segmentation":
            algorithms = self.image_segmentation_algorithms
        else:
            algorithms = []

        if algorithms:
            self.algorithm_label.setVisible(True)
            self.algorithm_combobox.setVisible(True)
            self.algorithm_combobox.clear()
            self.algorithm_combobox.addItem("Select an algorithm...")
            self.algorithm_combobox.model().item(0).setEnabled(False)
            self.algorithm_combobox.addItems(algorithms)
            self.algorithm_combobox.setCurrentIndex(0)
        else:
            self.algorithm_label.setVisible(False)
            self.algorithm_combobox.setVisible(False)

        self.adjustSize()

    def style_combobox_on_selection(self, index):
        pass
    
    def load_dataset(self):
        task_type = self.model_combobox.currentText()

        if task_type == "Image Classification" or task_type == "Image Segmentation":
            folder_path = QFileDialog.getExistingDirectory(
                self, "Select Image Folder", "")
            if folder_path:
                loader = DatasetLoader(folder_path, data_type="image_folder")
                self.dataset = loader.load()
                print(f"Image folder loaded from {folder_path}")
        elif task_type == "Voice Classification":
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Open CSV Dataset", "", "CSV Files (*.csv);;All Files (*)")
            if file_path:
                loader = DatasetLoader(file_path, data_type="csv")
                self.dataset = loader.load()
                print(f"CSV dataset loaded from {file_path}")
        else:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Open CSV Dataset", "", "CSV Files (*.csv);;All Files (*)")
            if file_path:
                loader = DatasetLoader(file_path, data_type="csv")
                self.dataset = loader.load()
                print(f"CSV dataset loaded from {file_path}")

    def train_model(self):
        if self.dataset is None:
            print("Please load a dataset first.")
            return

        task_type = self.model_combobox.currentText()

        if task_type == "Image Classification":
            algorithm = self.algorithm_combobox.currentText()
            if algorithm == "CNN (Recommended)":
                self.model = train_keras_model(self.dataset)
            elif algorithm == "SVM":
                self.model = train_random_forest_model(self.dataset)  # replace with actual SVM trainer if available
        else:
            # For other tasks, using Random Forest for simplicity
            self.model = train_random_forest_model(self.dataset)

        print("Model training complete.")

    def save_model(self):
        if self.model:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Save Model", "", "All Files (*)")
            if file_path:
                save_model_to_file(self.model, file_path)
                print(f"Model saved to {file_path}")