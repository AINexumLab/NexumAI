from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QComboBox, QLabel, QFileDialog, QMessageBox
from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QPainter, QColor
from presentation.widgets.setting_window import SettingsWindow
from infrastructure.data.dataset_loader import DatasetLoader
from core.model_trainer import ModelTrainer
from core.model_saver import save_model_to_file
from core.model_selector import ModelSelector
from torchvision import transforms

class TrainerWidget(QWidget):
    def __init__(self):
        """
        Initializes the TrainerWidget instance, sets up attributes, and invokes UI setup.
        """
        super().__init__()
        self.dataset = None
        self.model = None
        self.init_ui()

    def init_ui(self):
        """
        Constructs and configures the graphical components and layouts,
        establishing signal-slot connections for interactive workflow control.
        """
        layout = QVBoxLayout()
        
        self.model_label = QLabel("Choose Task Type:")
        layout.addWidget(self.model_label)

        self.model_combobox = QComboBox()
        self.model_combobox.addItem("Select a model...")
        self.model_combobox.addItems(ModelSelector.get_model_types())
        self.model_combobox.model().item(0).setEnabled(False)  # Disable default prompt selection
        self.model_combobox.currentIndexChanged.connect(self.update_model_section)
        self.model_combobox.currentIndexChanged.connect(self.style_combobox_on_selection)
        layout.addWidget(self.model_combobox)

        self.algorithm_label = QLabel("Choose Algorithm:")
        self.algorithm_label.setVisible(False)
        layout.addWidget(self.algorithm_label)

        self.algorithm_combobox = QComboBox()
        self.algorithm_combobox.setVisible(False)
        self.algorithm_combobox.currentIndexChanged.connect(self.algorithm_selected)
        self.algorithm_combobox.currentIndexChanged.connect(self.style_combobox_on_selection)
        layout.addWidget(self.algorithm_combobox)

        self.load_button = QPushButton("Load Dataset")
        self.load_button.clicked.connect(self.load_dataset)
        self.load_button.setEnabled(False)
        layout.addWidget(self.load_button)

        self.train_button = QPushButton("Train Model")
        self.train_button.clicked.connect(self.train_model)
        self.train_button.setEnabled(False)
        layout.addWidget(self.train_button)

        self.save_button = QPushButton("Save Model")
        self.save_button.clicked.connect(self.save_model)
        self.save_button.setEnabled(False)
        layout.addWidget(self.save_button)

        self.restart_label = QLabel(
            "Model saved successfully. \n\nTo train a new model, kindly initiate a fresh workflow by restarting the application."
        )
        self.restart_label.setWordWrap(True)
        self.restart_label.setVisible(False)
        self.restart_label.setFixedHeight(100)
        layout.addWidget(self.restart_label)

        settings_button = QPushButton("⛭")
        settings_button.setFixedSize(QSize(25, 25))
        settings_button.setStyleSheet("""
            QPushButton {
                font-size: 24px;
                padding: 0px;
                margin: 0px;
                border-radius: 4px;
                border: none;
            }
            QPushButton:hover {
                background-color: #808080;
            }
        """)
        settings_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        settings_button.clicked.connect(self.open_settings)

        settings_layout = QHBoxLayout()
        settings_layout.addStretch()
        settings_layout.addWidget(settings_button)
        settings_layout.addStretch()
        layout.addLayout(settings_layout)
        
        layout.addStretch()
        self.setLayout(layout)
        self.setWindowTitle("NexumAI")

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(0, 0, 0, 0))
        color = QColor(40, 40, 80, int(0.6 * 255))
        painter.fillRect(self.rect(), color)
                
    def open_settings(self):
        self.settings_window = SettingsWindow()
        self.settings_window.setWindowOpacity(.80)
        self.settings_window.show()
        
    def update_model_section(self, index):
        """
        Adjusts the visibility and content of the algorithm selection UI elements
        based on the current task/model type chosen by the user.

        Parameters
        ----------
        index : int
            The index of the currently selected item in the model_combobox.
        """
        selected_model = self.model_combobox.currentText()

        if selected_model == "Image Classification":
            algorithms = ModelSelector.get_algorithms_for("Image Classification")
        elif selected_model == "Image Segmentation":
            algorithms = ModelSelector.get_algorithms_for("Image Segmentation")
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
            self.load_button.setEnabled(False)
        else:
            self.algorithm_label.setVisible(False)
            self.algorithm_combobox.setVisible(False)
            self.load_button.setEnabled(True)
        self.load_button.setText("Load Dataset")
        self.train_button.setEnabled(False)
        self.adjustSize()

    def style_combobox_on_selection(self, index):
        """
        Placeholder method intended for applying custom styling or behavior
        to combobox elements upon user selection. Currently unimplemented.

        Parameters
        ----------
        index : int
            The index of the selected combobox item.
        """
        pass

    def algorithm_selected(self, index):
        """
        Enables the dataset loading button if a valid algorithm selection is made,
        while resetting the training button state.

        Parameters
        ----------
        index : int
            The index of the currently selected algorithm.
        """
        if index > 0:
            self.load_button.setEnabled(True)
            self.load_button.setText("Load Dataset")
        self.train_button.setEnabled(False)

    def load_dataset(self):
        """
        Opens a directory selection dialog for the user to specify the dataset location,
        loads the dataset with task-specific preprocessing transformations,
        and prepares the interface for training. Supports image and voice classification
        datasets with appropriate preprocessing pipelines.
        """
        task_type = self.model_combobox.currentText()
        folder_path = None

        if task_type in ["Image Classification", "Image Segmentation"]:
            folder_path = QFileDialog.getExistingDirectory(self, "Select Image Folder", "")
            if folder_path:
                # Compose preprocessing transforms suitable for grayscale image data
                transform = transforms.Compose([
                    transforms.Resize((128, 128)),
                    transforms.Grayscale(num_output_channels=1),
                    transforms.ToTensor()
                ])

                loader = DatasetLoader(
                    folder_path,
                    data_type="image_folder",
                    transform=transform
                )
                self.dataset = loader.load()
                print(f"Image folder loaded from {folder_path}")

        elif task_type == "Voice Classification":
            folder_path = QFileDialog.getExistingDirectory(self, "Select Audio Folder", "")
            if folder_path:
                loader = DatasetLoader(folder_path, data_type="audio_folder")
                self.dataset = loader.load()
                print(f"Audio folder loaded from {folder_path}")

        else:
            QMessageBox.warning(
                self,
                "Unsupported Task",
                f"The selected task type '{task_type}' is not supported for dataset loading.",
            )

        if self.dataset:
            self.load_button.setText("Loaded Successfully")
            self.load_button.setEnabled(False)
            self.train_button.setEnabled(True)
            print("Dataset loaded successfully.")

    def train_model(self):
        """
        Initiates the model training procedure using the loaded dataset,
        selected model type, and algorithm. It disables selection inputs during
        training to preserve workflow integrity and enables the model saving UI
        upon successful completion.
        """
        if self.dataset is None:
            print("Please load a dataset first.")
            QMessageBox.warning(
                self,
                "Dataset Required",
                "Please load a dataset before initiating training.",
            )
            return

        self.model_combobox.setEnabled(False)
        model_type = self.model_combobox.currentText()
        self.algorithm_combobox.setEnabled(False)
        task_type = self.algorithm_combobox.currentText()
        self.model = ModelTrainer(self.dataset).train(model_type=model_type, task_type=task_type)

        if self.model:
            self.train_button.setText("Trained Successfully")
            self.train_button.setEnabled(False)
            self.save_button.setEnabled(True)
            print("Model training complete.")

    def save_model(self):
        """
        Facilitates saving the trained model to persistent storage by opening a
        file save dialog, writing the model to the specified location, and providing
        user feedback. Post-save, prompts the user to restart the application to begin
        a new training session.
        """
        if self.model:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Save Model", "", "All Files (*)")
            if file_path:
                save_model_to_file(self.model, file_path)
                self.save_button.setText("Saved Successfully")
                self.save_button.setEnabled(False)
                print("Model saved successfully.")
                self.restart_label.setVisible(True)