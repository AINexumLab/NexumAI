import pytest
import pandas as pd
from torchvision.datasets import ImageFolder
from unittest.mock import MagicMock, patch
from PyQt6.QtWidgets import QApplication, QFileDialog
from presentation.widgets.trainer_widget import TrainerWidget

@pytest.fixture
def trainer_widget(qtbot):
    """Create and return a TrainerWidget instance."""
    widget = TrainerWidget()
    qtbot.addWidget(widget)
    widget.show()  # optional, can be useful for some tests
    return widget

@pytest.fixture
def valid_dataframe_dataset():
    return pd.DataFrame({'feature1': [1, 2], 'feature2': [3, 4], 'label': [0, 1]})

@pytest.fixture
def valid_imagefolder_dataset(tmp_path):
    class_to_dir = tmp_path / "class_a"
    class_to_dir.mkdir()
    dummy_file = class_to_dir / "dummy.jpg"
    dummy_file.write_text("dummy content")
    return ImageFolder(str(tmp_path))

def test_model_selection_shows_algorithms(trainer_widget, qtbot, valid_dataframe_dataset):
    trainer_widget.dataset = valid_dataframe_dataset
    trainer_widget.model_combobox.setCurrentIndex(1)
    qtbot.wait(100)
    assert trainer_widget.algorithm_label.isVisible()

def test_train_model_button_behavior(trainer_widget, qtbot, valid_dataframe_dataset):
    trainer_widget.dataset = valid_dataframe_dataset
    with patch("presentation.widgets.trainer_widget.ModelTrainer.train", return_value=MagicMock()) as mock_train:
        trainer_widget.train_model()
        qtbot.wait(100)
        mock_train.assert_called_once()

def test_save_model_button_behavior(trainer_widget, qtbot, valid_dataframe_dataset, tmp_path, monkeypatch):
    trainer_widget.dataset = valid_dataframe_dataset
    trainer_widget.model = MagicMock()

    fake_path = str(tmp_path / "model.pth")

    # Patch QFileDialog.getSaveFileName globally
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args, **kwargs: (fake_path, 'All Files (*)'))

    # Patch the actual saving logic to prevent real file I/O and serialization
    with patch("presentation.widgets.trainer_widget.save_model_to_file") as mock_save:
        trainer_widget.save_model()
        mock_save.assert_called_once_with(trainer_widget.model, fake_path)