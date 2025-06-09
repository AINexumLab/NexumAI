import numpy as np
import pandas as pd
import pytest
import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from core.model_trainer import ModelTrainer
from core.model_generator import ModelGenerator
from unittest.mock import MagicMock

# ----------- UTILITIES -----------

class DummyImageFolder(ImageFolder):
    def __init__(self, transform=None):
        self.samples = [(None, 0)] * 10
        self.targets = [0] * 10
        self.classes = ['class0']
        self.imgs = self.samples
        self.transform = transform

    def __len__(self):
        return 10

    def __getitem__(self, index):
        image = torch.randn(1, 128, 128)  # 1-channel image
        mask = torch.randint(0, 2, (128, 128))  # segmentation mask (binary)
        return image, mask

# ----------- TEST CASES -----------

def test_train_image_classification_cnn(monkeypatch):
    # Create a dummy tabular dataset
    data = pd.DataFrame(np.random.rand(100, 5))
    data.iloc[:, -1] = np.random.randint(0, 3, size=100)
    trainer = ModelTrainer(data)

    # Patch model.fit to avoid actual training
    dummy_model = ModelGenerator.get_image_classification_model("CNN (Recommended)", input_shape=(128, 128, 1), num_classes=3)
    dummy_model.fit = MagicMock(return_value=None)

    monkeypatch.setattr(ModelGenerator, "get_image_classification_model", lambda *args, **kwargs: dummy_model)

    model = trainer.train("Image Classification", "CNN (Recommended)")
    assert model is not None
    dummy_model.fit.assert_called_once()


def test_train_image_classification_svm():
    # Create dummy tabular dataset
    data = pd.DataFrame(np.random.rand(50, 4))
    data.iloc[:, -1] = np.random.randint(0, 2, size=50)
    trainer = ModelTrainer(data)

    model = trainer.train("Image Classification", "SVM")
    assert hasattr(model, "fit")
    assert hasattr(model, "predict")


def test_train_image_segmentation_unet(monkeypatch):
    # Fake image dataset for segmentation
    dataset = DummyImageFolder()
    trainer = ModelTrainer(dataset)

    # Patch model.fit to skip training
    dummy_model = ModelGenerator.get_image_segmentation_model("U-Net (Recommended)", input_shape=(128, 128, 1), num_classes=1)
    dummy_model.fit = MagicMock(return_value=None)

    monkeypatch.setattr(ModelGenerator, "get_image_segmentation_model", lambda *args, **kwargs: dummy_model)

    model = trainer.train("Image Segmentation", "U-Net (Recommended)")
    assert model is not None
    dummy_model.fit.assert_called_once()


def test_train_image_segmentation_cnn(monkeypatch):
    dataset = DummyImageFolder()
    trainer = ModelTrainer(dataset)

    dummy_model = ModelGenerator.get_image_segmentation_model("CNN", input_shape=(128, 128, 1), num_classes=1)
    dummy_model.fit = MagicMock(return_value=None)

    monkeypatch.setattr(ModelGenerator, "get_image_segmentation_model", lambda *args, **kwargs: dummy_model)

    model = trainer.train("Image Segmentation", "CNN")
    assert model is not None
    dummy_model.fit.assert_called_once()


def test_train_voice_classification(monkeypatch):
    # Create dummy voice features as tabular data
    data = pd.DataFrame(np.random.rand(60, 20))
    data.iloc[:, -1] = np.random.randint(0, 2, size=60)
    trainer = ModelTrainer(data)

    dummy_model = ModelGenerator.get_voice_classification_model()
    dummy_model.fit = MagicMock(return_value=None)

    monkeypatch.setattr(ModelGenerator, "get_voice_classification_model", lambda: dummy_model)

    model = trainer.train("Voice Classification", "")
    assert model is not None
    dummy_model.fit.assert_called_once()


def test_invalid_model_type_raises():
    data = pd.DataFrame(np.random.rand(10, 4))
    trainer = ModelTrainer(data)

    with pytest.raises(ValueError):
        trainer.train("Unknown Task", "")

def test_invalid_task_type_raises():
    data = pd.DataFrame(np.random.rand(10, 4))
    trainer = ModelTrainer(data)

    with pytest.raises(ValueError):
        trainer.train("Image Classification", "UnknownModel")

def test_invalid_dataset_type_raises():
    invalid_data = "I am not a dataset"
    with pytest.raises(TypeError):
        ModelTrainer(invalid_data)