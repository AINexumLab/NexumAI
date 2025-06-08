from torch.utils.data import DataLoader
import pandas as pd
from torchvision.datasets import ImageFolder
from core.model_generator import ModelGenerator
class ModelTrainer:
    def __init__(self, dataset):
        self.dataset = dataset

        if isinstance(dataset, pd.DataFrame):
            # Tabular dataset
            self.x = dataset.iloc[:, :-1].values
            self.y = dataset.iloc[:, -1].values

        elif isinstance(dataset, ImageFolder):
            # Image dataset
            dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)

            try:
                images, labels = next(iter(dataloader))
            except StopIteration:
                raise ValueError("The provided ImageFolder dataset is empty.")

            self.x = images.numpy()
            self.y = labels.numpy()

        else:
            raise TypeError("Unsupported dataset type. Expected pd.DataFrame or torchvision.datasets.ImageFolder.")
    def train(self, model_type: str, task_type: str):
        if model_type == "Image Classification":
            return self._train_image_classification_model(task_type)
        elif model_type == "Image Segmentation":
            return self._train_image_segmentation_model(task_type)
        elif model_type == "Voice Classification":
            return self._train_voice_classification_model()
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    # PRIVARE FUNCTIONS:
    
    def _train_image_classification_model(self, task_type: str):
        if task_type == "CNN (Recommended)":
            input_shape = (28, 28, 1)  # example shape
            model = ModelGenerator.get_image_classification_model(task_type, input_shape)
            model.fit(self.x, self.y, epochs=1, verbose=1)
            print("Trained CNN for image classification.")
            return model

        elif task_type == "SVM":
            input_shape = self.x.shape[1:]
            model = ModelGenerator.get_image_classification_model(task_type, input_shape)
            model.fit(self.x, self.y)
            print("Trained SVM for image classification.")
            return model

        else:
            raise ValueError(f"Unsupported task type for image classification: {task_type}")

    def _train_image_segmentation_model(self, task_type: str):
        if task_type == "U-Net (Recommended)":
            input_shape = (128, 128, 1)  # example shape
            model = ModelGenerator.get_image_segmentation_model(task_type, input_shape)
            # Dummy reshape; adapt this part to your segmentation preprocessing
            x = self.x.reshape(-1, 128, 128, 1)
            y = self.y.reshape(-1, 128, 128, 1)
            model.fit(x, y, epochs=5, verbose=1)
            print("Trained U-Net for image segmentation.")
            return model

        elif task_type == "CNN":
            input_shape = (128, 128, 1)
            model = ModelGenerator.get_image_segmentation_model(task_type, input_shape)
            x = self.x.reshape(-1, 128, 128, 1)
            y = self.y.reshape(-1, 1)
            model.fit(x, y, epochs=5, verbose=1)
            print("Trained CNN for image segmentation.")
            return model

        else:
            raise ValueError(f"Unsupported task type for image segmentation: {task_type}")

    def _train_voice_classification_model(self):
        model = ModelGenerator.get_voice_classification_model()
        model.fit(self.x, self.y)
        print("Trained SVM for voice classification.")
        return model