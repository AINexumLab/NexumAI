import pandas as pd
from torchvision import datasets, transforms
import os

class DatasetLoader:
    def __init__(self, file_path: str, data_type: str, **kwargs):
        self.file_path = file_path
        self.data_type = data_type.lower()
        self.kwargs = kwargs
        self.data = None

    def load(self):
        if self.data_type == "csv":
            self.data = self._load_csv()
        elif self.data_type == "image_folder":
            self.data = self._load_image_folder()
        else:
            raise ValueError(f"Unsupported data type: {self.data_type}")
        return self.data

    def _load_csv(self):
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"CSV file not found: {self.file_path}")
        return pd.read_csv(self.file_path, **self.kwargs)

    def _load_image_folder(self):
        if not os.path.isdir(self.file_path):
            raise FileNotFoundError(f"Image folder not found: {self.file_path}")
        transform = self.kwargs.get("transform", transforms.ToTensor())
        return datasets.ImageFolder(root=self.file_path, transform=transform)