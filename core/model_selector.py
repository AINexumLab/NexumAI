from typing import Dict, List

class ModelSelector:
    model_types: List[str] = [
        "Image Classification",
        "Image Segmentation",
        "Voice Classification"
    ]

    algorithms_by_type: Dict[str, List[str]] = {
        "Image Classification": [
            "CNN (Recommended)",
            "SVM"
        ],
        "Image Segmentation": [
            "U-Net (Recommended)",
            "CNN"
        ],
        "Voice Classification": []
    }
    
    @classmethod
    def get_model_types(cls) -> List[str]:
        """Returns the list of available model types."""
        return cls.model_types

    @classmethod
    def get_algorithms_for(cls, model_type: str) -> List[str]:
        """Returns the list of algorithms for a given model type."""
        return cls.algorithms_by_type.get(model_type, [])

    @classmethod
    def is_valid_selection(cls, model_type: str, algorithm: str) -> bool:
        """Checks whether the algorithm is valid for the selected model type."""
        return algorithm in cls.algorithms_by_type.get(model_type, [])
