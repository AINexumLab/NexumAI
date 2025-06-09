import pytest
from tensorflow.keras.models import Model
from sklearn.svm import SVC
from core.model_generator import ModelGenerator

def test_get_image_classification_model_cnn():
    input_shape = (64, 64, 3)
    model = ModelGenerator.get_image_classification_model("CNN (Recommended)", input_shape, num_classes=1)
    assert isinstance(model, Model)
    assert model.input_shape[1:] == input_shape
    assert model.output_shape[-1] == 1
    assert model.loss == 'sparse_categorical_crossentropy'

def test_get_image_classification_model_svm():
    model = ModelGenerator.get_image_classification_model("SVM", input_shape=(64,64,3))
    assert isinstance(model, SVC)
    assert model.kernel == "linear"
    assert model.probability is True

def test_get_image_classification_model_invalid_algorithm():
    with pytest.raises(ValueError):
        ModelGenerator.get_image_classification_model("InvalidAlgo", input_shape=(64,64,3))

def test_get_image_segmentation_model_unet():
    input_shape = (128, 128, 3)
    num_classes = 1
    model = ModelGenerator.get_image_segmentation_model("U-Net (Recommended)", input_shape, num_classes)
    assert isinstance(model, Model)
    assert model.input_shape[1:] == input_shape
    assert model.output_shape[-1] == num_classes
    # Check loss for binary classification
    assert model.loss == 'binary_crossentropy'

def test_get_image_segmentation_model_cnn():
    input_shape = (128, 128, 3)
    num_classes = 2
    model = ModelGenerator.get_image_segmentation_model("CNN", input_shape, num_classes)
    assert isinstance(model, Model)
    assert model.output_shape[-1] == num_classes
    # Multi-class last layer activation softmax
    last_layer_activation = model.layers[-1].activation.__name__
    assert last_layer_activation == "softmax"

def test_get_image_segmentation_model_invalid_algorithm():
    with pytest.raises(ValueError):
        ModelGenerator.get_image_segmentation_model("InvalidAlgo", input_shape=(64,64,3))

def test_get_voice_classification_model_returns_svc():
    model = ModelGenerator.get_voice_classification_model()
    assert isinstance(model, SVC)
    assert model.kernel == "linear"
    assert model.probability is True
