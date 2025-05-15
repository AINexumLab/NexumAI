import joblib
from tensorflow.keras.models import Sequential

def save_model_to_file(model, file_path):
    if isinstance(model, Sequential):
        model.save(file_path + ".h5")
        print(f"Model saved as {file_path}.h5")
    else:
        joblib.dump(model, file_path + ".pkl")
        print(f"Model saved as {file_path}.pkl")
