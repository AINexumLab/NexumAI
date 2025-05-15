from tensorflow import keras
from sklearn.ensemble import RandomForestClassifier

def train_keras_model(dataset):
    x = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_dim=x.shape[1]),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(x, y, epochs=5)
    print("Model trained using Keras.")
    return model

def train_random_forest_model(dataset):
    x = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    model = RandomForestClassifier()
    model.fit(x, y)
    print("Model trained using Scikit-learn Random Forest.")
    return model
