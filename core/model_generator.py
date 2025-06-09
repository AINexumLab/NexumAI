from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from sklearn.svm import SVC

class ModelGenerator:
    @classmethod
    def get_image_classification_model(self, algorithm: str, input_shape, num_classes=1):
        if algorithm == "CNN (Recommended)":
            return self._build_cnn(input_shape, num_classes)
        elif algorithm == "SVM":
            return SVC(kernel='linear', probability=True)
        else:
            raise ValueError(f"Unsupported algorithm for image classification: {algorithm}")

    @classmethod
    def get_image_segmentation_model(self, algorithm: str, input_shape, num_classes=1):
        if algorithm == "U-Net (Recommended)":
            return self._build_unet(input_shape, num_classes)
        elif algorithm == "CNN":
            return self._build_cnn(input_shape, num_classes)
        else:
            raise ValueError(f"Unsupported algorithm for image segmentation: {algorithm}")
    
    @classmethod
    def get_voice_classification_model(self):
        return SVC(kernel='linear', probability=True)
    
    # PRIVARE FUNCTIONS:
    
    @classmethod
    def _build_cnn(self, input_shape, num_classes):
        model = models.Sequential()
        model.add(layers.InputLayer(shape=input_shape))
        model.add(layers.Conv2D(32, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(num_classes, activation='sigmoid' if num_classes == 1 else 'softmax'))

        model.compile(optimizer=Adam(),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        return model
    
    @classmethod
    def _build_unet(self, input_shape, num_classes):
        inputs = layers.Input(shape=input_shape)

        # Encoder
        c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
        c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
        p1 = layers.MaxPooling2D((2, 2))(c1)

        c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
        c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
        p2 = layers.MaxPooling2D((2, 2))(c2)

        # Bottleneck
        c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
        c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c3)

        # Decoder
        u1 = layers.UpSampling2D((2, 2))(c3)
        u1 = layers.concatenate([u1, c2])
        c4 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u1)
        c4 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c4)

        u2 = layers.UpSampling2D((2, 2))(c4)
        u2 = layers.concatenate([u2, c1])
        c5 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u2)
        c5 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c5)

        outputs = layers.Conv2D(num_classes, (1, 1),
                                activation='sigmoid' if num_classes == 1 else 'softmax')(c5)

        model = models.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam',
                      loss='binary_crossentropy' if num_classes == 1 else 'categorical_crossentropy',
                      metrics=['accuracy'])
        return model