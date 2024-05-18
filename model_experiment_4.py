import os
import numpy as np
import tensorflow as tf
from keras import Sequential
from keras.src.callbacks import ModelCheckpoint, EarlyStopping
from keras.src.utils import load_img, img_to_array, image_dataset_from_directory
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.regularizers import L1L2

spect_types = ["Spectrograms/", "MelSpectrograms/"]
genres = [
    "Ambient",
    "Classical",
    "Dance",
    "Electronic",
    "Experimental",
    "Folk",
    "Hip-Hop",
    "Industrial & Noise",
    "Jazz",
    "Metal",
    "Pop",
    "Psychedelia",
    "Punk",
    "R&B",
    "Rock",
    "Singer-Songwriter",
]
main_dir = "Data/Experiment_4/"
save_dir = "Models/Experiment_4/"
test = "/Test"
train = "/Train"
valid = "/Valid"
seed = 97
batch_size = 200
epochs = 200
image_size = (250, 250, 3)
image_size_trim = (250, 250)


def load_images_from_directory(directory: str, label: str) -> np.array:
    images = []
    label_dir = os.path.join(directory, label)
    if os.path.isdir(label_dir):
        for filename in os.listdir(label_dir):
            filepath = os.path.join(label_dir, filename)
            image = load_img(filepath, target_size=image_size)
            image_array = img_to_array(image) / 255.0  # Normalize pixel values
            images.append(image_array)
    return np.array(images)


tf.random.set_seed(seed)
for genre in genres:
    print(genre)
    for spect_type in spect_types:
        print()
        print(spect_type)
        print()

        train_generator = image_dataset_from_directory(
            directory=main_dir + spect_type + genre + train,
            labels="inferred",
            image_size=image_size_trim,
            color_mode="rgb",
            batch_size=batch_size,
            label_mode="binary",
        )

        valid_generator = image_dataset_from_directory(
            directory=main_dir + spect_type + genre + valid,
            labels="inferred",
            image_size=image_size_trim,
            color_mode="rgb",
            batch_size=batch_size,
            label_mode="binary",
        )

        model_checkpoint_callback = ModelCheckpoint(
            filepath=save_dir
            + spect_type
            + genre
            + "-{epoch:02d}-{val_accuracy:.2f}-{val_loss:.2f}.keras",
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
        )

        early_stopping = EarlyStopping(
            monitor="val_loss", patience=50, verbose=0, min_delta=1e-4
        )

        l1l2 = L1L2(l1=0.03, l2=0.03)

        optimizer = Adam(learning_rate=0.001, use_ema=True, ema_momentum=0.7)
        model = Sequential(
            [
                Input(shape=image_size),
                Conv2D(32, (3, 3), activation="relu"),
                MaxPooling2D((2, 2)),
                Conv2D(64, (3, 3), activation="relu"),
                MaxPooling2D((2, 2)),
                Conv2D(128, (3, 3), activation="relu"),
                MaxPooling2D((2, 2)),
                Conv2D(256, (3, 3), activation="relu"),
                MaxPooling2D((2, 2)),
                Flatten(),
                Dense(256, activation="relu", kernel_regularizer=l1l2),
                Dropout(0.5),
                Dense(1, activation="sigmoid"),
            ]
        )

        model.compile(
            optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"]
        )

        model_history = model.fit(
            train_generator,
            validation_data=valid_generator,
            epochs=epochs,
            callbacks=[model_checkpoint_callback, early_stopping],
        )
