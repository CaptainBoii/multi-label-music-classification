import gc
import os
import numpy as np
import tensorflow as tf
from keras import Sequential
from keras.src.trainers.data_adapters.py_dataset_adapter import PyDataset
from keras.src.utils import load_img, img_to_array, image_dataset_from_directory
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input

results_labels = ['Test loss', 'Test accuracy', 'F1', 'Balanced', 'Precision', 'Recall']
spect_types = ['Spectrograms/', 'MelSpectrograms/']
genres = [
    'Ambient', 'Classical',
    # 'Dance', 'Electronic',
    # 'Experimental', 'Folk',
    # 'Hip-Hop', 'Industrial & Noise',
    # 'Jazz', 'Metal',
    # 'Pop', 'Psychedelia',
    # 'Punk', 'R&B',
    # 'Rock', 'Singer-Songwriter'
]
main_dir = 'Data/Experiment_1_2/'
save_dir = 'Models/Experiment_1/'
test = '/Test'
train = '/Train'
valid = '/Valid'
training = 'Training/'
seed = 97
batch_size = 1
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


class BalancedDataGenerator(PyDataset):
    def __init__(self, positive_data: np.array, negative_data: np.array, batch_size_: int):
        super().__init__()
        self.positive_data = positive_data
        self.negative_data = negative_data
        self.batch_size = batch_size_
        self.positive_indices = np.arange(len(positive_data))
        self.negative_indices = np.arange(len(negative_data))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.positive_data) / self.batch_size))

    def __getitem__(self, index):
        start_index = index * self.batch_size
        end_index = min((index + 1) * self.batch_size, len(self.positive_indices))

        positive_indices = self.positive_indices[start_index:end_index]
        negative_indices = self.negative_indices[start_index:end_index]

        X_positive = self.positive_data[positive_indices]
        X_negative = self.negative_data[negative_indices]
        X_positive = X_positive.reshape((self.batch_size,) + X_positive.shape[1:])
        X_negative = X_negative.reshape((self.batch_size,) + X_negative.shape[1:])

        X = np.concatenate([X_positive, X_negative], axis=0)
        y = np.concatenate([np.ones(len(X_positive)), np.zeros(len(X_negative))])

        return X, y

    def on_epoch_end(self):
        np.random.shuffle(self.positive_indices)
        np.random.shuffle(self.negative_indices)


tf.random.set_seed(seed)
for genre in genres:
    print(genre)
    for spect_type in spect_types:
        print()
        print(spect_type)
        print()
        train_positive = load_images_from_directory(main_dir + spect_type + training + genre + train, 'Positive')
        train_negative = load_images_from_directory(main_dir + spect_type + training + genre + train, 'Negative')

        data_generator = BalancedDataGenerator(train_positive, train_negative, batch_size)
        valid_generator = image_dataset_from_directory(
            directory=main_dir + spect_type + training + genre + valid,
            labels='inferred',
            image_size=image_size_trim,
            color_mode='rgb',
            batch_size=batch_size,
            label_mode='binary'
        )
        optimizer = Adam(learning_rate=0.001, use_ema=True, ema_momentum=0.7)
        model = Sequential([
            Input(shape=image_size),
            Conv2D(32, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(256, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(256, activation='relu'),
            Dense(256, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])

        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

        model_history = model.fit(
            data_generator,
            validation_data=valid_generator,
            epochs=epochs)

        model.save(save_dir + spect_type + genre + ".keras")
        del model
        del train_positive
        del train_negative
        del data_generator
        del valid_generator
        gc.collect()
