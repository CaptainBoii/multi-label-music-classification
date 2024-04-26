import os
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.models import Sequential
from keras.preprocessing import image_dataset_from_directory
from keras.utils import Sequence
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input

results_labels = ['Test loss', 'Test accuracy', 'F1', 'Balanced', 'Precision', 'Recall']
spect_types = ['Spectrograms/', 'MelSpectrograms/']
genres = ['Ambient', 'Classical', 'Dance', 'Electronic', 'Experimental', 'Folk', 'Hip-Hop', 'Industrial & Noise',
          'Jazz', 'Metal', 'Pop', 'Psychedelia', 'Punk', 'R&B', 'Rock', 'Singer-Songwriter']
main_dir = 'Data/Experiment_1_2/'
save_dir = 'Models/Experiment_1/'
test = '/Test'
train = '/Train'
valid = '/Valid'
seed = 97
batch_size = 50
epochs = 3
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


class BalancedDataGenerator(Sequence):
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


for genre in genres:
    for spect_type in spect_types:
        train_positive = load_images_from_directory(main_dir + spect_type + genre + train, 'Positive')
        train_negative = load_images_from_directory(main_dir + spect_type + genre + train, 'Negative')

        data_generator = BalancedDataGenerator(train_positive, train_negative, batch_size)
        # Load test data
        # test_generator = image_dataset_from_directory(
        #     directory=main_dir + spect_type + genre + test,
        #     labels='inferred',
        #     image_size=image_size_trim,
        #     color_mode='rgb',
        #     batch_size=100,
        #     label_mode='binary'
        # )
        valid_generator = image_dataset_from_directory(
            directory=main_dir + spect_type + genre + valid,
            labels='inferred',
            image_size=image_size_trim,
            color_mode='rgb',
            batch_size=100,
            label_mode='binary'
        )
        model = Sequential([
            Input(shape=image_size),
            Conv2D(32, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        model_history = model.fit(
            data_generator,
            validation_data=valid_generator,
            epochs=epochs)

        model.save(save_dir+spect_type+genre+"/"+genre+".keras")
