import os
import pandas as pd
import numpy as np
from keras.src.utils import image_dataset_from_directory
import tensorflow as tf
from tensorflow.keras.models import load_model
import glob
from sklearn.metrics import (
    recall_score,
    precision_score,
    balanced_accuracy_score,
    f1_score,
    hamming_loss,
    confusion_matrix,
    accuracy,
    loss,
)


batch_size = 250
image_size = (250, 250, 3)
image_size_trim = (250, 250)
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

model_path = "./Models/Experiment_"
results_labels = [
    "Genre",
    "Accuracy",
    "Loss",
    "F1",
    "Balanced",
    "Precision",
    "Recall",
    "Hamming",
]


def first_three() -> None:
    for experiment in range(1, 4):
        aggregated_test = {"Spectrograms": np.array(), "MelSpectrograms": np.array()}
        aggregated_result = {"Spectrograms": np.array(), "MelSpectrograms": np.array()}
        results = {
            "Spectrograms": np.zeros((17, 8)),
            "MelSpectrograms": np.zeros((17, 8)),
        }
        for spect_type in ["Spectrograms", "MelSpectrograms"]:
            for genre_id, genre in enumerate(genres):
                for genre_model in glob.glob(
                    "./Models/Experiment_3/" + spect_type + "/" + genre + "*.keras"
                ):
                    new_model = load_model(genre_model)
                    test_dataset = image_dataset_from_directory(
                        directory=os.path.join("./Data/Test/", spect_type, genre),
                        labels="inferred",
                        image_size=image_size_trim,
                        color_mode="rgb",
                        batch_size=batch_size,
                        label_mode="binary",
                    )
                    y_test = []
                    for images, labels in test_dataset:
                        y_test.extend(labels.numpy())
                    y_test = np.array(y_test)

                    test_loss, test_acc = new_model.evaluate(test_dataset)

                    y_result_pre = new_model.predict(test_dataset)
                    y_result = (y_result_pre > 0.5).astype("int32").flatten()

                    aggregated_test[spect_type] += y_test
                    aggregated_result[spect_type] += y_result

                    # CM = confusion_matrix(y_test, y_result)

                    results[spect_type][genre_id][0] = genre
                    results[spect_type][genre_id][1] = test_acc
                    results[spect_type][genre_id][2] = test_loss
                    results[spect_type][genre_id][3] = f1_score(
                        y_test, y_result, average="binary"
                    )
                    results[spect_type][genre_id][4] = balanced_accuracy_score(
                        y_test, y_result
                    )
                    results[spect_type][genre_id][5] = precision_score(
                        y_test, y_result, average="binary"
                    )
                    results[spect_type][genre_id][6] = recall_score(
                        y_test, y_result, average="binary"
                    )
                    results[spect_type][genre_id][7] = None
            results[spect_type][len(genres)][0] = "Aggregated"
            results[spect_type][len(genres)][1] = accuracy(
                aggregated_test[spect_type], aggregated_result[spect_type]
            )
            results[spect_type][len(genres)][2] = loss(
                aggregated_test[spect_type], aggregated_result[spect_type]
            )
            results[spect_type][len(genres)][3] = f1_score(
                aggregated_test[spect_type],
                aggregated_result[spect_type],
                average="binary",
            )
            results[spect_type][len(genres)][4] = balanced_accuracy_score(
                aggregated_test[spect_type], aggregated_result[spect_type]
            )
            results[spect_type][len(genres)][5] = precision_score(
                aggregated_test[spect_type],
                aggregated_result[spect_type],
                average="binary",
            )
            results[spect_type][len(genres)][6] = recall_score(
                aggregated_test[spect_type],
                aggregated_result[spect_type],
                average="binary",
            )
            results[spect_type][len(genres)][7] = hamming_loss(
                aggregated_test[spect_type], aggregated_result[spect_type]
            )
            df = pd.DataFrame(results[spect_type])
            df.to_csv(
                "results_experiment_" + str(experiment) + "_" + spect_type + ".csv",
                header=results_labels,
                index=False,
            )


if __name__ == "__main__":
    first_three()
