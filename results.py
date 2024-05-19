import os
import pandas as pd
import numpy as np
from keras.src.utils import image_dataset_from_directory
from tensorflow.keras.models import load_model
import glob
from sklearn.metrics import (
    recall_score,
    precision_score,
    f1_score,
    hamming_loss,
    accuracy_score,
    log_loss,
)


batch_size = 200
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
    "Precision",
    "Recall",
    "Hamming",
    "0/1",
]


def chunkify(lst: list) -> list:
    result = []
    for i in range(16):
        result.append(lst[i * 1801 : (i + 1) * 1801])
    return result


def metrics() -> None:
    for experiment in range(1, 6):
        aggregated_test = {"Spectrograms": [], "MelSpectrograms": []}
        aggregated_result = {"Spectrograms": [], "MelSpectrograms": []}
        results = {
            "Spectrograms": np.zeros((17, 8), dtype="O"),
            "MelSpectrograms": np.zeros((17, 8), dtype="O"),
        }
        dataset_type = "Default"
        if experiment == 4:
            dataset_type = "Processed"
        elif experiment == 5:
            dataset_type = "Scaled"
        for spect_type in ["Spectrograms", "MelSpectrograms"]:
            for genre_id, genre in enumerate(genres):
                for genre_model in glob.glob(
                    "./Models/Experiment_"
                    + str(experiment)
                    + "/"
                    + spect_type
                    + "/"
                    + genre
                    + "*.keras"
                ):
                    new_model = load_model(genre_model)
                    test_dataset = image_dataset_from_directory(
                        directory=os.path.join(
                            "./Data/Test/", dataset_type, spect_type, genre
                        ),
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

                    # test_loss, test_acc = new_model.evaluate(test_dataset)

                    y_result_pre = new_model.predict(test_dataset)
                    y_result = (y_result_pre > 0.5).astype("int32").flatten()

                    y_test_list = np.concatenate(y_test, axis=0)
                    aggregated_test[spect_type] += list(y_test_list)
                    aggregated_result[spect_type] += list(y_result)

                    # CM = confusion_matrix(y_test, y_result)

                    results[spect_type][genre_id][0] = genre
                    results[spect_type][genre_id][1] = round(
                        accuracy_score(y_test, y_result), 3
                    )
                    results[spect_type][genre_id][2] = round(
                        log_loss(y_test, y_result), 3
                    )
                    results[spect_type][genre_id][3] = round(
                        f1_score(y_test, y_result, average="binary"), 3
                    )
                    results[spect_type][genre_id][4] = round(
                        precision_score(y_test, y_result, average="binary"), 3
                    )
                    results[spect_type][genre_id][5] = round(
                        recall_score(y_test, y_result, average="binary"), 3
                    )
                    results[spect_type][genre_id][6] = None
                    results[spect_type][genre_id][7] = None
            results[spect_type][len(genres)][0] = "Aggregated"
            results[spect_type][len(genres)][1] = round(
                accuracy_score(
                    aggregated_test[spect_type], aggregated_result[spect_type]
                ),
                3,
            )
            results[spect_type][len(genres)][2] = round(
                log_loss(aggregated_test[spect_type], aggregated_result[spect_type]), 3
            )
            results[spect_type][len(genres)][3] = round(
                f1_score(
                    aggregated_test[spect_type],
                    aggregated_result[spect_type],
                    average="binary",
                ),
                3,
            )
            results[spect_type][len(genres)][4] = round(
                precision_score(
                    aggregated_test[spect_type],
                    aggregated_result[spect_type],
                    average="binary",
                ),
                3,
            )
            results[spect_type][len(genres)][5] = round(
                recall_score(
                    aggregated_test[spect_type],
                    aggregated_result[spect_type],
                    average="binary",
                ),
                3,
            )
            results[spect_type][len(genres)][6] = round(
                hamming_loss(
                    aggregated_test[spect_type], aggregated_result[spect_type]
                ),
                3,
            )

            binary_correct = 0

            chunk_test = chunkify(aggregated_test[spect_type])
            chunk_result = chunkify(aggregated_result[spect_type])

            for test_file in range(1801):
                partial_sum = 0
                for model in range(16):
                    if chunk_test[model][test_file] == chunk_result[model][test_file]:
                        partial_sum += 1
                if partial_sum == 16:
                    binary_correct += 1

            results[spect_type][len(genres)][7] = round(
                (float(binary_correct) / 1801), 3
            )
            df = pd.DataFrame(results[spect_type])
            df.to_csv(
                "./Results/results_experiment_"
                + str(experiment)
                + "_"
                + spect_type
                + ".csv",
                header=results_labels,
                index=False,
            )


if __name__ == "__main__":
    metrics()
