import csv
import glob
import random
from copy import deepcopy
import shutil
from math import ceil, floor

spectrogram_data = "Spectrograms/Spectrograms/"
melspectrogram_data = "Spectrograms/MelSpectrograms/"
training_data = "Data/Experiment_3/Spectrograms/Training/"
mel_training_data = "Data/Experiment_3/MelSpectrograms/Training/"

all_files = glob.glob(spectrogram_data + "*.png")
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

genres_files: list[list[int]] = [
    [],
    [],
    [],
    [],
    [],
    [],
    [],
    [],
    [],
    [],
    [],
    [],
    [],
    [],
    [],
    [],
]

all_files = glob.glob("Spectrograms/Spectrograms/*.png")
test_count = 1801


def stringify_list(lst: list) -> str:
    result = ""
    for element in lst:
        if result != "":
            result += "_"
        result += str(element)
    return result


for i, genre in enumerate(genres):
    with open("Labels/" + genre + ".txt", "r") as file:
        for line in file:
            genres_files[i].append(int(line.strip()))

unique_dict: dict[str, list[str]] = {}

for k in range(1, 1468):
    genre_classes = []
    for i in range(len(genres_files)):
        if k in genres_files[i]:
            genre_classes.append(i)
    code = stringify_list(genre_classes)
    if code in unique_dict:
        unique_dict[code] += glob.glob("Spectrograms/Spectrograms/" + str(k) + "_*.png")
    else:
        unique_dict[code] = glob.glob("Spectrograms/Spectrograms/" + str(k) + "_*.png")


with open("Data/test_labels.csv") as csv_file:
    reader = csv.reader(csv_file)
    test_dict = dict(reader)

    test_keys = list(test_dict.keys())
    test_files = list(map(lambda t_file: spectrogram_data + t_file, test_keys))
    all_files = list(set(all_files) - set(test_files))
    for test_file in test_files:
        for key in unique_dict.keys():
            if test_file in unique_dict[key]:
                unique_dict[key].remove(test_file)

for i, genre in enumerate(genres):
    positive_files = []
    for key in unique_dict.keys():
        if str(i) in key.split("_"):
            positive_files += unique_dict[key]

    random.shuffle(positive_files)
    positive_n = len(positive_files)
    k = int(positive_n / 10)
    data_positive = {"Valid": positive_files[:k], "Train": positive_files[k:]}

    total_negative = 0
    keys: list[str] = []
    for key in unique_dict.keys():
        if str(i) not in key.split("_"):
            keys.append(key)
            total_negative += len(unique_dict[key])

    others_val = []
    others_train = []
    data_negative: dict[str, list[str]] = {"Valid": [], "Train": []}
    for key in keys:
        key_n_pre = (len(unique_dict[key]) * positive_n) / total_negative
        key_n = (
            ceil(key_n_pre) if key_n_pre - int(key_n_pre) >= 0.4 else floor(key_n_pre)
        )
        temp_list = deepcopy(unique_dict[key])
        random.shuffle(temp_list)
        if key_n >= 10:
            val = key_n / 10
            val_final = ceil(val) if val - int(val) > 0.5 else floor(val)
            data_negative["Valid"] += temp_list[:val_final]
            data_negative["Train"] += temp_list[val_final:key_n]
        elif key_n > 1:
            data_negative["Train"] += temp_list[: key_n - 1]
            others_val += temp_list[-1:]
        else:
            others_train += temp_list

    random.shuffle(others_val)
    random.shuffle(others_train)
    others_val_n = len(data_positive["Valid"]) - len(data_negative["Valid"])
    others_train_n = len(data_positive["Train"]) - len(data_negative["Train"])
    data_negative["Valid"] += others_val[:others_val_n]
    data_negative["Train"] += others_train[:others_train_n]

    for set_name in ["Train", "Valid"]:
        for file in data_positive[set_name]:
            file_redux = file.split("/Spectrograms/")[1]
            shutil.copyfile(
                spectrogram_data + file_redux,
                training_data + genre + "/" + set_name + "/Positive/" + file_redux,
            )
            shutil.copyfile(
                melspectrogram_data + file_redux,
                mel_training_data + genre + "/" + set_name + "/Positive/" + file_redux,
            )

        for file in data_negative[set_name]:
            file_redux = file.split("/Spectrograms/")[1]
            shutil.copyfile(
                spectrogram_data + file_redux,
                training_data + genre + "/" + set_name + "/Negative/" + file_redux,
            )
            shutil.copyfile(
                melspectrogram_data + file_redux,
                mel_training_data + genre + "/" + set_name + "/Negative/" + file_redux,
            )
