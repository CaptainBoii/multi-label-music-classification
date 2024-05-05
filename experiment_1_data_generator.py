import csv
import glob
import random
import shutil

spectrogram_data = "Spectrograms/Spectrograms/"
melspectrogram_data = "Spectrograms/MelSpectrograms/"
training_data = "Data/Experiment_1_2/Spectrograms/Training/"
mel_training_data = "Data/Experiment_1_2/MelSpectrograms/Training/"

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

with open("Data/test_labels.csv") as csv_file:
    reader = csv.reader(csv_file)
    test_dict = dict(reader)

    test_keys = list(test_dict.keys())
    test_files = list(map(lambda t_file: spectrogram_data + t_file, test_keys))
    all_files = list(set(all_files) - set(test_files))

for file in test_files:
    file_redux = file.split("/Spectrograms/")[1]
    shutil.copyfile(spectrogram_data + file_redux, training_data + file_redux)
    shutil.copyfile(melspectrogram_data + file_redux, mel_training_data + file_redux)


for genre in genres:
    positive_files = []
    with open("Labels/" + genre + ".txt", "r") as file:
        for line in file:
            positive_files += glob.glob(spectrogram_data + line.strip() + "_*.png")

    positive_files = list(set(positive_files) - set(test_files))
    random.shuffle(positive_files)
    k = int(len(positive_files) / 10)
    data_positive = {"Valid": positive_files[:k], "Train": positive_files[k:]}

    negative_files = list(set(all_files) - set(positive_files))
    random.shuffle(negative_files)
    data_negative = {"Valid": negative_files[:k], "Train": negative_files[k:]}
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
