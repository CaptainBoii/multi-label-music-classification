import glob
import random
import shutil

genres = ['Ambient', 'Classical', 'Dance', 'Electronic', 'Experimental', 'Folk', 'Hip-Hop', 'Industrial & Noise',
          'Jazz', 'Metal', 'Pop', 'Psychedelia', 'R&B', 'Rock', 'Singer-Songwriter']

all_files = glob.glob('Spectrograms/Spectrograms/*.png')

for genre in genres:
    positive_files = []
    with open('Labels/' + genre + '.txt', 'r') as file:
        for line in file:
            positive_files += glob.glob('Spectrograms/Spectrograms/' + line.strip() + '_*.png')
    random.shuffle(positive_files)
    k = int(len(positive_files) / 10)
    data_positive = {
        "Test": positive_files[:k],
        "Valid": positive_files[k:2 * k],
        "Train": positive_files[2*k:]
    }

    negative_files = list(set(all_files) - set(positive_files))
    random.shuffle(negative_files)
    j = int(len(negative_files) / 10)
    data_negative = {
        "Test": negative_files[:j],
        "Valid": negative_files[j:2*j],
        "Train": negative_files[2*j:]
    }
    for set_name in ["Test", "Train", "Valid"]:
        for file in data_positive[set_name]:
            file_redux = file.split("/Spectrograms/")[1]
            shutil.copyfile("Spectrograms/Spectrograms/" + file_redux,
                            "Data/Experiment_1_2/Spectrograms/" + genre + "/" + set_name + "/Positive/" + file_redux)
            shutil.copyfile("Spectrograms/MelSpectrograms/" + file_redux,
                            "Data/Experiment_1_2/MelSpectrograms/" + genre + "/" + set_name + "/Positive/" + file_redux)

        for file in data_negative[set_name]:
            file_redux = file.split("/Spectrograms/")[1]
            shutil.copyfile("Spectrograms/Spectrograms/" + file_redux,
                            "Data/Experiment_1_2/Spectrograms/" + genre + "/" + set_name + "/Negative/" + file_redux)
            shutil.copyfile("Spectrograms/MelSpectrograms/" + file_redux,
                            "Data/Experiment_1_2/MelSpectrograms/" + genre + "/" + set_name + "/Negative/" + file_redux)
