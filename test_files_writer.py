import shutil
import csv

mel_path = "Spectrograms/MelSpectrograms/"
spec_path = "Spectrograms/Spectrograms/"
mel_save = "Data/Test/MelSpectrograms/"
spec_save = "Data/Test/Spectrograms/"

with open("Data/test_labels.csv") as csv_file:
    reader = csv.reader(csv_file)
    test_dict = dict(reader)
    test_keys = list(test_dict.keys())
    for filename in test_keys:
        shutil.copyfile(spec_path + filename, spec_save + filename)
        shutil.copyfile(mel_path + filename, mel_save + filename)
