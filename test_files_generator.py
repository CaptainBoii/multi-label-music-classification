import csv
import glob
import random
from math import ceil, floor

genres = ['Ambient', 'Classical', 'Dance', 'Electronic', 'Experimental', 'Folk', 'Hip-Hop', 'Industrial & Noise',
          'Jazz', 'Metal', 'Pop', 'Psychedelia', 'R&B', 'Rock', 'Singer-Songwriter']
genres_files = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], []]

all_files = glob.glob('Spectrograms/Spectrograms/*.png')
test_count = 1801

unique_dict = {}

unique_classes = []


def stringify_list(lst: list) -> str:
    result = ""
    for element in lst:
        if result != "": result += "_"
        result += str(element)
    return result


for i, genre in enumerate(genres):
    with open('Labels/' + genre + '.txt', 'r') as file:
        for line in file:
            genres_files[i].append(line.strip())

for k in range(1, 1468):
    genre_classes = []
    for i in range(len(genres_files)):
        if str(k) in genres_files[i]:
            genre_classes.append(i)
    code = stringify_list(genre_classes)
    if code in unique_dict:
        unique_dict[code] += (glob.glob('Spectrograms/Spectrograms/' + str(k) + '_*.png'))
    else:
        unique_dict[code] = glob.glob('Spectrograms/Spectrograms/' + str(k) + '_*.png')

sum = 0
small = 0
other_files = []
other_labels = []
final_dict = {}

for key, values in unique_dict.items():
    if len(values) < 10:
        other_labels.append(key)
        other_files += values
    else:
        val = len(values) / 10
        val_final = ceil(val) if val - int(val) > 0.5 else floor(val)
        files = values
        random.shuffle(files)
        for i in range(val_final):
            final_dict[files[i]] = key

random.shuffle(other_files)
for i in range(11):
    for key in other_labels:
        if other_files[i] in unique_dict[key]:
            final_dict[other_files[i]] = key

with open("Data/Experiment_1_2/test_labels.csv", 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=",")
    for key, value in final_dict.items():
        writer.writerow([key[26:], value])
