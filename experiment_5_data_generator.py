import os
import shutil

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

for spect in ["Spectrograms", "MelSpectrograms"]:
    for genre in genres:
        for mode in ["Train", "Valid"]:
            for class_ in ["Positive", "Negative"]:
                path = os.path.join("./Data/Experiment_3/", spect, genre, mode, class_)
                target = os.path.join(
                    "./Data/Experiment_5/", spect, genre, mode, class_
                )
                files = os.listdir(path)
                for file in files:
                    shutil.copy(
                        os.path.join(
                            "Spectrograms", "Processed", "Scaled", spect, file
                        ),
                        os.path.join(target, file),
                    )
