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

if __name__ == "__main__":
    with open("./Data/test_labels.csv") as file:
        for line in file:
            image, genres_list = line.strip().split(",")
            genres_ = genres_list.split("_")
            for i, genre in enumerate(genres):
                if str(i) in genres_:
                    shutil.copy(
                        "./Data/Test/Spectrograms/" + image,
                        "./Data/Test/Spectrograms/" + genre + "/Positive/" + image,
                    )
                    shutil.copy(
                        "./Data/Test/MelSpectrograms/" + image,
                        "./Data/Test/MelSpectrograms/" + genre + "/Positive/" + image,
                    )

                else:
                    shutil.copy(
                        "./Data/Test/Spectrograms/" + image,
                        "./Data/Test/Spectrograms/" + genre + "/Negative/" + image,
                    )
                    shutil.copy(
                        "./Data/Test/MelSpectrograms/" + image,
                        "./Data/Test/MelSpectrograms/" + genre + "/Negative/" + image,
                    )
