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
    "R&B",
    "Rock",
    "Singer-Songwriter",
    "Punk",
]

test = "./Data/Test/"
input = "./Spectrograms/"

types = [
    "Default/",
    "Processed/",
    "Scaled/",
]

if __name__ == "__main__":
    with open("./Data/test_labels.csv") as file:
        for line in file:
            image, genres_list = line.strip().split(",")
            genres_ = genres_list.split("_")
            for i, genre in enumerate(genres):
                bol = "/Positive/" if str(i) in genres_ else "/Negative/"
                for typ in types:
                    for spect_type in ["Spectrograms/", "MelSpectrograms/"]:
                        shutil.copy(
                            input + typ + spect_type + image,
                            test + typ + spect_type + genre + bol + image,
                        )
