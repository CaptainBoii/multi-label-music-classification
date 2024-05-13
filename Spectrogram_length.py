import os
from concurrent.futures import ProcessPoolExecutor

from PIL import Image


def convert_argb(length: int) -> tuple:
    if length >= 1024:
        return 255, 255, 255, 0
    elif length >= 768:
        return 255, 255, 255, 255 - (length % 768)
    elif length >= 512:
        return 255, 255, length % 512, 255
    elif length >= 256:
        return 255, length % 256, 0, 255
    else:
        return length, 0, 0, 255


def write_files(song_file: str, argb: tuple) -> None:
    with Image.open("Spectrograms\\" + song_file) as im:
        im.putpixel((0, 0), argb)
        im.save("Spectrograms\\" + song_file, "PNG")
    with Image.open("MelSpectrograms\\" + song_file) as im:
        im.putpixel((0, 0), argb)
        im.save("MelSpectrograms\\" + song_file, "PNG")


if __name__ == "__main__":
    with open("lengths.txt", mode="r", encoding="utf-8") as file:
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            for songs_pre in file:
                print(songs_pre)
                song_file = songs_pre.split("^")[0]
                argb = convert_argb(int(songs_pre.split("^")[1]))
                executor.submit(write_files, song_file, argb)
