import numpy as np
import librosa
from scipy.io.wavfile import write
import os
import threading
from concurrent.futures import ProcessPoolExecutor

from tinytag import TinyTag

import librosa.display
import librosa.feature

n_fft = 2048
hop_length = 512
plot_size = (2.5, 2.5)
lock = threading.Lock()


def process_audio(music_file: str, album_no: int) -> None:
    with lock:
        audio = TinyTag.get(music_file)
        track = int(audio.track.split("/")[0])
    y, sr = librosa.load(music_file)
    y = np.float64(y)

    kernel = 64

    y2 = np.copy(y)
    acc = []
    idx = np.arange(y2.shape[0])

    for i in range(kernel):
        addr = (idx[::kernel] + i) % y.shape[0]
        acc.append(y[addr])

    acc = np.array(acc)

    y2 = np.median(acc, axis=0)
    path = os.path.join(
        "./Spectrograms/Processed/AudioFiles", f"{album_no}_{track}.mp3"
    )
    write(path, sr // kernel, y2)


if __name__ == "__main__":
    with open("output.txt", mode="r", encoding="utf-8") as file:
        for folder_pre in file:
            folder = r"{}".format(folder_pre.split("^")[0])
            album_no = int(folder_pre.split("^")[1])
            with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
                for music_file in os.listdir(folder):
                    if (
                        music_file.endswith(".mp3")
                        or music_file.endswith(".flac")
                        or music_file.endswith(".m4a")
                    ):
                        executor.submit(
                            process_audio, folder + "/" + music_file, album_no
                        )
