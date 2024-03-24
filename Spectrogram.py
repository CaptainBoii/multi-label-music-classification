import os
import threading
from concurrent.futures import ProcessPoolExecutor

from tinytag import TinyTag

import librosa.display
import librosa.feature
import matplotlib.pyplot as plt
import numpy as np

n_fft = 2048
hop_length = 512
plot_size = (2.5, 2.5)
lock = threading.Lock()


def process_audio(music_file: str, album_no: int) -> None:
    with lock:
        audio = TinyTag.get(music_file)
        track = int(audio.track.split('/')[0])

    signal, sr = librosa.load(music_file)
    audio_stft = librosa.core.stft(signal, hop_length=hop_length, n_fft=n_fft)
    spectrogram = np.abs(audio_stft)
    log_spectrogram = librosa.amplitude_to_db(spectrogram)
    fig = plt.figure(figsize=plot_size)
    librosa.display.specshow(log_spectrogram, sr=sr, cmap='magma', hop_length=hop_length)
    fig.subplots_adjust(bottom=0, top=1, right=1, left=0)
    spectrogram_path = os.path.join("Spectrograms", f"{album_no}_{track}.png")
    plt.savefig(spectrogram_path)
    plt.close(fig)

    mel_signal = librosa.feature.melspectrogram(y=signal, sr=sr, hop_length=hop_length, n_fft=n_fft)
    mel_spectrogram = np.abs(mel_signal)
    power_to_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
    fig1 = plt.figure(figsize=plot_size)
    librosa.display.specshow(power_to_db, sr=sr, cmap='magma', hop_length=hop_length)
    fig1.subplots_adjust(bottom=0, top=1, right=1, left=0)
    mel_spectrogram_path = os.path.join("MelSpectrograms", f"{album_no}_{track}.png")
    plt.savefig(mel_spectrogram_path)
    plt.close(fig1)


if __name__ == "__main__":
    with open("output.csv", mode='r', encoding="utf-8") as file:
        for folder_pre in file:
            folder = folder_pre.split("^")[0]
            album_no = int(folder_pre.split("^")[1])
            with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
                for music_file in os.listdir(folder):
                    if music_file.endswith(".mp3") or music_file.endswith(".flac") or music_file.endswith(".m4a"):
                        executor.submit(process_audio, folder + "\\" + music_file, album_no)
