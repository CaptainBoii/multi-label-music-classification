import os
import threading
from concurrent.futures import ProcessPoolExecutor

import librosa.display
import librosa.feature
import matplotlib.pyplot as plt
import numpy as np

n_fft = 2048
hop_length = 512
plot_size = (2.5, 2.5)
lock = threading.Lock()


def process_audio(music_file: str) -> None:
    music_file_path = "./Spectrograms/Processed/AudioFiles/" + music_file
    music_file_save = music_file[:-4]
    signal, sr = librosa.load(music_file_path)
    audio_stft = librosa.core.stft(signal, hop_length=hop_length, n_fft=n_fft)
    spectrogram = np.abs(audio_stft)
    log_spectrogram = librosa.amplitude_to_db(spectrogram)
    fig = plt.figure(figsize=plot_size)
    librosa.display.specshow(
        log_spectrogram, sr=sr, cmap="magma", hop_length=hop_length
    )
    fig.subplots_adjust(bottom=0, top=1, right=1, left=0)
    spectrogram_path = os.path.join(
        "./Spectrograms/Processed/Spectrograms", f"{music_file_save}.png"
    )
    plt.savefig(spectrogram_path)
    plt.close(fig)

    mel_signal = librosa.feature.melspectrogram(
        y=signal, sr=sr, hop_length=hop_length, n_fft=n_fft
    )
    mel_spectrogram = np.abs(mel_signal)
    power_to_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
    fig1 = plt.figure(figsize=plot_size)
    librosa.display.specshow(power_to_db, sr=sr, cmap="magma", hop_length=hop_length)
    fig1.subplots_adjust(bottom=0, top=1, right=1, left=0)
    mel_spectrogram_path = os.path.join(
        "./Spectrograms/Processed/MelSpectrograms", f"{music_file_save}.png"
    )
    plt.savefig(mel_spectrogram_path)
    plt.close(fig1)


if __name__ == "__main__":
    all_files = os.listdir("./Spectrograms/Processed/AudioFiles/")
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        for music_file in all_files:
            executor.submit(process_audio, music_file)
