import os
from concurrent.futures import ProcessPoolExecutor
import librosa.display
import librosa.feature
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import gc

n_fft = 2048
hop_length = 512
plot_size = (2.5, 2.5)
audiofile_path = "./Spectrograms/Processed/AudioFiles/"


def process_audio(music_file: str) -> None:
    music_file_path = audiofile_path + music_file
    music_file_save = music_file[:-4]
    signal, sr = librosa.load(music_file_path)
    audio_stft = librosa.core.stft(signal, hop_length=hop_length, n_fft=n_fft)
    spectrogram = np.abs(audio_stft)
    log_spectrogram = librosa.amplitude_to_db(spectrogram)
    fig = plt.figure(figsize=plot_size)
    #plt.ylim(top=32)
    librosa.display.specshow(
        log_spectrogram, sr=sr, cmap="magma", hop_length=hop_length
    )
    fig.subplots_adjust(bottom=0, top=1, right=1, left=0)
    spectrogram_path = os.path.join(
        "./Spectrograms/Processed/Default/Spectrograms", f"{music_file_save}.png"
    )
    plt.savefig(spectrogram_path)
    plt.close(fig)

    mel_signal = librosa.feature.melspectrogram(
        y=signal, sr=sr, hop_length=hop_length, n_fft=n_fft
    )
    mel_spectrogram = np.abs(mel_signal)
    power_to_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
    fig1 = plt.figure(figsize=plot_size)
    #plt.ylim(top=16)
    librosa.display.specshow(power_to_db, sr=sr, cmap="magma", hop_length=hop_length)
    fig1.subplots_adjust(bottom=0, top=1, right=1, left=0)
    mel_spectrogram_path = os.path.join(
        "./Spectrograms/Processed/Default/MelSpectrograms", f"{music_file_save}.png"
    )
    plt.savefig(mel_spectrogram_path)
    plt.close(fig1)
    
    del signal
    del sr
    del audio_stft
    del spectrogram
    del log_spectrogram
    del mel_signal
    del mel_spectrogram
    del power_to_db
    
    gc.collect()


if __name__ == "__main__":
    matplotlib.use('Agg')
    all_files = os.listdir(audiofile_path)
    with ProcessPoolExecutor(max_workers=8) as executor:
        for music_file in all_files:
            executor.submit(process_audio, music_file)
