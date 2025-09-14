import numpy as np
import librosa
np.savez_compressed(
            "mel_filters.npz",
            mel_64=librosa.filters.mel(sr=32000, n_fft=800, n_mels=64),
            mel_80=librosa.filters.mel(sr=32000, n_fft=800, n_mels=80),
            mel_128=librosa.filters.mel(sr=32000, n_fft=800, n_mels=128),
        )
