import torch
import librosa

def load_audio(filename: str, sr: int = 32000, mono: bool = True) -> torch.Tensor:

    waveform_32k, _ = librosa.load(filename, sr=sr, mono=mono)

    waveform2=torch.from_numpy(waveform_32k)

    return waveform2


