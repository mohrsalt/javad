import torch
from soundfile import read as sf_read_audio
from subprocess import CalledProcessError, run
import warnings
import librosa

def load_audio(filename: str, sr: int = 16000, mono: bool = True) -> torch.Tensor:
    """Load audio file with automatic fallback between ffmpeg and soundfile.

    Args:
        filename (str): Path to input audio file.
        sr (int, optional): Target sample rate. Defaults to 16000.
        mono (bool, optional): Convert to mono if True. Defaults to True.

    Returns:
        torch.Tensor: Audio waveform as float32 tensor, scaled to [-1, 1].
            If mono=True, shape is [samples]. Otherwise [samples, channels].

    Raises:
        IOError: If both ffmpeg and soundfile fail to load the audio file.

    Example:
        >>> audio = load_audio("input.wav", sr=16000, mono=True)
        >>> audio.shape
        torch.Size([16000])  # 1 second of mono audio at 16kHz

    Note:
        Tries ffmpeg first, falls back to soundfile if ffmpeg fails.
        Supports common audio formats (wav, mp3, flac).
    """
    try:
        data = load_audio_ffmpeg(filename, sr)
        waveform_16k, _ = librosa.load(filename, sr=16000, mono=True)
        print("Waveform_16k shape: ",waveform_16k.shape)
    except Exception as e:
        try:
            # Fallback to soundfile
            data = sf_read_audio(filename, sr, dtype="float32").to(torch.float32)
            
            
            
        except Exception as sf_error:
            raise IOError(
                f"Failed to load audio file {filename} with both ffmpeg and soundfile. "
                f"ffmpeg error: {e}. Soundfile error: {sf_error}"
            ) from sf_error

    if mono is True and len(data.shape) > 1 and data.shape[1] > 1:
        data = data.mean(axis=1)
    print("Ret data shape: ",data.shape)
    return data


def load_audio_ffmpeg(file: str, sr: int = 16000) -> torch.Tensor:
    """Load and preprocess audio file using ffmpeg.

    Args:
        file (str): Path to input audio file.
        sr (int, optional): Target sample rate. Defaults to 16000.

    Returns:
        torch.Tensor: Audio waveform as mono float32 tensor, scaled to [-1, 1].

    Raises:
        RuntimeError: If ffmpeg fails to process the audio file.

    Example:
        >>> audio = load_audio_ffmpeg("audio.mp3", sr=16000)
        >>> audio.shape
        torch.Size([16000])  # 1 second of audio at 16kHz

    Note:
        Requires ffmpeg executable in system PATH. Automatically handles:
        - Resampling to target sample rate
        - Converting to mono
        - Converting to 16-bit PCM
    """

    # fmt: off
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-threads", "0",
        "-i", file,
        "-f", "s16le",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        "-ar", str(sr),
        "-"
    ]
    # fmt: on
    try:
        out = run(cmd, capture_output=True, check=True).stdout
    except CalledProcessError as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    warnings.filterwarnings("ignore", message="The given buffer is not writable")
    return (
        torch.frombuffer(out, dtype=torch.int16)
        .view(-1)
        .to(torch.float32)
        .div_(32768.0)
    )
