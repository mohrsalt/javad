import torch
from javad import Processor
from javad.extras import load_audio
from typing import List, Tuple, Union


def get_speech_intervals(
    audio: str,
    model_name: str = "balanced",
    device: Union[torch.device, str] = torch.device("cpu"),
) -> List[Tuple[float, float]]:
    """
    Process an audio file to detect voice activity intervals.

    Args:
        audio (str): path to audio file.
        model_name (str, optional): Model type to use ("tiny", "balanced" or "precise"). Defaults to "balanced".
        device (Union[torch.device, str], optional): Device to run inference on. Defaults to CPU.

    Returns:
        List[Tuple[float, float]]: List of voice intervals as (start_time, end_time) pairs in seconds.

    Example:
        >>> intervals = get_speech_intervals(audio)
        >>> intervals
        [(0.5, 1.2), (1.8, 3.4)]
    """
    if isinstance(device, str):
        device = torch.device(device)
    pipeline = Processor(model_name=model_name, device=device)
    data = load_audio(audio)
    return pipeline.intervals(data)
