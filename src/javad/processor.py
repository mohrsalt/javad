import numpy as np
import torch
from javad.src.javad.main import initialize, from_pretrained, MODELINFO, load_checkpoint
from javad.src.javad.utils import exact_div, load_mel_filters, log_mel_spectrogram
import warnings
from typing import List, Tuple, Union
from torch.utils.data import TensorDataset, DataLoader
from types import SimpleNamespace


class Processor:
    def __init__(
        self,
        model_name: str = "balanced",
        checkpoint: Union[str, None] = None,
        step: Union[float, None] = None,
        onset: float = 0.0,
        offset: float = 0.0,
        padding: tuple = (0.0, 0.0),
        min_duration: float = 0.3,
        min_silence: float = 0.3,
        batch_size: int = 32,
        num_workers: int = 0,
        threshold: Union[float, None] = None,
        device: Union[torch.device, str] = torch.device("cpu"),
    ) -> None:
        """Initialize the Speech Processing module.
        This class handles speech detection and processing using neural networks.

        Args:
            model_name (str, optional): Name of the model to use. Defaults to "balanced". Available values are: "tiny" and "precise"
            checkpoint (Union[str, None], optional): Path to a custom model checkpoint. If None, uses the default model. Defaults to None.
            step (Union[float, None], optional): Step size for sliding windows. If None, uses model input length (windows do not overlap). Defaults to None.
            onset (float, optional): Onset threshold for speech detection. Defaults to 0.0.
            offset (float, optional): Offset threshold for speech detection. Defaults to 0.0.
            padding (tuple, optional): Padding added to detected speech segments (before, after). Defaults to (0.0, 0.0).
            min_duration (float, optional): Minimum duration for detected speech segments. Defaults to 0.0.
            min_silence (float, optional): Minimum duration of silence between segments. Defaults to 0.0.
            batch_size (int, optional): Batch size for processing. Defaults to 32.
            num_workers (int, optional): Number of worker processes. Defaults to 0.
            threshold (Union[float, None], optional): Detection threshold. If None, uses model default. Defaults to None.
            device (Union[torch.device, str], optional): Device to run the model on. Defaults to CPU.

        Returns:
            None
        """

        self.__device = (
            device if isinstance(device, torch.device) else torch.device(device)
        )
        # Initialize model
        if checkpoint is not None:
            cpt = load_checkpoint(checkpoint, is_asset=False)
            model_name = cpt["model_name"]
            self.__model = from_pretrained(checkpoint=checkpoint).to(self.__device)
        else:
            self.__model = from_pretrained(name=model_name).to(self.__device)
        self.__model.eval()

        modelinfo = MODELINFO[model_name]
        fps = int(exact_div(modelinfo["sample_rate"], modelinfo["hop_length"]))
        # with sample rate = 16000, hop length = 160 -> fps = 100
        # means 1 second of audio is 100 spectrogram frames
        self.config = SimpleNamespace(
            model_name=model_name,
            batch_size=batch_size,
            num_workers=num_workers,
            fps=fps,
            step=(step or modelinfo["input_length"]),
            sample_rate=modelinfo["sample_rate"],
            window_size_frames=int(modelinfo["input_length"] * fps),
            model_output_length=modelinfo["output_length"],
            n_mels=modelinfo["n_mels"],
            threshold=(threshold or modelinfo["threshold"]),
            onset=onset,
            offset=offset,
            padding=padding,
            min_duration=min_duration,
            min_silence=min_silence,
        )
        assert (
            self.config.step <= modelinfo["input_length"]
        ), f"Step size {self.config.step}s cannot exceed model input length {modelinfo['input_length']}s"
        # Preload mel filters
        self.preload_mel_filters(n_mels=self.config.n_mels)

    @property
    def device(self) -> torch.device:
        return self.__device

    @device.setter
    def device(self, d: Union[torch.device, str]):
        if isinstance(d, str):
            d = torch.device(d)
        self.__device = d
        self.__model.to(self.device)

    def to(self, device: Union[torch.device, str]) -> "Processor":
        self.device = device
        return self

    def preload_mel_filters(self, n_mels: int) -> torch.Tensor:
        """Load mel filter bank matrices for a given number of mel bins."""
        if self.__device == torch.device("mps"):
            self.mel_filters = (
                load_mel_filters(n_mels=n_mels).to(torch.float32).to(self.__device)
            )
        else:
            self.mel_filters = load_mel_filters(n_mels=n_mels).to(self.__device)

    def get_min_input(self) -> int:
        """Get the minimum input duration in samples."""
        return int(
            self.config.sample_rate * self.config.window_size_frames / self.config.fps
        )

    def logits(
        self, audio: Union[np.ndarray, torch.Tensor], step: Union[float, None] = None
    ) -> torch.Tensor:
        """Process audio data to generate logit predictions using a trained model.
        This method converts audio input to a spectrogram, normalizes it, splits it into
        overlapping windows, and processes these windows through the model to generate
        predictions. The predictions from overlapping windows are averaged to produce
        the final output.

        Args:
            audio (Union[np.ndarray, torch.Tensor]): Input audio data as either numpy array
                or PyTorch tensor.
            step (Optional[float]): Step size in seconds for sliding window. If None,
                uses the configured default step size. Defaults to None.

        Returns:
            torch.Tensor: Averaged model predictions across all windows, with shape
                matching the temporal dimension of the input spectrogram.

        Raises:
            Warning: If the standard deviation of the spectrogram is zero, indicating
                potential issues with the input audio.

        Notes:
            - The input audio is converted to a log-mel spectrogram before processing
            - The spectrogram is normalized using mean and standard deviation
            - Processing is done in batches to handle resources efficiently
            - Overlapping predictions are averaged to smooth transitions
        """
        if isinstance(audio, np.ndarray):
            audio = torch.tensor(audio)
        if len(audio) < self.get_min_input():
            raise ValueError(
                f"Input audio is too short. Minimum length is {self.get_min_input()} samples."
            )

        # Convert audio to spectrogram
        spectrogram = log_mel_spectrogram(
            audio=audio,
            n_mels=self.config.n_mels,
            mel_filters=self.mel_filters,
            device=self.__device,
        )
        # Normalize the spectrogram
        mean = torch.mean(spectrogram)
        std = torch.std(spectrogram)
        if std == 0:
            std = 1e-6
            warnings.warn(
                "The standard deviation of the spectrogram is zero. Check the input audio."
            )
        spectrogram = (spectrogram - mean) / std

        if step is None:
            step = self.config.step
        step_spg = int(step * self.config.fps)

        # Get complete slices using unfold
        slices = spectrogram.unfold(1, self.config.window_size_frames, step_spg)
        # Pad the last slice if necessary
        last_start = slices.size(1) * step_spg
        last_slice = spectrogram[:, last_start:]
        last_slice_length = self.config.window_size_frames - last_slice.shape[1]
        if last_slice_length > 0:
            last_slice = torch.nn.functional.pad(last_slice, (0, last_slice_length))
            slices = torch.cat([slices, last_slice.unsqueeze(1)], dim=1)
        slices = torch.permute(slices, (1, 0, 2))
        if self.config.num_workers > 0:
            slices = slices.cpu()

        dataset = TensorDataset(slices)
        dataloader = DataLoader(
            batch_size=self.config.batch_size,
            dataset=dataset,
            shuffle=False,
            num_workers=self.config.num_workers,
        )

        output_length = spectrogram.shape[1]

        # Create accumulator tensors
        accumulated_predictions = torch.zeros((output_length)).to(self.__device)
        counts = torch.zeros(output_length).to(self.__device)

        current_position = 0
        for batch in dataloader:
            with torch.no_grad():
                inputs = batch[0].unsqueeze(1).to(self.device)
                outputs = self.__model(inputs)

                for output in outputs:
                    # Add prediction to accumulator
                    end_pos = current_position + self.config.window_size_frames
                    if end_pos > output_length:
                        output = output[: output_length - current_position]
                    accumulated_predictions[current_position:end_pos] += output
                    counts[current_position:end_pos] += 1
                    current_position += step_spg

        # Average by dividing by counts
        averaged_predictions = accumulated_predictions / counts
        return averaged_predictions

    def predict(
        self, audio: Union[np.ndarray, torch.Tensor], step: Union[float, None] = None
    ) -> torch.Tensor:
        """Predict voice activity from audio signal. Converts logits (values) to boolean predictions."""
        logits = self.logits(audio=audio, step=step)
        predictions = logits > self.config.threshold  # Convert to boolean predictions
        return predictions

    def intervals(
        self, audio: Union[np.ndarray, torch.Tensor], step: Union[float, None] = None
    ) -> List[Tuple[float, float]]:
        """Process audio to find voice activity intervals.

        This method analyzes the audio signal to detect voice activity and returns a list of
        time intervals where speech is present. The process includes:
        1. Getting voice activity predictions
        2. Converting predictions to initial intervals
        3. Filtering out intervals that are too short
        4. Padding remaining intervals
        5. Handling onset/offset boundaries
        6. Merging intervals that are too close together

        Args:
            audio (Union[np.ndarray, torch.Tensor]): Input audio signal
            step (Union[float, None], optional): Step size for processing audio.
                If None, uses default configuration. Defaults to None.

        Returns:
            List[Tuple[float, float]]: List of intervals where voice activity is detected.
            Each interval is a tuple of (start_time, end_time) in seconds.

        Note:
            The returned intervals are affected by several configuration parameters:
            - min_duration: Minimum valid interval duration
            - padding: Amount of padding to add to intervals
            - onset: Start time to ignore
            - offset: End time to ignore
            - min_silence: Minimum silence duration between intervals
        """
        predictions = self.predict(audio=audio, step=step)
        intervals = self.predictions_to_intervals(predictions, self.config.fps)

        filtered_intervals = []
        for start, end in intervals:
            # Remove intervals that are too short
            if end - start < self.config.min_duration:
                continue
            else:
                filtered_intervals.append((start, end))

        # Pad intervals and check that intervals do not overlap with onset and offset
        audio_length = len(audio) / self.config.sample_rate
        padded_intervals = []
        if filtered_intervals:
            for start, end in filtered_intervals:
                # Add padding
                padded_start = max(0, start - self.config.padding[0])
                padded_end = min(audio_length, end + self.config.padding[1])

                # Trim at onset boundary
                if self.config.onset > 0:
                    if padded_end <= self.config.onset:
                        continue  # Skip if entirely in onset
                    padded_start = max(padded_start, self.config.onset)

                # Trim at offset boundary
                if self.config.offset > 0:
                    offset_start = audio_length - self.config.offset
                    if padded_start >= offset_start:
                        continue  # Skip if entirely in offset
                    padded_end = min(padded_end, offset_start)

                padded_intervals.append((padded_start, padded_end))

        # Merge intervals that are too close
        merged_intervals = []
        if padded_intervals:
            if self.config.min_silence > 0:
                current_interval = padded_intervals[0]
                for interval in padded_intervals[1:]:
                    if interval[0] - current_interval[1] < self.config.min_silence:
                        current_interval = (
                            current_interval[0],
                            max(current_interval[1], interval[1]),
                        )
                    else:
                        merged_intervals.append(current_interval)
                        current_interval = interval
                merged_intervals.append(current_interval)
            else:
                merged_intervals = padded_intervals
        return merged_intervals

    def __repr__(self) -> str:

        window_size_sec = self.config.window_size_frames / self.config.fps
        return (
            f"JaVAD Processor(\n"
            f"    model name : {self.config.model_name!r},\n"
            f"    threshold  : {self.config.threshold!r}\n"
            f"    step       : {self.config.step!r}s,\n"
            f"    batch_size : {self.config.batch_size!r},\n"
            f"    num_workers: {self.config.num_workers!r},\n"
            f"    device     : {self.device!r},\n"
            f"    ══[INFO]═══════════\n"
            f"    duration, min   : {self.config.min_duration!r}s,\n"
            f"    silence, min    : {self.config.min_silence!r}s,\n"
            f"    onset/offset    : {self.config.onset!r}s/{self.config.offset!r}s,\n"
            f"    speech padings  : {self.config.padding!r}s,\n"
            f"    window size     : {window_size_sec!r}s,\n"
            f"    output, timespan: {(window_size_sec/self.config.model_output_length)!r}s,\n"
            f")"
        )

    @staticmethod
    def predictions_to_intervals(
        bool_array: torch.Tensor, fps: int
    ) -> List[Tuple[float, float]]:
        """
        Converts a boolean tensor array of predictions into a list of time intervals.
        This function identifies contiguous sequences of True values in the boolean array
        and converts them into time intervals based on the given frames per second (fps).

        Args:
            bool_array (torch.Tensor): A 1D boolean tensor where True values represent active segments
            fps (int): Frames per second, used to convert frame indices to time in seconds

        Returns:
            List[Tuple[float, float]]: A list of tuples where each tuple contains
                (start_time, end_time) in seconds for each detected interval
        """

        # Get indices where values change
        changes = torch.where(bool_array[:-1] != bool_array[1:])[0] + 1

        # Add start and end indices
        if bool_array[0]:
            changes = torch.cat([torch.tensor([0]).to(changes.device), changes])
        if bool_array[-1]:
            changes = torch.cat(
                [changes, torch.tensor([len(bool_array)]).to(changes.device)]
            )

        # Convert to intervals
        intervals = []
        for start, end in zip(changes[::2], changes[1::2]):
            # Convert frame indices to seconds (each frame is 10ms)
            start_time = float(start) / fps
            end_time = float(end) / fps
            intervals.append((start_time, end_time))
        return intervals
