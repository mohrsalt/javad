import numpy as np
import torch
from javad.main import from_pretrained, MODELINFO, load_checkpoint
from javad.utils import exact_div
from javad.utils import load_mel_filters, log_mel_spectrogram
from types import SimpleNamespace
from typing import List, Tuple, Dict, Union
import warnings


class Pipeline:
    def __init__(
        self,
        model_name: str = "balanced",
        checkpoint: Union[str, None] = None,
        mode: str = "gradual",
        threshold: Union[float, None] = None,
        device: Union[torch.device, str] = torch.device("cpu"),
    ) -> None:
        """
        Initialize the stream pipeline for voice activity detection.
        This class processes audio streams for voice activity detection using various models.

        Args:
            model_name (str, optional): Name of the model to use. Defaults to "balanced" (there are also "tiny" and "precise" options).
            checkpoint (Union[str, None], optional): Path to a custom model checkpoint. If None, uses the default model.
            mode (str, optional): Processing mode - "instant" or "gradual". Defaults to "gradual".
                'instant' mode immediately returns latest predictions, although it may not be as accurate
                as 'gradual' mode which maintains and updates predictions while chunks are moving across buffer.
            threshold (Union[float, None], optional): Detection threshold. If None, uses model's default.
                Defaults to None.
            device (Union[torch.device, str], optional): Device to run computations on.
                Defaults to torch.device("cpu").

        Attributes:
            mode (str): Processing mode.
            config (SimpleNamespace): Configuration parameters including:
                - model_name: Name of the model
                - sample_rate: Audio sample rate
                - fps: Frames per second
                - window_size: Size of processing window in samples
                - window_size_frames: Size of processing window in frames
                - model_output_length: Model output length
                - n_mels: Number of mel frequency bands
                - hop: Hop length
                - threshold: Detection threshold
                - padding_size: Size of padding added to buffer to prevent inaccuracy in
                    spectrograms at the start of the buffer
                - buffer_size: Size of audio buffer
            flags (SimpleNamespace): Processing flags
            audio_buffer (torch.Tensor): Buffer for audio processing
            model: Neural network model for VAD
            mel_filters: Mel-frequency filterbank
            mean (float): Running mean for statistics
            variance (float): Running variance for statistics
            chunk_count (int): Counter for processed chunks
            predictions_storage (dict): Storage for predictions
            frames_tracker (list): Tracker for processed frames
            predicted_intervals (dict): Storage for predicted intervals
            detection_carry (int): Carryover detection counter
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

        self.mode = mode
        modelinfo = MODELINFO[model_name]
        fps = int(exact_div(modelinfo["sample_rate"], modelinfo["hop_length"]))
        # with sample rate = 16000, hop length = 160 -> fps = 100
        # means 1 second of audio is 100 spectrogram frames
        self.config = SimpleNamespace(
            model_name=model_name,
            sample_rate=modelinfo["sample_rate"],
            fps=fps,
            window_size=int(modelinfo["input_length"] * modelinfo["sample_rate"]),
            window_size_frames=int(modelinfo["input_length"] * fps),
            model_output_length=modelinfo["output_length"],
            n_mels=modelinfo["n_mels"],
            hop=modelinfo["hop_length"],
            threshold=(threshold or modelinfo["threshold"]),
        )
        self.flags = SimpleNamespace(input_padded=False)

        self.config.padding_size = modelinfo["n_fft"] // modelinfo["hop_length"]
        self.config.buffer_size = int(
            modelinfo["input_length"] * modelinfo["sample_rate"]
            + self.config.padding_size * modelinfo["hop_length"]
        )

        self.audio_buffer = torch.zeros(
            self.config.buffer_size, dtype=torch.float32
        ).to(device)

        # Preload mel filters
        self.preload_mel_filters(n_mels=self.config.n_mels)

        self.mean = 0.0
        self.variance = 0.0
        self.chunk_count = -1
        self.predictions_storage = {}
        self.frames_tracker = []
        self.predicted_intervals = {}
        self.detection_carry = 0

    def reset(self):
        """Reset the pipeline to initial state."""
        self.audio_buffer.zero_()
        self.mean = 0.0
        self.variance = 0.0
        self.chunk_count = -1
        self.predictions_storage = {}
        self.frames_tracker = []
        self.predicted_intervals = {}
        self.detection_carry = 0
        self.flags.input_padded = False
        return self

    @property
    def device(self) -> torch.device:
        return self.__device

    @device.setter
    def device(self, d: Union[torch.device, str]):
        if isinstance(d, str):
            d = torch.device(d)
        self.__device = d
        self.__model.to(self.device)

    def to(self, device: Union[torch.device, str]) -> "Pipeline":
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

    def update_stats(self, spectrogram: torch.Tensor):
        """
        Update running statistics (mean and standard deviation) of the spectrogram data.
        This method uses Welford's online algorithm to compute running statistics
        of streaming spectrogram data.

        Args:
            spectrogram : torch.Tensor
                Input spectrogram tensor of shape (frequency_bins, time_frames)

        Returns:
            tuple
                A tuple containing:
                - mean (float): Updated running mean of the spectrogram
                - std (torch.Tensor): Updated running standard deviation of the spectrogram
                normalized by total number of frames and frequency bins
        Notes:
            The method tracks the total number of frames processed using self.frames_tracker
            and updates statistics incrementally using Welford's method for numerical stability.
        """
        frames_chunk = self.frames_tracker[-1]
        frames_total = sum(self.frames_tracker)

        spg = spectrogram[:, -frames_chunk:]
        delta = spg.mean() - self.mean
        self.mean += delta * frames_chunk / frames_total
        # Update the variance using Welford's method
        self.variance += torch.sum((spg - self.mean) ** 2)
        return self.mean, torch.sqrt(self.variance / (frames_total * spg.shape[0]))

    def normalize_spectrogram(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """Normalizes the spectrogram using running mean and standard deviation.

        Args:
            spectrogram (torch.Tensor): Input spectrogram tensor to be normalized.
        Returns:
            torch.Tensor: Normalized spectrogram tensor with zero mean and unit variance.
                         If standard deviation is 0, returns original spectrogram unchanged.
        """

        mean, std = self.update_stats(spectrogram)
        if std == 0.0:
            return spectrogram
        return (spectrogram - mean) / std

    def push(
        self, chunk: Union[List, np.ndarray, torch.Tensor]
    ) -> Union[torch.Tensor, Dict]:
        """
        Pushes a chunk of audio data through the model for prediction.
        This method processes audio chunks for prediction by:
        1. Converting input to torch tensor if needed
        2. Padding the chunk if it's not divisible by hop length
        3. Managing a rolling audio buffer
        4. Computing log mel spectrogram
        5. Normalizing the spectrogram
        6. Running prediction
        7. Tracking and aggregating predictions across chunks

        Args:
            chunk : Union[List, np.ndarray, torch.Tensor]
                Audio chunk to process. Can be a list, numpy array or torch tensor.
                Length must not exceed model's window_size.

        Returns:
            Union[torch.Tensor, Dict[int, torch.Tensor]]
                If mode is "instant": Returns tensor of predictions for current chunk
                If mode is "gradual": Returns dict mapping chunk numbers to mean predictions
                across all passes that included that chunk

        Raises:
            ValueError
                If chunk length is larger than model window size
                If non-final chunk length is not divisible by hop length
        """

        # convert chunk to torch.tensor
        if isinstance(chunk, (list, np.ndarray)):
            chunk = torch.tensor(chunk, dtype=torch.float32).to(self.device)

        # if chunk is not divisible by hop, pad with zeroes
        if len(chunk) % self.config.hop != 0:
            if self.flags.input_padded:
                raise ValueError(
                    f"All chunks except last one should have size divisible by hop length {self.config.hop}. Current size {len(chunk)}"
                )
            if not self.flags.input_padded:
                self.flags.input_padded = True

            chunk = torch.nn.functional.pad(
                chunk, (0, self.config.hop - len(chunk) % self.config.hop)
            )

        if len(chunk) > self.config.window_size:
            raise ValueError(
                f"Chunk size {len(chunk)} cannot be larger than model input ({self.config.window_size})"
            )

        self.chunk_count += 1
        self.predictions_storage[self.chunk_count] = []
        self.frames_tracker.append(len(chunk) // self.config.hop)

        # Shift buffer to the left by the chunk size and write new chunk
        self.audio_buffer = torch.roll(self.audio_buffer, -len(chunk), dims=0)
        self.audio_buffer[-len(chunk) :] = chunk

        spectrogram = log_mel_spectrogram(
            audio=self.audio_buffer,
            n_mels=self.config.n_mels,
            mel_filters=self.mel_filters,
            device=self.device,
        )
        # trim padding if needed
        if self.config.padding_size > 0:
            spectrogram = spectrogram[:, self.config.padding_size :]

        # normalize spectrogram with running mean and std
        spectrogram = self.normalize_spectrogram(spectrogram)

        # if we are through initial steps and not every element of the buffer is filled
        # zero spectrogram elements that were produced with zeroes in buffer
        frames_filled = sum(self.frames_tracker)
        if frames_filled < spectrogram.shape[1]:
            spectrogram[:, : spectrogram.shape[1] - frames_filled] = 0.0

        # predict
        with torch.no_grad():
            predictions = self.__model(spectrogram.unsqueeze(0).unsqueeze(0)).flatten()

        # update chunk tracker with predictions per chunk
        accounted_frames = 0
        dispose = []
        for chunk_num in reversed(self.predictions_storage):
            chunk_frames = self.frames_tracker[chunk_num]
            start_idx = max(
                self.config.padding_size,
                len(predictions) - accounted_frames - chunk_frames,
            )
            end_idx = len(predictions) - accounted_frames
            if end_idx < start_idx:
                dispose.append(chunk_num)
                continue
            chunk_predictions = predictions[start_idx:end_idx]
            # if chunk went over left side of the buffer so that
            # predictions are less than chunk_frames, pad with NaNs
            # then when computing mean over all prediction series,
            # ignore NaNs
            if len(chunk_predictions) < chunk_frames:
                chunk_predictions = torch.nn.functional.pad(
                    chunk_predictions,
                    (0, chunk_frames - len(chunk_predictions)),
                    mode="constant",
                    value=float("nan"),
                )
            self.predictions_storage[chunk_num].append(chunk_predictions)
            if self.mode == "instant":
                del self.predictions_storage[chunk_num]
                return chunk_predictions
            accounted_frames += chunk_frames

        # prepare output
        mean_predictions = {}
        for chunk_num in reversed(self.predictions_storage):
            predictions = torch.stack(self.predictions_storage[chunk_num])
            mean = torch.nanmean(predictions, dim=0)
            mean_predictions[chunk_num] = mean

        # delete chunks from tracker that are no longer needed
        for chunk_num in dispose:
            del self.predictions_storage[chunk_num]

        return mean_predictions

    update = push
    logits = push

    def predict(
        self, chunk: Union[List, np.ndarray, torch.Tensor]
    ) -> Union[torch.Tensor, Dict]:
        """
        Predicts whether audio chunks contain speech based on model predictions.

        Args:
            chunk (Union[List, np.ndarray, torch.Tensor]): Input audio chunk to process.
                Can be a list, numpy array or PyTorch tensor.
        Returns:

            Union[torch.Tensor, Dict]:
                If mode is "instant": Returns boolean tensor where True indicates speech was detected
                    (predictions above threshold)
                If mode is "gradual": Returns dictionary mapping chunk numbers to boolean tensors
                    indicating speech detection for each chunk

        Raises:
            ValueError: If input chunk has invalid format or dimensions
        """

        predictions = self.push(chunk)
        if self.mode == "instant":
            return predictions > self.config.threshold
        output = {}
        for chunk_num in predictions:
            output[chunk_num] = predictions[chunk_num] > self.config.threshold
        return output

    def detect(
        self, chunk: Union[List, np.ndarray, torch.Tensor], min_duration: float = 0.0
    ) -> bool:
        """
        Detect speech presence in the provided audio chunk.
        This method analyzes an audio chunk to detect speech segments and determines if any speech segment
        exceeds the minimum duration threshold.

        Args:
            chunk (Union[List, np.ndarray, torch.Tensor]): Audio data chunk to analyze.
            min_duration (float, optional): Minimum duration in seconds for a speech segment to be considered valid.
                Defaults to 0.0. If 0.0, uses the pipeline's default minimum duration.

        Returns:
            bool: True if speech segments longer than min_duration are detected, False otherwise.

        Notes:
            - The method maintains a detection_carry state variable to handle speech segments that span multiple chunks
            - Speech segments are identified by analyzing state changes in model predictions
            - Duration is calculated based on the configured frames per second (fps)
        """
        if self.mode == "gradual":
            warnings.warn(
                '"gradual" mode detected. Switching to "instant" mode for detection.'
            )
            self.mode = "instant"
        predictions = self.predict(chunk)
        # Find state changes
        changes = torch.diff(
            predictions.int(), prepend=torch.tensor([0], device=predictions.device)
        )
        # Get start positions of True sequences
        starts = torch.nonzero(changes == 1).flatten()
        # Get end positions of True sequences
        ends = torch.nonzero(changes == -1).flatten()

        if len(starts) == 0:
            self.detection_carry = 0
            return False

        # Handle case where sequence ends with True
        if len(ends) < len(starts):
            ends = torch.cat(
                [ends, torch.tensor([len(predictions)], device=predictions.device)]
            )

        # Calculate durations in frames
        lengths = ends - starts
        # Convert to time
        durations = lengths.float() / self.config.fps
        # if first duration starts at zero and self.detection_carry > 0
        # add detection_carry
        if starts[0] == 0 and self.detection_carry > 0:
            durations[0] += self.detection_carry
        # if last ends at the end of the chunk, store the carry
        if ends[-1] == len(predictions):
            self.detection_carry = durations[-1]
        else:
            self.detection_carry = 0
        # check if any duration is > min_duration
        if torch.any(durations > min_duration):
            return True

        return False

    def intervals(
        self, chunk: Union[List, np.ndarray, torch.Tensor]
    ) -> Union[List[Tuple], Dict]:
        """
        Process the chunk of data and return intervals based on predictions.
        This method processes input data chunks and returns time intervals based on the prediction mode.
        For 'instant' mode, it directly converts predictions from latest chunk to intervals.
        For 'gradual' mode, it maintains and updates intervals across all chunks.

        Args:
            chunk (Union[List, numpy.ndarray, torch.Tensor]): The chunk of data to process.

        Returns:
            Union[List[Tuple], Dict]: The intervals based on predictions.
        """
        predictions = self.predict(chunk)
        if self.mode == "instant":
            return self.predictions_to_intervals(self.chunk_count, predictions)
        # else if self.mode == 'gradual'
        for chunk_num in predictions:
            ivs = self.predictions_to_intervals(chunk_num, predictions[chunk_num])
            self.predicted_intervals[chunk_num] = ivs
        output = {"last": None, "revised": None, "final": None}
        last_chunk_num = max(self.predicted_intervals)
        output["last"] = self.predicted_intervals[last_chunk_num]

        revised_intervals = [
            self.predicted_intervals[k] for k in predictions if k != last_chunk_num
        ]
        revised_intervals_merged = self.merge_intervals(
            [t for ival in revised_intervals for t in ival]
        )
        output["revised"] = revised_intervals_merged

        final_intervals = [
            self.predicted_intervals[k]
            for k in self.predicted_intervals
            if k not in predictions
        ]
        final_intervals_merged = self.merge_intervals(
            [t for ival in final_intervals for t in ival]
        )
        output["final"] = final_intervals_merged
        return output

    @staticmethod
    def merge_intervals(intervals: List[Tuple]) -> List:
        """
        Merges adjacent intervals in a list of tuples.
        This function takes a list of intervals (start, end) and merges any overlapping
        intervals into a single interval. Two intervals are considered overlapping if
        the start of one interval is within 0.01 of the end of another interval.

        Args:
            intervals (List[Tuple]): List of tuples where each tuple contains
                start and end points of an interval.

        Returns:
            List: A new list containing merged intervals with no overlaps.
        """

        if len(intervals) == 0:
            return []
        intervals.sort(key=lambda x: x[0])
        merged = [intervals[0]]
        for current in intervals:
            previous = merged[-1]
            if current[0] <= previous[1]:
                previous = (previous[0], max(previous[1], current[1]))
                merged[-1] = previous
            else:
                merged.append(current)
        return merged

    def predictions_to_intervals(
        self, chunk_num: int, predictions: torch.Tensor
    ) -> List[Tuple[float, float]]:
        """
        Convert binary predictions tensor into a list of time intervals.

        Args:
            chunk_num (int): Index of the current chunk being processed
            predictions (torch.Tensor): Binary tensor containing predictions (0s and 1s)
                indicating presence/absence of target signal

        Returns:
            List[Tuple[float, float]]: List of time intervals (start_time, end_time) where
                target signal is present. Times are in seconds relative to start of recording.
        """
        start_frame = sum(self.frames_tracker[:chunk_num])
        offset_time = start_frame / self.config.fps
        scale = 1.0 / self.config.fps

        if predictions.all():
            return [(offset_time, offset_time + (len(predictions)) * scale)]
        elif not predictions.any():
            return []

        changes = torch.diff(predictions.int())
        change_points = torch.nonzero(changes).flatten()
        if predictions[0]:
            change_points = torch.cat(
                [torch.tensor([0], device=predictions.device), change_points]
            )
        if predictions[-1]:
            change_points = torch.cat(
                [
                    change_points,
                    torch.tensor([len(predictions)], device=predictions.device),
                ]
            )
        pairs = change_points.reshape(-1, 2)

        return [
            (offset_time + pair[0].item() * scale, offset_time + pair[1].item() * scale)
            for pair in pairs
        ]

    def __repr__(self) -> str:
        window_size_sec = self.config.window_size_frames / self.config.fps
        return (
            f"JaVAD [stream] Pipeline(\n"
            f"    model name: {self.config.model_name!r},\n"
            f"    mode      : {self.mode!r},\n"
            f"    threshold : {self.config.threshold!r}\n"
            f"    device    : {self.device!r},\n"
            f"    ══[INFO]═══════════\n"
            f"    output, timespan: {(window_size_sec/self.config.model_output_length)!r}s,\n"
            f"    max_chunk_size  : {self.config.window_size_frames!r},\n"
            f"    window size     : {window_size_sec!r}s,\n"
            f")"
        )
