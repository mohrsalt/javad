import numpy as np
import pytest
import torch
from javad.processor import Processor
from javad.extras import load_audio


@pytest.fixture(scope="module")
def sample_audio():
    audio_data = load_audio("tests/files/ranger-bill.flac")
    if isinstance(audio_data, np.ndarray):
        audio_data = torch.from_numpy(audio_data)
    return audio_data.float()  # Convert to float32 tensor


class TestProcessor:
    def test_init_default(self):
        processor = Processor()
        assert processor.config.model_name == "balanced"
        assert processor.device == torch.device("cpu")

    @pytest.mark.parametrize("model_name", ["tiny", "balanced", "precise"])
    def test_init_custom_model(self, model_name, sample_audio):
        processor = Processor(model_name=model_name)
        print(type(sample_audio), sample_audio)
        # should not raise error
        _ = processor.intervals(sample_audio)

    @pytest.mark.parametrize("device", ["cpu", "cuda", "mps"])
    def test_init_custom_device(self, device, sample_audio):
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        if device == "mps" and not torch.mps.is_available():
            pytest.skip("MPS not available")
        processor = Processor(device=device)
        # should not raise error
        _ = processor.intervals(sample_audio)

    def test_logits(self, sample_audio):
        processor = Processor()
        logits = processor.logits(sample_audio)
        assert isinstance(logits, torch.Tensor)

    def test_predict(self, sample_audio):
        processor = Processor()
        predictions = processor.predict(sample_audio)
        assert isinstance(predictions, torch.Tensor)
        assert all(p in [0, 1] for p in predictions)

    def test_intervals(self, sample_audio):
        processor = Processor()
        intervals = processor.intervals(sample_audio)
        assert isinstance(intervals, list)
        assert all(isinstance(i, tuple) and len(i) == 2 for i in intervals)
        assert all(isinstance(t, float) for i in intervals for t in i)

    def test_intervals_with_threshold(self, sample_audio):
        processor = Processor(threshold=0.7)
        intervals = processor.intervals(sample_audio)
        assert isinstance(intervals, list)
        assert all(isinstance(i, tuple) and len(i) == 2 for i in intervals)
        assert all(isinstance(t, float) for i in intervals for t in i)

    def test_intervals_with_num_workers(self, sample_audio):
        processor = Processor(num_workers=2)
        intervals = processor.intervals(sample_audio)
        assert isinstance(intervals, list)
        assert all(isinstance(i, tuple) and len(i) == 2 for i in intervals)
        assert all(isinstance(t, float) for i in intervals for t in i)

    def test_intervals_with_custom_step(self, sample_audio):
        processor = Processor(step=0.5)
        intervals = processor.intervals(sample_audio)
        assert isinstance(intervals, list)
        assert all(isinstance(i, tuple) and len(i) == 2 for i in intervals)
        assert all(isinstance(t, float) for i in intervals for t in i)

    def test_intervals_with_custom_onset_offset(self, sample_audio):
        processor = Processor(onset=0.5, offset=1.5)
        intervals = processor.intervals(sample_audio)
        assert isinstance(intervals, list)
        assert all(isinstance(i, tuple) and len(i) == 2 for i in intervals)
        assert all(isinstance(t, float) for i in intervals for t in i)

    def test_intervals_with_custom_min_duration(self, sample_audio):
        processor = Processor(min_duration=0.5)
        intervals = processor.intervals(sample_audio)
        assert isinstance(intervals, list)
        assert all(isinstance(i, tuple) and len(i) == 2 for i in intervals)
        assert all(isinstance(t, float) for i in intervals for t in i)

    def test_intervals_with_custom_min_silence(self, sample_audio):
        processor = Processor(min_silence=0.5)
        intervals = processor.intervals(sample_audio)
        assert isinstance(intervals, list)
        assert all(isinstance(i, tuple) and len(i) == 2 for i in intervals)
        assert all(isinstance(t, float) for i in intervals for t in i)
