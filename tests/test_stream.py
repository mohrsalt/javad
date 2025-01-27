import pytest
import torch
import numpy as np
from javad.stream import Pipeline
from javad.extras import load_audio


@pytest.fixture(scope="module")
def sample_audio():
    audio_data = load_audio("tests/files/ranger-bill.flac")
    if isinstance(audio_data, np.ndarray):
        audio_data = torch.from_numpy(audio_data)
    return audio_data.float()  # Convert to float32 tensor


class TestPipeline:
    def test_init_default(self):
        pipeline = Pipeline()
        assert pipeline.mode == "gradual"
        assert pipeline.config.model_name == "balanced"

    def test_push_invalid_chunk_size(self, sample_audio):
        pipeline = Pipeline()
        assert len(sample_audio) > pipeline.config.window_size
        with pytest.raises(ValueError):
            pipeline.push(sample_audio)  # Should fail as larger than window_size

    def test_normalization(self):
        pipeline = Pipeline()
        spec = torch.randn(80, 100)  # Mock spectrogram
        pipeline.frames_tracker.append(spec.shape[1])
        normalized = pipeline.normalize_spectrogram(spec)
        assert normalized.shape == spec.shape
        assert torch.isclose(normalized.mean(), torch.tensor(0.0), atol=1e-6)

    def test_intervals_instant_mode(self, sample_audio):
        pipeline = Pipeline(mode="instant")
        chunk = sample_audio[:1600]  # 0.1s chunk
        results = pipeline.intervals(chunk)
        assert isinstance(results, list)

    def test_intervals_gradual_mode(self, sample_audio):
        pipeline = Pipeline(mode="gradual")
        chunk = sample_audio[:1600]
        results = pipeline.intervals(chunk)
        assert isinstance(results, dict)
        assert "last" in results
        assert "revised" in results
        assert "final" in results

    @pytest.mark.parametrize("model_name", ["tiny", "balanced", "precise"])
    def test_different_models(self, model_name, sample_audio):
        pipeline = Pipeline(model_name=model_name)
        chunk = sample_audio[:1600]
        _ = pipeline.push(chunk)  # Should not raise error

    @pytest.mark.parametrize(
        "checkpoint",
        [
            "src/javad/assets/balanced.pt",
            "src/javad/assets/tiny.pt",
            "src/javad/assets/precise.pt",
        ],
    )
    def test_init_custom_checkpoint(self, checkpoint, sample_audio):
        pipeline = Pipeline(checkpoint=checkpoint)
        chunk = sample_audio[:1600]
        _ = pipeline.push(chunk)  # Should not raise error

    @pytest.mark.parametrize("device", ["cpu", "cuda", "mps"])
    def test_device(self, device, sample_audio):
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        if device == "mps" and not torch.mps.is_available():
            pytest.skip("MPS not available")
        pipeline = Pipeline(device=device)
        chunk = sample_audio[:1600]
        _ = pipeline.push(chunk)  # Should not raise error

    @pytest.mark.parametrize("mode", ["instant", "gradual"])
    def test_predict(self, mode, sample_audio):
        pipeline = Pipeline(mode=mode)
        chunk = sample_audio[:1600]
        results = pipeline.predict(chunk)
        if mode == "instant":
            predictions = results
        else:
            predictions = results[0]
        assert isinstance(predictions, torch.Tensor)
