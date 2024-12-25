# JaVAD: Just Another Voice Activity Detector

JaVAD is a state-of-the-art Voice Activity Detection package, lightweight and fast, built on PyTorch with minimal dependencies. Core functionality (without audio loading) requires only NumPy and PyTorch, with no registration, tokens, or installation of large unnecessary packages. While it is built using sliding windows over mel spectrograms, it supports streaming. You can also export results to RTTM, CSV, or TextGrid.

There are three models:
- **tiny**: 0.64s window, optimal for quickest voice detection
- **balanced**: 1.92s window, fastest while small
- **precise**: 3.84s window with extra DirectionalAlignment layer and 2 additional transformer layers for best accuracy.

*All models use mono audio with a sample rate of 16,000 Hz.*

## Comparison

- For evaluation, Google's AVA Speech dataset was used, or, to be accurate, only clips that are still available (74 in total). Since AVA Speech has only 15 minutes from each clip labeled, 18.5 hours of audio in total were used.  
- For evaluation purposes, JaVAD was trained on a custom, manually labelled dataset using a separate, different collection of YouTube clips. Production models were trained on all available data.

| Model                  | Precision | Recall | F1 Score   | AUROC      | Time, GPU<br>Nvidia 3090 | Time, CPU<br>Ryzen 3900XT |
|------------------------|-----------|--------|------------|------------|------------|------------|
| Nvidia NEMO            | 0.7676    | 0.9526 | 0.8502     | 0.9201     | 26.24s     | **56.94s** |
| WebRTC (via py-webrtc) | 0.6099    | 0.9454 | 0.7415     | -¹         | -²         | 59.85s     |
| Google Speechbrain     | 0.8213    | 0.8534 | 0.8370     | 0.8961     | 1371.00s   | 1981.40s   |
| Pyannote               | 0.9173    | 0.8463 | 0.8804     | 0.9495     | 75.49s     | 823.19s    |
| Silero                 | 0.9678    | 0.6503 | 0.9050     | 0.9169     | 830.27s³   | 695.58s    |
| JaVAD tiny⁴*           | 0.9263    | 0.8846 | 0.8961     | 0.9550     | 22.32s     | 476.93s    |
| JaVAD balanced*        | 0.9284    | 0.8938 |   0.9108   | 0.9642     | **16.38s** | 220.00s    |
| JaVAD precise*         | 0.9359    | 0.8980 | **0.9166** | **0.9696** | 18.58s     | 236.61s    |


¹ *WebRTC does not return logits* ² *WebRTC via py-webrtc can be run only on CPU*  
³ *Silero JIT model is slower on GPU, and ONNX model cannot be run on GPU.*  
⁴ *Tiny model is the slowest here due to the smaller window size of 0.64s. It is best applicable for immediate speech detection in the streaming pipeline.*  
**For information about training dataset see text above the table*

![ROCs](ROCs.png)


## Installation

### Requirements

- Python 3.8+
- PyTorch 2.0.0+
- NumPy 1.20.0+
- Optional: `soundfile` for loading audio and simplified processing

### Install via pip

If you already have audio loaded and just want to process it with minimum dependencies:
```bash
pip install javad 
```

If you want to load audio:
```bash
pip install javad[extras]  # with audio loading 
# if you're using zsh, add quotes
pip install 'javad[extras]'
```

## Documentation

https://javad.readthedocs.io/en/latest/


## Usage

### Basic Usage (if installed with [extras]), single file/CPU:

```python
from javad.extras import get_speech_intervals

intervals = get_speech_intervals("path/to/audio.wav")
print(intervals)
```

### Usage via Processor class, single file/CUDA[if available]:
```python
import torch
from javad import Processor
from javad.extras import load_audio

# Load audio file
audio = load_audio("path/to/audio.wav")

# Initialize Processor with default 'balanced' model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = Processor(device=device)
print(processor)

# Process audio
# Get logits
logits = processor.logits(audio).cpu().numpy() 
print(logits)
# Get boolean predictions based on threshold
predictions = processor.predict(audio).cpu().numpy() 
print(predictions)
# Get speech intervals
intervals = processor.intervals(audio) 
print(intervals)
```
You can increase accuracy by specifying the step size for the sliding window. The smaller the step, the longer it takes to compute and average predictions, resulting in a more accurate outcome.


### Stream Processing, stream/CUDA[if available]:
```python
import torch
from javad.stream import Pipeline
from javad.extras import load_audio

# Initialize pipeline
pipeline = Pipeline()  # by default, Pipeline uses 'balanced' model
pipeline.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
print(pipeline)

# Load audio file
audio = load_audio("path/to/audio.wav")

# Process audio in chunks
chunk_size = int(pipeline.config.sample_rate * 0.5)  # 0.5-second chunks
for i in range(0, len(audio), chunk_size):
    audio_chunk = audio[i : i + chunk_size]
    predictions = pipeline.intervals(audio_chunk)
    print(predictions)
```

There are two modes for streaming: `instant` and `gradual`. The `instant` mode returns results only for the current chunk pushed into the pipeline, while the `gradual` mode updates and averages predictions while the chunk is within the audio buffer. For example, with a chunk size of 0.25s and the balanced model's window size of 1.92s, it will provide 8 updates for that chunk.

TL;DR: Use `instant` mode for the fastest response, or `gradual` mode for the most accurate results in stream mode.


### Instant detection
```python
import torch
from javad.stream import Pipeline
from javad.extras import load_audio

# Initialize pipeline
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pipeline = Pipeline(device=device) # by default, Pipeline uses 'balanced' model

# Generate chunk of audio
audio = load_audio("path/to/audio.wav")
chunk_size = int(pipeline.config.sample_rate * 0.5)  # 0.5-second chunks
audio_chunk = audio[:chunk_size]

# Process and detect speech once per stream
bool_prediction = pipeline.detect(audio_chunk)
print(bool_prediction)

# Reset Pipeline for new stream
pipeline.reset()
```

## License

This project is licensed under the MIT License.

## Citation

If you use this package in your research, please cite it as follows:

> @misc{JaVAD, author = {Sergey Skrebnev}, title = {JaVAD: Just Another Voice Activity Detector}, year = {2024}, publisher = {GitHub}, journal = {GitHub repository}, howpublished = {\url{https://github.com/skrbnv/javad}}, }

Alternatively, you can use the following BibTeX entry:

> @software{JaVAD, author = {Sergey Skrebnev}, title = {JaVAD: Just Another Voice Activity Detector}, year = {2024}, url = {https://github.com/skrbnv/javad}, }

