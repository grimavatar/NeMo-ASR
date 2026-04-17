# NeMo ASR

NeMo ASR is a simple Python interface for NVIDIA NeMo ASR models. It transcribes audio and extract timestamps at segment, word, or token level with minimal setup.

NeMo ASR is CPU-friendly, running at 2–3× faster than real time.

## Install

```bash
pip install git+https://github.com/grimavatar/NeMo-ASR.git
```

## Example

```python
from pathlib import Path
from nemo_asr import NeMoASR

model = NeMoASR()

audio_paths = list(Path(".").absolute().glob("*wav"))

texts, alignments = model.transcribe(audio_paths)
```

## License

NeMo is licensed under the Apache License 2.0.
