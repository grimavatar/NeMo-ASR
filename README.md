# NeMo ASR

NeMo ASR is a simple Python interface for NVIDIA NeMo ASR models. It transcribes audio and extract timestamps at segment, word, or token level with minimal setup.

NFA is CPU-friendly, running at 1.5–4× faster than real time.

## Install

```bash
pip install git+https://github.com/grimavatar/NeMo-ASR.git
```

## Example

```python
from pathlib import Path
...

audio_paths = [str(e) for e in Path(".").absolute().glob("*wav")]

...
```

## License

NeMo is licensed under the Apache License 2.0.
