try:
    import nemo.collections.asr as nemo_asr
except ImportError:
    raise ImportError(
        "Missing required dependency for NeMo ASR. "
        "Install NeMo with ASR utilities support:\n"
        "  pip install 'nemo_toolkit[asr]==2.7.2'"
    )

from pathlib import Path

import numpy as np
import soundfile as sf
from .utils.resample import resample
from .utils.utils import make_ref

class NeMoASR:
    def __init__(self, model_name: str = "nvidia/parakeet-tdt-0.6b-v3", sr: int = 16000) -> tuple[list[str], list[list[dict]]]:
        self.model = nemo_asr.models.ASRModel.from_pretrained(model_name = model_name)
        self.sr = int(sr)
    
    def transcribe(self, audio: str | Path | tuple[np.ndarray, int] | list, alignment_level: str = "word"):
        alignment_level = alignment_level.lower()
        assert alignment_level in {"segment", "word", "token"}, \
            "alignment_level must be one of: 'segment', 'word', or 'token'"
        
        if alignment_level == "token":
            alignment_level = "char"
        
        if not isinstance(audio, list):
            audio = [audio]
        
        audio = [self.load_audio(e) for e in audio]

        results = self.model.transcribe(audio, timestamps=True, use_lhotse=False, verbose=False)
        texts = [e.text.strip() for e in results]
        alignments = [e.timestamp[alignment_level] for e in results]
        return texts, alignments
    
    def load_audio(self, audio: str | Path | tuple[np.ndarray, int]) -> np.ndarray:
        if isinstance(audio, (str, Path)):
            wav, sr = sf.read(audio, dtype = "float32", always_2d = False)
        else:
            wav, sr = audio
            if not (isinstance(wav, np.ndarray) and isinstance(sr, int)):
                raise ValueError(f"'audio' must be a str, Path, or tuple of (np.ndarray, int), but got ({type(wav).__name__}, {type(sr).__name__})")
        if sr != self.sr:
            wav = resample(y = wav, orig_sr = int(sr), target_sr = self.sr)
        return wav

    def compare_texts(self, src_text: str, tgt_text: str) -> bool:
        src_text = make_ref(src_text)
        tgt_text = make_ref(tgt_text)
        return src_text == tgt_text
