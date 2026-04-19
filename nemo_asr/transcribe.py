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
    def __init__(self, model_name: str = "nvidia/parakeet-tdt-0.6b-v3", batch_size = 32) -> None:
        self.model = nemo_asr.models.ASRModel.from_pretrained(model_name = model_name)
        self.model.eval()

        self.batch_size = batch_size
        self.max_duration_allowed = 24 * 60  # 24 mins in secs

        self.cfg = self.model.cfg
        self.sr = self.cfg.sample_rate
        if self.sr != self.cfg.preprocessor.sample_rate:
            raise ValueError("Sample rate mismatch in model configuration")

        self.self_attention_model = self.cfg.encoder.self_attention_model
        self.att_context_size = self.cfg.encoder.att_context_size
    
    def transcribe(self, audio: str | Path | tuple[np.ndarray, int] | list, alignment_level: str = "word") -> tuple[list[str], list[list[dict]]]:
        alignment_level = alignment_level.lower()
        assert alignment_level in {"segment", "word", "token"}, \
            "alignment_level must be one of: 'segment', 'word', or 'token'"
        
        if alignment_level == "token":
            alignment_level = "char"
        
        if not isinstance(audio, list):
            audio = [audio]
        
        audio = [self.load_audio(e) for e in audio]

        max_duration = self.get_max_duration(audio)  # max duration in secs
        if max_duration <= self.max_duration_allowed:
            self.model.change_attention_model(self_attention_model = self.self_attention_model, att_context_size = self.att_context_size)
        else:
            self.model.change_attention_model(self_attention_model = "rel_pos_local_attn", att_context_size = [256, 256])

        results = self.model.transcribe(audio, use_lhotse = False, batch_size = self.batch_size, timestamps = True, verbose = False)
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

    def get_duration(self, audio: np.ndarray | str | Path | tuple[np.ndarray, int]) -> float:
        """Get duration in secs"""
        if isinstance(audio, np.ndarray):
            wav = audio
        else:
            wav = self.load_audio(audio)
        return wav.shape[0] / self.sr

    def get_max_duration(self, audio: list[np.ndarray | str | Path | tuple[np.ndarray, int]]) -> float:
        if not isinstance(audio, list):
            audio = [audio]
        return max(self.get_duration(e) for e in audio)

    def compare_texts(self, src_text: str, tgt_text: str) -> bool:
        src_text = make_ref(src_text)
        tgt_text = make_ref(tgt_text)
        return src_text == tgt_text
