try:
    from nemo.collections.asr.models import ASRModel
except ImportError:
    raise ImportError(
        "Missing required dependency for NeMo ASR. "
        "Install NeMo with ASR utilities support:\n"
        "  'pip install nemo_toolkit[asr]==2.7.2'"
    )

from pathlib import Path

import numpy as np
from huggingface_hub import snapshot_download

from grim_modal_tools.audio.utils import load_audio, get_duration
from grim_modal_tools.text.evaluations import compare_texts, align_to_source


NEMO_MODEL_ID = "nvidia/parakeet-tdt-0.6b-v3"


class NeMoASR:
    def __init__(self, model_name: str = NEMO_MODEL_ID, batch_size = 32) -> None:
        self._load_model(model_name)

        self.batch_size = batch_size
        self.max_duration_allowed = 24 * 60  # 24 mins in secs

        self.cfg = self.model.cfg
        self.sr = self.cfg.sample_rate
        if self.sr != self.cfg.preprocessor.sample_rate:
            raise ValueError("Sample rate mismatch in model configuration")

        self.self_attention_model = self.cfg.encoder.self_attention_model
        self.att_context_size = self.cfg.encoder.att_context_size

    @classmethod
    def _download_model(self, model_name: str = NEMO_MODEL_ID, pattern = "*.nemo") -> None:
        model_path = snapshot_download(model_name, allow_patterns = pattern)
        return next(Path(model_path).glob(pattern), None)

    def _load_model(self, model_name: str = NEMO_MODEL_ID) -> None:
        model_path = self._download_model(model_name)
        self.model = ASRModel.restore_from(model_path)
        self.model.eval()

    def transcribe(self, audio: str | Path | tuple[np.ndarray, int] | list, alignment_level: str = "word") -> tuple[list[str], list[list[dict]]] | tuple[None, None]:
        if not audio:
            return None, None

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
        return load_audio(audio, self.sr)[0]

    def get_duration(self, audio: str | Path | np.ndarray | tuple[np.ndarray, int]) -> float:
        """Get duration in secs"""
        if isinstance(audio, np.ndarray):
            audio = audio, self.sr
        return get_duration(audio)

    def get_max_duration(self, audio: list[str | Path | np.ndarray | tuple[np.ndarray, int]]) -> float:
        if not isinstance(audio, list):
            audio = [audio]
        return max(self.get_duration(e) for e in audio)

    def compare_texts(self, src_text: str, tgt_text: str) -> bool:
        return compare_texts(src_text, tgt_text)

    def align_to_source(src_text: str, alignment: list[dict]) -> list[dict] | None:
        return align_to_source(src_text, alignment)
