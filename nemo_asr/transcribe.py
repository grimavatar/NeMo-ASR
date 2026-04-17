import logging; logging.disable(logging.WARNING)

try:
    import nemo.collections.asr as nemo_asr
except ImportError:
    raise ImportError(
        "Missing required dependency for NeMo ASR. "
        "Install NeMo with ASR utilities support:\n"
        "  pip install 'nemo_toolkit[asr]==2.7.2'"
    )


class NeMoASR:
    def __init__(self, model_name: str = "nvidia/parakeet-tdt-0.6b-v3"):
        self.model = nemo_asr.models.ASRModel.from_pretrained(model_name = model_name)
    
    def transcribe(self, audio, alignment_level: str = "word"):
        alignment_level = alignment_level.lower()
        assert alignment_level in {"segment", "word", "token"}, \
            "alignment_level must be one of: 'segment', 'word', or 'token'"
        
        if alignment_level == "token":
            alignment_level = "char"

        results = self.model.transcribe(audio, timestamps=True, use_lhotse=False, verbose=False)
        texts = [e.text.strip() for e in results]
        alignments = [e.timestamp[alignment_level] for e in results]
        return texts, alignments
