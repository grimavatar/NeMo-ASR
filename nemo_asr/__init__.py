from .utils.nemo_logging import suppress_logging; suppress_logging()

import torch; torch.manual_seed(42)

from .transcribe import NeMoASR
