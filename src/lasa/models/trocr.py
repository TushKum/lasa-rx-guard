"""TrOCR loading and batched inference.

The wrapper exists mainly to centralise one footgun: a
``VisionEncoderDecoderModel`` loaded from a checkpoint does not necessarily come
with ``decoder_start_token_id`` / ``pad_token_id`` set on ``model.config``. When
they are missing, training runs without error and produces fluent garbage, and
generation silently starts from the wrong token. Setting them explicitly at load
time is the difference between a working fine-tune and a mystifying one.
"""

from __future__ import annotations

import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

DEFAULT_MODEL = "microsoft/trocr-base-handwritten"
SMALL_MODEL = "microsoft/trocr-small-handwritten"


def auto_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_trocr(model_name: str = DEFAULT_MODEL, device: str | torch.device = "auto",
               max_length: int = 24, num_beams: int = 4):
    """Return ``(model, processor)`` with generation config made explicit."""
    dev = auto_device(device) if isinstance(device, str) else device
    processor = TrOCRProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name)

    tok = processor.tokenizer
    model.config.decoder_start_token_id = (
        tok.cls_token_id if tok.cls_token_id is not None else tok.bos_token_id
    )
    model.config.pad_token_id = tok.pad_token_id
    model.config.eos_token_id = tok.sep_token_id or tok.eos_token_id
    model.config.vocab_size = model.config.decoder.vocab_size

    # Drug names are short; a tight cap keeps beam-search evaluation affordable.
    model.generation_config.max_length = max_length
    model.generation_config.num_beams = num_beams
    model.generation_config.early_stopping = True
    model.generation_config.decoder_start_token_id = model.config.decoder_start_token_id
    model.generation_config.pad_token_id = model.config.pad_token_id
    model.generation_config.eos_token_id = model.config.eos_token_id
    # MUST stay 0: real drug names repeat bigrams ("cocaine", "acetazolamide"),
    # so blocking repeated n-grams would inject errors.
    model.generation_config.no_repeat_ngram_size = 0

    if model.config.decoder_start_token_id is None:
        raise ValueError(
            f"{model_name} has no usable decoder_start_token_id; training would "
            f"silently produce garbage."
        )
    return model.to(dev), processor


@torch.no_grad()
def predict(model, processor, images, device: str | torch.device = "auto",
            num_beams: int = 4, batch_size: int = 16) -> list[str]:
    """``images``: list of RGB numpy arrays. Returns decoded strings."""
    dev = auto_device(device) if isinstance(device, str) else device
    model.eval()
    out: list[str] = []
    for i in range(0, len(images), batch_size):
        chunk = images[i:i + batch_size]
        pv = processor(images=chunk, return_tensors="pt").pixel_values.to(dev)
        ids = model.generate(pv, num_beams=num_beams)
        out.extend(processor.batch_decode(ids, skip_special_tokens=True))
    return [s.strip().lower() for s in out]
