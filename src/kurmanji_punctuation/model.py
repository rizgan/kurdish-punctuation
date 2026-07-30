"""Model loading helpers."""

from __future__ import annotations

from transformers import AutoConfig, AutoModelForTokenClassification, AutoTokenizer

from .constants import ID2LABEL, LABEL2ID, LABELS


def load_tokenizer(model_name: str):
    return AutoTokenizer.from_pretrained(model_name)


def load_model(model_name_or_path: str, num_labels: int | None = None):
    num_labels = num_labels or len(LABELS)
    config = AutoConfig.from_pretrained(
        model_name_or_path,
        num_labels=num_labels,
        id2label={str(k) if isinstance(k, int) else k: v for k, v in ID2LABEL.items()},
        label2id=LABEL2ID,
    )
    # Ensure int keys in id2label for HF
    config.id2label = ID2LABEL
    config.label2id = LABEL2ID
    model = AutoModelForTokenClassification.from_pretrained(
        model_name_or_path,
        config=config,
        ignore_mismatched_sizes=True,
    )
    return model


def bf16_supported() -> bool:
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        # Ampere (8.0+) and newer support BF16 well.
        major, _minor = torch.cuda.get_device_capability(0)
        return major >= 8
    except Exception:
        return False
