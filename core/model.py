"""Model wrapper: loads a Hugging Face text classifier and times every inference."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Protocol

DEFAULT_MODEL_ID = os.getenv("MODEL_ID", "juliodourado1/bertimbau-legal-judgment-ptbr")


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float
    scores: dict[str, float]
    latency_ms: float
    model_version: str


class Classifier(Protocol):
    model_version: str
    labels: list[str]

    def predict(self, text: str) -> Prediction: ...


def head_tail_ids(ids: list[int], max_length: int = 512, head: int = 128) -> list[int]:
    """Keep the first `head` and last `max_length - head - 2` tokens (room for [CLS]/[SEP]).

    In an ementa the decisive sentences sit at the end, so right-truncation would drop them.
    """
    budget = max_length - 2
    return ids if len(ids) <= budget else ids[:head] + ids[-(budget - head):]


class HFClassifier:
    """Thin wrapper over a `transformers` sequence-classification model."""

    def __init__(self, model_id: str = DEFAULT_MODEL_ID, max_length: int = 512) -> None:
        import torch
        from huggingface_hub import model_info
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id).eval()
        self.max_length = max_length
        self.labels = [self.model.config.id2label[i] for i in range(self.model.config.num_labels)]
        try:
            sha = model_info(model_id).sha[:7]
        except Exception:  # offline / local path
            sha = "local"
        self.model_version = f"{model_id}@{sha}"

    def encode(self, text: str):
        """Head+tail truncation, identical to training (see the model card)."""
        ids = head_tail_ids(self.tokenizer(text, add_special_tokens=False)["input_ids"], self.max_length)
        ids = [self.tokenizer.cls_token_id, *ids, self.tokenizer.sep_token_id]
        return {"input_ids": self._torch.tensor([ids]), "attention_mask": self._torch.ones(1, len(ids), dtype=self._torch.long)}

    def predict(self, text: str) -> Prediction:
        t0 = time.perf_counter()
        enc = self.encode(text)
        with self._torch.inference_mode():
            probs = self.model(**enc).logits.softmax(-1)[0].tolist()
        latency_ms = (time.perf_counter() - t0) * 1000
        scores = dict(zip(self.labels, probs))
        label = max(scores, key=scores.get)
        return Prediction(label, scores[label], scores, latency_ms, self.model_version)
