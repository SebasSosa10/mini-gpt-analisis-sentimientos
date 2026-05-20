"""Pruebas unitarias para prediccion de sentimiento."""

from __future__ import annotations

from src.predict import get_id_to_label, print_prediction
from src.tokenizer import SimpleTokenizer


def test_get_id_to_label_predict() -> None:
    """Valida inversion de label_mapping en prediccion."""
    label_mapping = {"negative": 0, "positive": 1}

    assert get_id_to_label(label_mapping) == {0: "negative", 1: "positive"}


def test_print_prediction() -> None:
    """Valida que print_prediction no falle con resultado simulado."""
    result = {
        "text": "The food was delicious",
        "predicted_label": "positive",
        "predicted_label_id": 1,
        "confidence": 0.95,
        "probabilities": {"negative": 0.05, "positive": 0.95},
        "logits": [0.1, 2.0],
    }

    print_prediction(result, show_probs=True, show_logits=True)


def test_predict_sentiment_with_tiny_model() -> None:
    """Valida inferencia con un modelo pequeno si PyTorch esta disponible."""
    import pytest

    torch = pytest.importorskip("torch")
    from src.model import MiniGPTForSentiment
    from src.predict import predict_sentiment

    tokenizer = SimpleTokenizer(max_vocab_size=50, min_freq=1)
    tokenizer.fit(["good food", "bad food"])
    model = MiniGPTForSentiment(
        vocab_size=tokenizer.vocab_size,
        max_context_length=16,
        embed_dim=32,
        num_heads=4,
        num_layers=1,
        num_classes=2,
        dropout=0.1,
    )

    result = predict_sentiment(
        text="good food",
        model=model,
        tokenizer=tokenizer,
        id_to_label={0: "negative", 1: "positive"},
        device=torch.device("cpu"),
        max_length=16,
    )

    assert result["predicted_label"] in {"negative", "positive"}
    assert 0 <= result["confidence"] <= 1
    assert set(result["probabilities"]) == {"negative", "positive"}
