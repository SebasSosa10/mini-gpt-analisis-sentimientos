"""Pruebas unitarias de la arquitectura Mini-GPT."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from src.model import MiniGPTForSentiment


def build_small_model(pooling: str = "mean") -> MiniGPTForSentiment:
    """Crea un modelo pequeno para pruebas rapidas."""
    return MiniGPTForSentiment(
        vocab_size=100,
        max_context_length=16,
        embed_dim=36,
        num_heads=6,
        num_layers=3,
        num_classes=2,
        dropout=0.1,
        pooling=pooling,
    )


def test_forward_without_labels() -> None:
    """Valida forward sin loss."""
    model = build_small_model()
    input_ids = torch.randint(0, 100, (2, 16))
    attention_mask = torch.ones(2, 16, dtype=torch.long)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    assert outputs["logits"].shape == (2, 2)
    assert outputs["loss"] is None


def test_forward_with_labels() -> None:
    """Valida forward con CrossEntropyLoss."""
    model = build_small_model()
    input_ids = torch.randint(0, 100, (2, 16))
    attention_mask = torch.ones(2, 16, dtype=torch.long)
    labels = torch.tensor([0, 1], dtype=torch.long)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    assert outputs["logits"].shape == (2, 2)
    assert outputs["loss"] is not None


def test_mean_pooling() -> None:
    """Valida que mean pooling funcione correctamente."""
    model = build_small_model(pooling="mean")
    input_ids = torch.randint(1, 100, (2, 16))
    attention_mask = torch.ones(2, 16, dtype=torch.long)
    attention_mask[1, 8:] = 0

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    assert outputs["logits"].shape == (2, 2)


def test_last_token_pooling() -> None:
    """Valida que last-token pooling siga funcionando."""
    model = build_small_model(pooling="last")
    input_ids = torch.randint(1, 100, (2, 16))
    attention_mask = torch.ones(2, 16, dtype=torch.long)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    assert outputs["logits"].shape == (2, 2)


def test_invalid_pooling_raises_error() -> None:
    """Valida error con pooling invalido."""
    with pytest.raises(ValueError, match="pooling"):
        MiniGPTForSentiment(
            vocab_size=100,
            max_context_length=16,
            embed_dim=36,
            num_heads=6,
            num_layers=3,
            num_classes=2,
            dropout=0.1,
            pooling="invalid",
        )


def test_count_parameters_positive() -> None:
    """Valida conteo de parametros."""
    model = build_small_model()
    counts = model.count_parameters()

    assert counts["total"] > 0
    assert counts["trainable"] > 0
    assert counts["trainable"] <= counts["total"]


def test_sequence_longer_than_context_raises_error() -> None:
    """Valida error claro si la secuencia supera el contexto."""
    model = build_small_model()
    input_ids = torch.randint(0, 100, (2, 17))

    with pytest.raises(ValueError, match="supera max_context_length"):
        model(input_ids=input_ids)


def test_embed_dim_must_be_divisible_by_num_heads() -> None:
    """Valida restriccion de dimensiones de atencion multi-cabeza."""
    with pytest.raises(ValueError, match="divisible"):
        MiniGPTForSentiment(
            vocab_size=100,
            max_context_length=16,
            embed_dim=30,
            num_heads=4,
            num_layers=2,
            num_classes=2,
            dropout=0.1,
        )
