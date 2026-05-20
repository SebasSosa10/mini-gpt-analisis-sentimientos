"""Pruebas unitarias para utilidades de entrenamiento."""

from __future__ import annotations

import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from src.model import MiniGPTForSentiment
from src.train import compute_class_weights, save_checkpoint, train_one_epoch


def test_compute_class_weights() -> None:
    """La clase minoritaria debe recibir mayor peso."""
    train_df = pd.DataFrame({"label_id": [0, 0, 0, 0, 1]})

    weights = compute_class_weights(train_df, num_classes=2)

    assert weights.shape[0] == 2
    assert weights[1] > weights[0]


def test_save_checkpoint(tmp_path) -> None:
    """Valida guardado minimo de checkpoint."""
    model = MiniGPTForSentiment(
        vocab_size=50,
        max_context_length=8,
        embed_dim=16,
        num_heads=4,
        num_layers=1,
        num_classes=2,
        dropout=0.1,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    checkpoint_path = tmp_path / "checkpoint.pt"

    save_checkpoint(
        path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        epoch=1,
        history={"epochs": []},
        model_config=model.get_model_config(),
        label_mapping={"negative": 0, "positive": 1},
        tokenizer_path="tokenizer.json",
        metric_name="val_accuracy",
        metric_value=0.5,
    )

    assert checkpoint_path.exists()


def test_train_one_epoch_tiny() -> None:
    """Ejecuta un entrenamiento minimo sobre datos sinteticos."""
    dataset = torch.utils.data.TensorDataset(
        torch.randint(0, 50, (4, 8)),
        torch.ones(4, 8, dtype=torch.long),
        torch.tensor([0, 1, 0, 1], dtype=torch.long),
    )

    def collate_fn(batch):
        input_ids, attention_mask, labels = zip(*batch)
        return {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_mask),
            "labels": torch.stack(labels),
        }

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=collate_fn,
    )
    model = MiniGPTForSentiment(
        vocab_size=50,
        max_context_length=8,
        embed_dim=16,
        num_heads=4,
        num_layers=1,
        num_classes=2,
        dropout=0.1,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    metrics = train_one_epoch(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        criterion=criterion,
        device=torch.device("cpu"),
        grad_clip=1.0,
        max_batches=1,
    )

    assert "loss" in metrics
    assert "accuracy" in metrics
    assert metrics["loss"] >= 0
    assert 0 <= metrics["accuracy"] <= 1
