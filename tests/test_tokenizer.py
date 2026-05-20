"""Pruebas unitarias del tokenizer y Dataset."""

from __future__ import annotations

import pandas as pd
import pytest

from src.tokenizer import SimpleTokenizer


def test_simple_tokenizer_fit_and_special_tokens() -> None:
    """Valida ajuste basico y tokens especiales."""
    tokenizer = SimpleTokenizer(max_vocab_size=20, min_freq=1)
    tokenizer.fit(["The food was delicious!", "The food was bad."])

    assert tokenizer.pad_token_id == 0
    assert tokenizer.unk_token_id == 1
    assert tokenizer.bos_token_id == 2
    assert tokenizer.eos_token_id == 3
    assert tokenizer.vocab_size > 4


def test_encode_plus_padding_and_attention_mask() -> None:
    """Valida longitud fija, padding y attention_mask."""
    tokenizer = SimpleTokenizer(max_vocab_size=20, min_freq=1)
    tokenizer.fit(["the food was delicious"])

    encoded = tokenizer.encode_plus("the food", max_length=8)

    assert len(encoded["input_ids"]) == 8
    assert len(encoded["attention_mask"]) == 8
    assert encoded["input_ids"][-1] == tokenizer.pad_token_id
    assert encoded["attention_mask"][-1] == 0


def test_encode_decode_roundtrip_is_legible() -> None:
    """Valida que decode reconstruya texto legible."""
    tokenizer = SimpleTokenizer(max_vocab_size=20, min_freq=1)
    tokenizer.fit(["the food was delicious"])

    token_ids = tokenizer.encode("The food was delicious!")
    decoded = tokenizer.decode(token_ids)

    assert "the food was delicious" in decoded


def test_tokenizer_save_and_load(tmp_path) -> None:
    """Valida que save/load conserve vocabulario."""
    tokenizer = SimpleTokenizer(max_vocab_size=20, min_freq=1)
    tokenizer.fit(["good food", "bad food"])

    path = tmp_path / "tokenizer.json"
    tokenizer.save(path)
    loaded = SimpleTokenizer.load(path)

    assert loaded.get_vocab() == tokenizer.get_vocab()
    assert loaded.encode("good food") == tokenizer.encode("good food")


def test_sentiment_dataset_item_shapes() -> None:
    """Valida que el Dataset retorne tensores con shapes esperadas."""
    pytest.importorskip("torch")
    from src.dataset import SentimentDataset

    dataframe = pd.DataFrame(
        {
            "text_clean": ["good food", "bad food"],
            "label_id": [1, 0],
        }
    )
    tokenizer = SimpleTokenizer(max_vocab_size=20, min_freq=1).fit(
        dataframe["text_clean"]
    )
    dataset = SentimentDataset(dataframe, tokenizer, max_length=8)
    item = dataset[0]

    assert len(dataset) == 2
    assert item["input_ids"].shape[0] == 8
    assert item["attention_mask"].shape[0] == 8
    assert item["labels"].shape == ()
