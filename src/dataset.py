"""Datasets y dataloaders para analisis de sentimiento."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src import config

try:
    import torch
    from torch.utils.data import DataLoader, Dataset

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - depende del entorno local.
    torch = None
    DataLoader = None

    class Dataset:  # type: ignore[no-redef]
        """Fallback minimo para permitir importar el modulo sin PyTorch."""

        pass

    TORCH_AVAILABLE = False


class SentimentDataset(Dataset):
    """Dataset de PyTorch para textos tokenizados y etiquetas."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        tokenizer: Any,
        text_column: str = "text_clean",
        label_column: str = "label_id",
        max_length: int = 128,
    ) -> None:
        require_torch()
        missing_columns = [
            column
            for column in (text_column, label_column)
            if column not in dataframe.columns
        ]
        if missing_columns:
            raise ValueError(
                "Faltan columnas requeridas en el DataFrame: "
                f"{missing_columns}. Columnas disponibles: {list(dataframe.columns)}"
            )

        self.dataframe = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.text_column = text_column
        self.label_column = label_column
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.dataframe.iloc[idx]
        encoded = self.tokenizer.encode_plus(
            row[self.text_column],
            max_length=self.max_length,
            padding=True,
            truncation=True,
        )
        label_id = int(row[self.label_column])

        return {
            "input_ids": torch.tensor(encoded["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(encoded["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(label_id, dtype=torch.long),
        }


def load_processed_split(
    split_name: str,
    processed_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Carga train, val o test desde data/processed/."""
    if split_name not in {"train", "val", "test"}:
        raise ValueError("split_name debe ser 'train', 'val' o 'test'.")

    base_dir = Path(processed_dir) if processed_dir is not None else config.PROCESSED_DATA_DIR
    split_path = base_dir / f"{split_name}.csv"
    if not split_path.exists():
        raise FileNotFoundError(
            f"No existe {split_path}. Ejecuta primero python -m src.prepare_data."
        )

    return pd.read_csv(split_path)


def create_dataloader(
    dataset: SentimentDataset,
    batch_size: int = 32,
    shuffle: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    """Crea un DataLoader de PyTorch."""
    require_torch()
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )


def create_dataloaders(
    tokenizer: Any,
    batch_size: int = 32,
    max_length: int = 128,
    processed_dir: str | Path | None = None,
    num_workers: int = 0,
    include_test: bool = True,
) -> dict[str, DataLoader]:
    """Crea dataloaders para train, val y test."""
    require_torch()
    train_df = load_processed_split("train", processed_dir=processed_dir)
    val_df = load_processed_split("val", processed_dir=processed_dir)

    train_dataset = SentimentDataset(
        train_df,
        tokenizer=tokenizer,
        max_length=max_length,
    )
    val_dataset = SentimentDataset(
        val_df,
        tokenizer=tokenizer,
        max_length=max_length,
    )

    dataloaders = {
        "train": create_dataloader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        ),
        "val": create_dataloader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
    }

    if include_test:
        test_df = load_processed_split("test", processed_dir=processed_dir)
        test_dataset = SentimentDataset(
            test_df,
            tokenizer=tokenizer,
            max_length=max_length,
        )
        dataloaders["test"] = create_dataloader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

    return dataloaders


def require_torch() -> None:
    """Valida disponibilidad de PyTorch con un mensaje claro."""
    if not TORCH_AVAILABLE:
        raise RuntimeError(
            "PyTorch no esta instalado en este entorno. Instala dependencias con "
            "pip install -r requirements.txt antes de usar SentimentDataset."
        )
