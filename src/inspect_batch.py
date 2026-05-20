"""Inspeccion rapida de tokenizer, Dataset y DataLoader."""

from __future__ import annotations

import argparse
import sys

from src import config
from src.dataset import create_dataloaders
from src.tokenizer import SimpleTokenizer


def main() -> int:
    """Carga un batch de entrenamiento y muestra sus dimensiones."""
    parser = argparse.ArgumentParser(
        description="Inspecciona un batch tokenizado sin entrenar el modelo."
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=config.MAX_CONTEXT_LENGTH)
    args = parser.parse_args()

    try:
        if not config.TOKENIZER_PATH.exists():
            raise FileNotFoundError(
                f"No existe {config.TOKENIZER_PATH}. Ejecuta primero "
                "python -m src.build_vocab."
            )

        tokenizer = SimpleTokenizer.load(config.TOKENIZER_PATH)
        dataloaders = create_dataloaders(
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            max_length=args.max_length,
            processed_dir=config.PROCESSED_DATA_DIR,
        )
        batch = next(iter(dataloaders["train"]))
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]
    first_input_ids = input_ids[0].tolist()

    print("Batch inspeccionado correctamente.")
    print(f"- input_ids shape: {tuple(input_ids.shape)}")
    print(f"- attention_mask shape: {tuple(attention_mask.shape)}")
    print(f"- labels shape: {tuple(labels.shape)}")
    print(f"- primeros labels: {labels[: min(10, len(labels))].tolist()}")
    print(f"- ejemplo decodificado: {tokenizer.decode(first_input_ids)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
