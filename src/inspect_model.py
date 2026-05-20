"""Inspeccion del forward pass del Mini-GPT sin entrenamiento."""

from __future__ import annotations

import argparse
import sys

from src import config


def main() -> int:
    """Crea modelo, carga un batch y ejecuta un forward pass sin entrenar."""
    parser = argparse.ArgumentParser(
        description="Inspecciona la arquitectura Mini-GPT con un forward pass."
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=config.MAX_CONTEXT_LENGTH)
    args = parser.parse_args()

    try:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "PyTorch no esta instalado. Ejecuta: pip install -r requirements.txt"
            ) from exc

        if not config.TOKENIZER_PATH.exists():
            raise FileNotFoundError(
                "No existe data/processed/tokenizer.json. Ejecuta: "
                "python -m src.build_vocab"
            )

        for split_name in ("train", "val", "test"):
            split_path = config.PROCESSED_DATA_DIR / f"{split_name}.csv"
            if not split_path.exists():
                raise FileNotFoundError(
                    f"No existe {split_path}. Ejecuta: python -m src.prepare_data"
                )

        from src.dataset import create_dataloaders
        from src.model import create_model_from_config
        from src.tokenizer import SimpleTokenizer
        from src.utils import get_device

        tokenizer = SimpleTokenizer.load(config.TOKENIZER_PATH)
        dataloaders = create_dataloaders(
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            max_length=args.max_length,
            processed_dir=config.PROCESSED_DATA_DIR,
        )
        model = create_model_from_config(vocab_size=tokenizer.vocab_size)
        device = get_device()
        model.to(device)
        model.eval()

        batch = next(iter(dataloaders["train"]))
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            logits = outputs["logits"]
            loss = outputs["loss"]
            probabilities = torch.softmax(logits, dim=-1)
            predictions = torch.argmax(probabilities, dim=-1)

        parameter_counts = model.count_parameters()
        model_config = model.get_model_config()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print("Inspeccion del modelo completada.")
    print(f"- Dispositivo: {device}")
    print(f"- vocab_size: {model_config['vocab_size']}")
    print(f"- num_classes: {model_config['num_classes']}")
    print(f"- Configuracion del modelo: {model_config}")
    print(f"- Parametros totales: {parameter_counts['total']}")
    print(f"- Parametros entrenables: {parameter_counts['trainable']}")
    print(f"- input_ids shape: {tuple(input_ids.shape)}")
    print(f"- logits shape: {tuple(logits.shape)}")
    print(f"- loss inicial: {float(loss.detach().cpu()) if loss is not None else None}")
    print(f"- probabilidades ejemplo: {probabilities[0].detach().cpu().tolist()}")
    print(f"- labels reales: {labels.detach().cpu().tolist()}")
    print(f"- predicciones iniciales: {predictions.detach().cpu().tolist()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
