"""Prediccion de sentimiento para textos nuevos."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any

from src import config
from src.tokenizer import SimpleTokenizer
from src.utils import load_json


def parse_args() -> argparse.Namespace:
    """Parsea argumentos de prediccion."""
    parser = argparse.ArgumentParser(
        description="Predice sentimiento con el Mini-GPT entrenado."
    )
    parser.add_argument("--text", default=None, help="Texto a clasificar.")
    parser.add_argument(
        "--checkpoint",
        default=str(config.MODELS_DIR / "mini_gpt_sentiment_best_weighted.pt"),
        help="Ruta del checkpoint entrenado.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=config.MAX_CONTEXT_LENGTH,
        help="Longitud maxima de contexto para tokenizacion.",
    )
    parser.add_argument("--show-probs", action="store_true")
    parser.add_argument("--show-logits", action="store_true")
    parser.add_argument("--interactive", action="store_true")
    return parser.parse_args()


def check_prediction_requirements(checkpoint_path: str | Path) -> None:
    """Valida dependencias y archivos necesarios para predecir."""
    if importlib.util.find_spec("torch") is None:
        raise RuntimeError("PyTorch no esta instalado. Ejecuta: pip install -r requirements.txt")

    checkpoint = resolve_project_path(checkpoint_path)
    required_files = {
        checkpoint: (
            f"No existe el checkpoint {checkpoint}. Ejecuta el entrenamiento o "
            "verifica la ruta con --checkpoint."
        ),
        config.TOKENIZER_PATH: "Falta tokenizer.json. Ejecuta: python -m src.build_vocab",
        config.PROCESSED_DATA_DIR
        / "label_mapping.json": "Falta label_mapping.json. Ejecuta: python -m src.prepare_data",
    }

    for path, message in required_files.items():
        if not Path(path).exists():
            raise FileNotFoundError(message)


def load_checkpoint_safely(checkpoint_path: str | Path, device: Any) -> dict[str, Any]:
    """Carga checkpoint compatible con versiones recientes de PyTorch."""
    torch = import_torch()
    path = Path(checkpoint_path)
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def get_id_to_label(label_mapping: dict[str, int]) -> dict[int, str]:
    """Invierte el diccionario label -> id."""
    return {int(label_id): str(label) for label, label_id in label_mapping.items()}


def load_prediction_assets(
    checkpoint_path: str | Path,
    device: Any,
) -> tuple[Any, SimpleTokenizer, dict[str, int], dict[int, str]]:
    """Carga modelo, tokenizer y mapeos de etiquetas para inferencia."""
    from src.model import MiniGPTForSentiment

    tokenizer = SimpleTokenizer.load(config.TOKENIZER_PATH)
    label_mapping = load_json(config.PROCESSED_DATA_DIR / "label_mapping.json")
    id_to_label = get_id_to_label(label_mapping)

    checkpoint = load_checkpoint_safely(resolve_project_path(checkpoint_path), device)
    model_config = checkpoint.get("model_config")
    if not model_config:
        raise ValueError("El checkpoint no contiene model_config.")

    model = MiniGPTForSentiment(
        vocab_size=int(model_config["vocab_size"]),
        max_context_length=int(model_config["max_context_length"]),
        embed_dim=int(model_config["embed_dim"]),
        num_heads=int(model_config["num_heads"]),
        num_layers=int(model_config["num_layers"]),
        num_classes=int(model_config["num_classes"]),
        dropout=float(model_config["dropout"]),
        pad_token_id=int(model_config.get("pad_token_id", 0)),
        pooling=str(model_config.get("pooling", "last")),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, tokenizer, label_mapping, id_to_label


def predict_sentiment(
    text: str,
    model: Any,
    tokenizer: SimpleTokenizer,
    id_to_label: dict[int, str],
    device: Any,
    max_length: int = config.MAX_CONTEXT_LENGTH,
) -> dict[str, Any]:
    """Predice sentimiento de un texto usando modelo y tokenizer cargados."""
    torch = import_torch()
    encoded = tokenizer.encode_plus(
        text,
        max_length=max_length,
        padding=True,
        truncation=True,
    )
    input_ids = torch.tensor([encoded["input_ids"]], dtype=torch.long, device=device)
    attention_mask = torch.tensor(
        [encoded["attention_mask"]],
        dtype=torch.long,
        device=device,
    )

    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs["logits"]
        probabilities_tensor = torch.softmax(logits, dim=-1)
        predicted_label_id = int(torch.argmax(probabilities_tensor, dim=-1).item())

    probabilities = probabilities_tensor.squeeze(0).detach().cpu().tolist()
    logits_list = logits.squeeze(0).detach().cpu().tolist()
    predicted_label = id_to_label[predicted_label_id]
    confidence = float(probabilities[predicted_label_id])

    return {
        "text": text,
        "predicted_label": predicted_label,
        "predicted_label_id": predicted_label_id,
        "confidence": confidence,
        "probabilities": {
            id_to_label[index]: float(probabilities[index])
            for index in sorted(id_to_label)
        },
        "logits": [float(value) for value in logits_list],
    }


def print_prediction(
    result: dict[str, Any],
    show_probs: bool = False,
    show_logits: bool = False,
) -> None:
    """Imprime una prediccion en formato legible para consola."""
    print("Texto:")
    print(result["text"])
    print("")
    print("Prediccion:")
    print(result["predicted_label"])
    print("")
    print("Confianza:")
    print(f"{result['confidence']:.4f}")

    if show_probs:
        print("")
        print("Probabilidades:")
        for label, probability in result["probabilities"].items():
            print(f"{label}: {probability:.4f}")

    if show_logits:
        print("")
        print("Logits:")
        print([round(float(value), 6) for value in result["logits"]])


def interactive_mode(
    model: Any,
    tokenizer: SimpleTokenizer,
    id_to_label: dict[int, str],
    device: Any,
    max_length: int,
) -> None:
    """Permite clasificar multiples frases desde consola."""
    print("Mini-GPT Sentiment - modo interactivo")
    print("Escribe una frase o 'salir' para terminar.")

    while True:
        text = input("> ").strip()
        if text.lower() in {"salir", "exit", "quit"}:
            break
        if not text:
            continue

        result = predict_sentiment(
            text=text,
            model=model,
            tokenizer=tokenizer,
            id_to_label=id_to_label,
            device=device,
            max_length=max_length,
        )
        print_prediction(result, show_probs=True)
        print("")


def main() -> int:
    """Punto de entrada para prediccion desde consola."""
    args = parse_args()

    if not args.text and not args.interactive:
        print("Indica --text o usa --interactive.")
        print('Ejemplo: python -m src.predict --text "The food was delicious"')
        return 1

    try:
        check_prediction_requirements(args.checkpoint)
        from src.utils import get_device

        device = get_device()
        model, tokenizer, _, id_to_label = load_prediction_assets(args.checkpoint, device)

        if args.interactive:
            interactive_mode(
                model=model,
                tokenizer=tokenizer,
                id_to_label=id_to_label,
                device=device,
                max_length=args.max_length,
            )
            return 0

        result = predict_sentiment(
            text=args.text,
            model=model,
            tokenizer=tokenizer,
            id_to_label=id_to_label,
            device=device,
            max_length=args.max_length,
        )
        print_prediction(
            result,
            show_probs=args.show_probs,
            show_logits=args.show_logits,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


def import_torch() -> Any:
    """Importa PyTorch con mensaje claro si falta."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch no esta instalado. Ejecuta: pip install -r requirements.txt") from exc

    return torch


def resolve_project_path(path: str | Path) -> Path:
    """Resuelve rutas relativas desde la raiz del proyecto."""
    resolved = Path(path)
    if resolved.is_absolute():
        return resolved
    return config.PROJECT_ROOT / resolved


if __name__ == "__main__":
    raise SystemExit(main())
