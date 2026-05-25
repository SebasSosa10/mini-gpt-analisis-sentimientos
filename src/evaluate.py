"""Evaluacion final del Mini-GPT sobre un split procesado."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src import config
from src.utils import ensure_dir, load_json, save_json


def parse_args() -> argparse.Namespace:
    """Parsea argumentos de evaluacion."""
    parser = argparse.ArgumentParser(
        description="Evalua un checkpoint Mini-GPT sobre train, val o test."
    )
    parser.add_argument(
        "--checkpoint",
        default=str(config.MODELS_DIR / "mini_gpt_sentiment_best_weighted.pt"),
        help="Ruta del checkpoint entrenado.",
    )
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--max-length", type=int, default=config.MAX_CONTEXT_LENGTH)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--output-dir", default=str(config.REPORTS_DIR))
    parser.add_argument(
        "--save-predictions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Guardar CSV con predicciones.",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def check_evaluation_requirements(args: argparse.Namespace) -> None:
    """Valida dependencias y archivos requeridos para evaluar."""
    if importlib.util.find_spec("torch") is None:
        raise RuntimeError("PyTorch no esta instalado. Ejecuta: pip install -r requirements.txt")

    checkpoint_path = resolve_project_path(args.checkpoint)
    required_files = {
        checkpoint_path: (
            f"No existe el checkpoint {checkpoint_path}. Verifica la ruta o ejecuta "
            "el entrenamiento correspondiente."
        ),
        config.TOKENIZER_PATH: "Falta tokenizer.json. Ejecuta: python -m src.build_vocab",
        config.PROCESSED_DATA_DIR
        / "label_mapping.json": "Falta label_mapping.json. Ejecuta: python -m src.prepare_data",
        config.PROCESSED_DATA_DIR
        / f"{args.split}.csv": (
            f"Falta data/processed/{args.split}.csv. Ejecuta: python -m src.prepare_data"
        ),
    }

    for path, message in required_files.items():
        if not Path(path).exists():
            raise FileNotFoundError(message)


def load_checkpoint_safely(checkpoint_path: str | Path, device: Any) -> dict[str, Any]:
    """Carga un checkpoint de forma compatible con versiones recientes de PyTorch."""
    torch = import_torch()
    path = Path(checkpoint_path)
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_model_from_checkpoint(
    checkpoint_path: str | Path,
    device: Any,
) -> tuple[Any, dict[str, Any]]:
    """Reconstruye el modelo desde model_config y carga pesos entrenados."""
    from src.model import MiniGPTForSentiment

    checkpoint = load_checkpoint_safely(checkpoint_path, device)
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
    return model, checkpoint


def get_id_to_label(label_mapping: dict[str, int]) -> dict[int, str]:
    """Invierte el diccionario label -> id."""
    return {int(label_id): str(label) for label, label_id in label_mapping.items()}


def evaluate_model(
    model: Any,
    dataloader: Any,
    device: Any,
    id_to_label: dict[int, str],
    max_batches: int | None = None,
) -> dict[str, Any]:
    """Evalua el modelo y acumula predicciones, probabilidades y loss."""
    torch = import_torch()
    total_loss = 0.0
    total_examples = 0
    y_true: list[int] = []
    y_pred: list[int] = []
    probabilities: list[list[float]] = []
    confidence: list[float] = []

    with torch.no_grad():
        for batch_index, batch in enumerate(dataloader):
            if max_batches is not None and batch_index >= max_batches:
                break

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            logits = outputs["logits"]
            loss = outputs["loss"]
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)

            batch_size = labels.size(0)
            if loss is not None:
                total_loss += float(loss.detach().cpu()) * batch_size
            total_examples += int(batch_size)
            y_true.extend(labels.detach().cpu().tolist())
            y_pred.extend(preds.detach().cpu().tolist())
            probabilities.extend(probs.detach().cpu().tolist())
            confidence.extend(torch.max(probs, dim=-1).values.detach().cpu().tolist())

    return {
        "loss": total_loss / max(total_examples, 1),
        "y_true": y_true,
        "y_pred": y_pred,
        "probabilities": probabilities,
        "confidence": confidence,
        "num_examples": total_examples,
        "target_names": [id_to_label[index] for index in sorted(id_to_label)],
    }


def compute_metrics(
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray,
    target_names: list[str],
) -> dict[str, Any]:
    """Calcula metricas principales de clasificacion."""
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )

    labels = list(range(len(target_names)))
    conf_matrix = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "precision_weighted": float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall_weighted": float(
            recall_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "classification_report": make_json_safe(report),
        "confusion_matrix": conf_matrix.astype(int).tolist(),
        "target_names": target_names,
    }


def save_predictions_csv(
    split_df: pd.DataFrame,
    y_true: list[int],
    y_pred: list[int],
    probabilities: list[list[float]],
    id_to_label: dict[int, str],
    output_path: str | Path,
) -> pd.DataFrame:
    """Guarda un CSV con predicciones y probabilidades por clase."""
    num_rows = len(y_true)
    keep_columns = [
        column
        for column in ("text", "text_clean", "labels", "ratings_overall")
        if column in split_df.columns
    ]
    predictions_df = split_df.head(num_rows)[keep_columns].copy()
    predictions_df["true_label_id"] = y_true
    predictions_df["true_label"] = [id_to_label[int(label_id)] for label_id in y_true]
    predictions_df["predicted_label_id"] = y_pred
    predictions_df["predicted_label"] = [
        id_to_label[int(label_id)] for label_id in y_pred
    ]
    predictions_array = np.asarray(probabilities, dtype=float)
    predictions_df["confidence"] = predictions_array.max(axis=1)

    for class_id in sorted(id_to_label):
        label = id_to_label[class_id]
        predictions_df[f"prob_{label}"] = predictions_array[:, class_id]

    predictions_df["is_correct"] = predictions_df["true_label_id"] == predictions_df[
        "predicted_label_id"
    ]

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    predictions_df.to_csv(output, index=False)
    return predictions_df


def plot_confusion_matrix(
    confusion_matrix_values: list[list[int]],
    target_names: list[str],
    output_path: str | Path,
) -> Path:
    """Grafica matriz de confusion."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        import seaborn as sns
    except ImportError:
        sns = None

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    matrix = np.asarray(confusion_matrix_values, dtype=int)

    plt.figure(figsize=(6, 5))
    if sns is not None:
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=target_names,
            yticklabels=target_names,
        )
    else:
        plt.imshow(matrix, cmap="Blues")
        plt.colorbar()
        plt.xticks(range(len(target_names)), target_names)
        plt.yticks(range(len(target_names)), target_names)
        for row_index in range(matrix.shape[0]):
            for col_index in range(matrix.shape[1]):
                plt.text(
                    col_index,
                    row_index,
                    str(matrix[row_index, col_index]),
                    ha="center",
                    va="center",
                )

    plt.xlabel("Prediccion")
    plt.ylabel("Etiqueta real")
    plt.title("Matriz de confusion")
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()
    return output


def plot_classification_metrics(metrics: dict[str, Any], output_path: str | Path) -> Path:
    """Grafica metricas finales principales."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    metric_names = ["precision_macro", "recall_macro", "f1_macro", "accuracy"]
    values = [metrics[name] for name in metric_names]

    plt.figure(figsize=(8, 5))
    plt.bar(metric_names, values, color="#4C78A8")
    plt.ylim(0, 1)
    plt.ylabel("Valor")
    plt.title("Metricas finales en test")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()
    return output


def analyze_errors(
    predictions_df: pd.DataFrame,
    output_path: str | Path,
    max_examples: int = 30,
) -> dict[str, Any]:
    """Guarda errores ordenados por confianza y resume patrones relevantes."""
    error_columns = [
        column
        for column in (
            "text_clean",
            "true_label",
            "predicted_label",
            "confidence",
            "prob_negative",
            "prob_positive",
        )
        if column in predictions_df.columns
    ]
    errors_df = predictions_df[~predictions_df["is_correct"]].copy()
    errors_df = errors_df.sort_values("confidence", ascending=False)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    errors_df[error_columns].to_csv(output, index=False)

    false_positives = errors_df[
        (errors_df["true_label"] == "negative")
        & (errors_df["predicted_label"] == "positive")
    ]
    false_negatives = errors_df[
        (errors_df["true_label"] == "positive")
        & (errors_df["predicted_label"] == "negative")
    ]
    short_errors = errors_df[
        errors_df.get("text_clean", pd.Series(dtype=str)).fillna("").str.split().str.len()
        <= 3
    ]
    high_confidence_errors = errors_df[errors_df["confidence"] >= 0.8]

    return {
        "total_errors": int(len(errors_df)),
        "false_positives": int(len(false_positives)),
        "false_negatives": int(len(false_negatives)),
        "short_text_errors": int(len(short_errors)),
        "high_confidence_errors": int(len(high_confidence_errors)),
        "top_errors": records_for_summary(errors_df, max_examples),
        "false_positive_examples": records_for_summary(false_positives, max_examples=5),
        "false_negative_examples": records_for_summary(false_negatives, max_examples=5),
        "short_error_examples": records_for_summary(short_errors, max_examples=5),
        "high_confidence_error_examples": records_for_summary(
            high_confidence_errors, max_examples=5
        ),
    }


def generate_evaluation_summary(
    metrics: dict[str, Any],
    checkpoint_info: dict[str, Any],
    split_name: str,
    output_path: str | Path,
    error_stats: dict[str, Any],
) -> Path:
    """Genera resumen Markdown de evaluacion final."""
    lines: list[str] = []
    lines.append("# Evaluacion final del Mini-GPT")
    lines.append("")

    lines.append("## Checkpoint evaluado")
    lines.append("")
    lines.append(f"- Checkpoint: `{checkpoint_info['checkpoint_path']}`")
    lines.append(f"- Epoca del checkpoint: {checkpoint_info.get('epoch')}")
    lines.append(f"- Metrica de seleccion: {checkpoint_info.get('metric_name')}")
    lines.append(f"- Valor de seleccion: {checkpoint_info.get('metric_value')}")
    lines.append("")

    lines.append("## Dataset usado para evaluacion")
    lines.append("")
    lines.append(f"- Split evaluado: `{split_name}`")
    lines.append("- `test.csv` se usa solo para evaluacion final.")
    lines.append("- El modelo fue seleccionado con validacion, no con test.")
    lines.append("")

    lines.append("## Metricas generales")
    lines.append("")
    rows = [
        ("accuracy", metrics["accuracy"]),
        ("precision_macro", metrics["precision_macro"]),
        ("recall_macro", metrics["recall_macro"]),
        ("f1_macro", metrics["f1_macro"]),
        ("f1_weighted", metrics["f1_weighted"]),
    ]
    lines.extend(markdown_table(["Metrica", "Valor"], rows))
    lines.append("")

    lines.append("## Metricas por clase")
    lines.append("")
    report = metrics["classification_report"]
    class_rows = []
    for label in metrics["target_names"]:
        class_metrics = report.get(label, {})
        class_rows.append(
            (
                label,
                class_metrics.get("precision"),
                class_metrics.get("recall"),
                class_metrics.get("f1-score"),
                class_metrics.get("support"),
            )
        )
    lines.extend(
        markdown_table(["Clase", "Precision", "Recall", "F1", "Support"], class_rows)
    )
    lines.append("")

    lines.append("## Matriz de confusion")
    lines.append("")
    lines.append("Filas = etiqueta real; columnas = prediccion.")
    matrix_rows = [
        (metrics["target_names"][row_index], *row)
        for row_index, row in enumerate(metrics["confusion_matrix"])
    ]
    lines.extend(
        markdown_table(["Real/Pred"] + metrics["target_names"], matrix_rows)
    )
    lines.append("")

    lines.append("## Analisis de errores")
    lines.append("")
    lines.append(f"- Total de errores: {error_stats['total_errors']}")
    lines.append(f"- Falsos positivos: {error_stats['false_positives']}")
    lines.append(f"- Falsos negativos: {error_stats['false_negatives']}")
    lines.append(f"- Errores en textos muy cortos: {error_stats['short_text_errors']}")
    lines.append(
        f"- Errores con alta confianza: {error_stats['high_confidence_errors']}"
    )
    lines.append("")

    lines.append("## Interpretacion de resultados")
    lines.append("")
    lines.append(
        "- Macro-F1 es importante porque el dataset binario esta desbalanceado."
    )
    lines.append(
        "- La matriz de confusion permite observar si el modelo confunde mas "
        "positivos con negativos o negativos con positivos."
    )
    lines.append("")

    lines.append("## Limitaciones")
    lines.append("")
    lines.append(
        "- Los errores pueden deberse a textos ambiguos, sarcasmo, reseñas muy "
        "cortas, vocabulario poco frecuente o ruido del dataset."
    )
    lines.append(
        "- Esta evaluacion no modifica el checkpoint ni realiza nuevo entrenamiento."
    )
    lines.append("")

    lines.append("## Archivos generados")
    lines.append("")
    for path in checkpoint_info.get("generated_files", {}).values():
        lines.append(f"- `{path}`")
    lines.append("")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def save_metrics_json(metrics: dict[str, Any], output_path: str | Path) -> Path:
    """Guarda metricas en JSON."""
    return save_json(metrics, output_path)


def main() -> int:
    """Punto de entrada de evaluacion."""
    args = parse_args()

    try:
        check_evaluation_requirements(args)

        torch = import_torch()
        from src.dataset import SentimentDataset, create_dataloader, load_processed_split
        from src.tokenizer import SimpleTokenizer
        from src.utils import get_device

        output_paths = build_output_paths(args.output_dir, args.split)
        ensure_dir(output_paths["figures_dir"])

        device = get_device()
        tokenizer = SimpleTokenizer.load(config.TOKENIZER_PATH)
        label_mapping = load_json(config.PROCESSED_DATA_DIR / "label_mapping.json")
        id_to_label = get_id_to_label(label_mapping)
        target_names = [id_to_label[index] for index in sorted(id_to_label)]

        split_df = load_processed_split(args.split, processed_dir=config.PROCESSED_DATA_DIR)
        dataset = SentimentDataset(
            split_df,
            tokenizer=tokenizer,
            max_length=args.max_length,
        )
        dataloader = create_dataloader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )

        checkpoint_path = resolve_project_path(args.checkpoint)
        model, checkpoint = load_model_from_checkpoint(checkpoint_path, device)
        evaluation = evaluate_model(
            model=model,
            dataloader=dataloader,
            device=device,
            id_to_label=id_to_label,
            max_batches=args.max_batches,
        )
        metrics = compute_metrics(
            evaluation["y_true"],
            evaluation["y_pred"],
            target_names=target_names,
        )
        metrics["loss"] = float(evaluation["loss"])
        metrics["num_examples"] = int(evaluation["num_examples"])
        metrics["split"] = args.split
        metrics["checkpoint"] = str(relative_to_project(checkpoint_path))
        metrics["device"] = str(device)

        generated_files: dict[str, str] = {}
        save_metrics_json(metrics, output_paths["metrics_json"])
        generated_files["metrics_json"] = str(relative_to_project(output_paths["metrics_json"]))

        predictions_df = pd.DataFrame()
        if args.save_predictions:
            predictions_df = save_predictions_csv(
                split_df=split_df,
                y_true=evaluation["y_true"],
                y_pred=evaluation["y_pred"],
                probabilities=evaluation["probabilities"],
                id_to_label=id_to_label,
                output_path=output_paths["predictions_csv"],
            )
            generated_files["predictions_csv"] = str(
                relative_to_project(output_paths["predictions_csv"])
            )

        plot_confusion_matrix(
            metrics["confusion_matrix"],
            target_names,
            output_paths["confusion_matrix_png"],
        )
        generated_files["confusion_matrix_png"] = str(
            relative_to_project(output_paths["confusion_matrix_png"])
        )

        plot_classification_metrics(metrics, output_paths["metrics_barplot_png"])
        generated_files["metrics_barplot_png"] = str(
            relative_to_project(output_paths["metrics_barplot_png"])
        )

        if args.save_predictions:
            error_stats = analyze_errors(predictions_df, output_paths["error_analysis_csv"])
            generated_files["error_analysis_csv"] = str(
                relative_to_project(output_paths["error_analysis_csv"])
            )
        else:
            error_stats = {
                "total_errors": None,
                "false_positives": None,
                "false_negatives": None,
                "short_text_errors": None,
                "high_confidence_errors": None,
            }

        generated_files["summary_md"] = str(relative_to_project(output_paths["summary_md"]))
        checkpoint_info = {
            "checkpoint_path": str(relative_to_project(checkpoint_path)),
            "epoch": checkpoint.get("epoch"),
            "metric_name": checkpoint.get("metric_name"),
            "metric_value": checkpoint.get("metric_value"),
            "generated_files": generated_files,
        }
        generate_evaluation_summary(
            metrics=metrics,
            checkpoint_info=checkpoint_info,
            split_name=args.split,
            output_path=output_paths["summary_md"],
            error_stats=error_stats,
        )

    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print("")
    print("Evaluacion completada.")
    print(f"- Device: {device}")
    print(f"- Checkpoint: {relative_to_project(checkpoint_path)}")
    print(f"- Split: {args.split}")
    print(f"- Accuracy: {metrics['accuracy']:.6f}")
    print(f"- Precision macro: {metrics['precision_macro']:.6f}")
    print(f"- Recall macro: {metrics['recall_macro']:.6f}")
    print(f"- F1 macro: {metrics['f1_macro']:.6f}")
    print(f"- F1 weighted: {metrics['f1_weighted']:.6f}")
    print(f"- Errores: {error_stats.get('total_errors')}")
    print("- Archivos generados:")
    for path in generated_files.values():
        print(f"  - {path}")
    return 0


def build_output_paths(output_dir: str | Path, split_name: str) -> dict[str, Path]:
    """Construye rutas de salida respetando output_dir."""
    base_dir = Path(output_dir)
    if not base_dir.is_absolute():
        base_dir = config.PROJECT_ROOT / base_dir
    figures_dir = base_dir / "figures"

    predictions_name = "test_predictions.csv" if split_name == "test" else f"{split_name}_predictions.csv"
    confusion_name = (
        "confusion_matrix_test.png"
        if split_name == "test"
        else f"confusion_matrix_{split_name}.png"
    )

    return {
        "base_dir": base_dir,
        "figures_dir": figures_dir,
        "metrics_json": base_dir / "evaluation_metrics.json",
        "summary_md": base_dir / "evaluation_summary.md",
        "predictions_csv": base_dir / predictions_name,
        "error_analysis_csv": base_dir / "error_analysis.csv",
        "confusion_matrix_png": figures_dir / confusion_name,
        "metrics_barplot_png": figures_dir / "final_metrics_barplot.png",
    }


def records_for_summary(dataframe: pd.DataFrame, max_examples: int) -> list[dict[str, Any]]:
    """Convierte ejemplos de errores a registros compactos para JSON/Markdown."""
    columns = [
        column
        for column in (
            "text_clean",
            "true_label",
            "predicted_label",
            "confidence",
            "prob_negative",
            "prob_positive",
        )
        if column in dataframe.columns
    ]
    return make_json_safe(dataframe[columns].head(max_examples).to_dict(orient="records"))


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


def markdown_table(headers: list[str], rows: Any) -> list[str]:
    """Construye tabla Markdown sencilla."""
    row_list = list(rows)
    lines = [
        "| " + " | ".join(str(header) for header in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in row_list:
        if not isinstance(row, tuple):
            row = tuple(row)
        lines.append("| " + " | ".join(escape_markdown_cell(value) for value in row) + " |")
    return lines


def escape_markdown_cell(value: Any) -> str:
    """Escapa valores para celdas Markdown."""
    return str(value).replace("\n", " ").replace("\r", " ").replace("|", "\\|")


def relative_to_project(path: str | Path) -> Path:
    """Devuelve ruta relativa al proyecto cuando es posible."""
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(config.PROJECT_ROOT)
    except ValueError:
        return resolved


def make_json_safe(value: Any) -> Any:
    """Convierte objetos numpy/pandas a tipos JSON serializables."""
    if isinstance(value, dict):
        return {str(key): make_json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [make_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [make_json_safe(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if pd.isna(value):
        return None
    return value


if __name__ == "__main__":
    raise SystemExit(main())
