"""Entrenamiento del Mini-GPT para analisis de sentimiento."""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from src import config
from src.utils import ensure_dir, load_json, save_json, set_seed


def parse_args() -> argparse.Namespace:
    """Parsea argumentos de entrenamiento."""
    parser = argparse.ArgumentParser(
        description="Entrena el Mini-GPT para clasificacion de sentimiento."
    )
    parser.add_argument("--epochs", type=int, default=config.DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=config.WEIGHT_DECAY)
    parser.add_argument("--max-length", type=int, default=config.MAX_CONTEXT_LENGTH)
    parser.add_argument("--grad-clip", type=float, default=config.GRAD_CLIP)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--use-class-weights", action="store_true")
    parser.add_argument("--use-focal-loss", action="store_true")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--focal-alpha", type=float, default=None)
    parser.add_argument("--label-smoothing", type=float, default=config.LABEL_SMOOTHING)
    parser.add_argument("--warmup-ratio", type=float, default=config.WARMUP_RATIO)
    parser.add_argument("--patience", type=int, default=config.PATIENCE)
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument(
        "--checkpoint-name",
        default=config.BEST_MODEL_PATH.name,
        help="Nombre del checkpoint con mejor validacion.",
    )
    parser.add_argument(
        "--last-checkpoint-name",
        default=config.LAST_MODEL_PATH.name,
        help="Nombre del ultimo checkpoint guardado.",
    )
    return parser.parse_args()


def check_training_requirements() -> None:
    """Valida dependencias y artefactos necesarios antes de entrenar."""
    if importlib.util.find_spec("torch") is None:
        raise RuntimeError("PyTorch no esta instalado. Ejecuta: pip install -r requirements.txt")

    required_files = {
        config.TOKENIZER_PATH: "Falta tokenizer.json. Ejecuta: python -m src.build_vocab",
        config.PROCESSED_DATA_DIR
        / "train.csv": "Falta train.csv. Ejecuta: python -m src.prepare_data",
        config.PROCESSED_DATA_DIR
        / "val.csv": "Falta val.csv. Ejecuta: python -m src.prepare_data",
        config.PROCESSED_DATA_DIR
        / "label_mapping.json": (
            "Falta label_mapping.json. Ejecuta: python -m src.prepare_data"
        ),
    }

    for path, message in required_files.items():
        if not Path(path).exists():
            raise FileNotFoundError(message)


def compute_class_weights(train_df: pd.DataFrame, num_classes: int) -> Any:
    """Calcula pesos inversos a la frecuencia de clase.

    Esto ayuda a compensar el desbalance observado en la tarea binaria
    (negative=136676, positive=82239 antes del split).
    """
    torch = import_torch()
    if "label_id" not in train_df.columns:
        raise ValueError("train_df debe contener la columna label_id.")

    counts = train_df["label_id"].value_counts().to_dict()
    total = int(sum(counts.values()))
    weights: list[float] = []
    for class_id in range(num_classes):
        class_count = int(counts.get(class_id, 0))
        if class_count == 0:
            weights.append(0.0)
        else:
            weights.append(total / (num_classes * class_count))

    return torch.tensor(weights, dtype=torch.float32)


class FocalLoss:
    """Focal Loss para manejar desbalance de clases y ejemplos dificiles.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Any = None,
        label_smoothing: float = 0.0,
    ) -> None:
        torch = import_torch()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        if alpha is not None and not isinstance(alpha, torch.Tensor):
            alpha = torch.tensor(alpha, dtype=torch.float32)
        self.alpha = alpha

    def __call__(self, logits: Any, targets: Any) -> Any:
        torch = import_torch()
        import torch.nn.functional as F

        if self.label_smoothing > 0:
            num_classes = logits.size(-1)
            smooth = self.label_smoothing / num_classes
            one_hot = torch.zeros_like(logits).scatter(1, targets.unsqueeze(1), 1.0)
            one_hot = one_hot * (1.0 - self.label_smoothing) + smooth
            log_probs = F.log_softmax(logits, dim=-1)
            ce_loss = -(one_hot * log_probs).sum(dim=-1)
        else:
            ce_loss = F.cross_entropy(logits, targets, reduction="none")

        probs = torch.softmax(logits, dim=-1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_weight = (1.0 - pt) ** self.gamma

        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device).gather(0, targets)
            focal_weight = alpha_t * focal_weight

        return (focal_weight * ce_loss).mean()


def create_scheduler(
    optimizer: Any,
    num_training_steps: int,
    warmup_ratio: float = 0.1,
) -> Any:
    """Crea un cosine scheduler con warmup lineal."""
    torch = import_torch()
    num_warmup_steps = int(num_training_steps * warmup_ratio)

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_one_epoch(
    model: Any,
    dataloader: Any,
    optimizer: Any,
    criterion: Any,
    device: Any,
    grad_clip: float = 1.0,
    max_batches: int | None = None,
    scheduler: Any = None,
    scaler: Any = None,
) -> dict[str, float]:
    """Entrena una epoca y retorna perdida y accuracy promedio."""
    torch = import_torch()
    tqdm = get_tqdm()
    use_amp = scaler is not None

    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    processed_batches = 0
    total_batches = limited_total_batches(dataloader, max_batches)

    progress = tqdm(dataloader, total=total_batches, desc="train", leave=False)
    for batch_index, batch in enumerate(progress):
        if max_batches is not None and batch_index >= max_batches:
            break

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.amp.autocast(device_type=device.type):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs["logits"]
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            if grad_clip is not None and grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs["logits"]
            loss = criterion(logits, labels)
            loss.backward()
            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        batch_size = labels.size(0)
        predictions = torch.argmax(logits, dim=-1)
        total_loss += float(loss.detach().cpu()) * batch_size
        total_correct += int((predictions == labels).sum().detach().cpu())
        total_examples += int(batch_size)
        processed_batches += 1

        if hasattr(progress, "set_postfix"):
            progress.set_postfix(
                loss=total_loss / max(total_examples, 1),
                acc=total_correct / max(total_examples, 1),
            )

    return {
        "loss": total_loss / max(total_examples, 1),
        "accuracy": total_correct / max(total_examples, 1),
        "batches": float(processed_batches),
    }


def validate_one_epoch(
    model: Any,
    dataloader: Any,
    criterion: Any,
    device: Any,
    max_batches: int | None = None,
) -> dict[str, float | None]:
    """Evalua una epoca sobre validacion sin actualizar pesos."""
    torch = import_torch()
    tqdm = get_tqdm()

    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    processed_batches = 0
    all_predictions: list[int] = []
    all_labels: list[int] = []
    total_batches = limited_total_batches(dataloader, max_batches)

    progress = tqdm(dataloader, total=total_batches, desc="val", leave=False)
    with torch.no_grad():
        for batch_index, batch in enumerate(progress):
            if max_batches is not None and batch_index >= max_batches:
                break

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs["logits"]
            loss = criterion(logits, labels)
            predictions = torch.argmax(logits, dim=-1)

            batch_size = labels.size(0)
            total_loss += float(loss.detach().cpu()) * batch_size
            total_correct += int((predictions == labels).sum().detach().cpu())
            total_examples += int(batch_size)
            processed_batches += 1
            all_predictions.extend(predictions.detach().cpu().tolist())
            all_labels.extend(labels.detach().cpu().tolist())

            if hasattr(progress, "set_postfix"):
                progress.set_postfix(
                    loss=total_loss / max(total_examples, 1),
                    acc=total_correct / max(total_examples, 1),
                )

    macro_f1 = compute_macro_f1(all_labels, all_predictions)
    return {
        "loss": total_loss / max(total_examples, 1),
        "accuracy": total_correct / max(total_examples, 1),
        "macro_f1": macro_f1,
        "batches": float(processed_batches),
    }


def save_checkpoint(
    path: str | Path,
    model: Any,
    optimizer: Any,
    epoch: int,
    history: dict[str, Any],
    model_config: dict[str, Any],
    label_mapping: dict[str, int],
    tokenizer_path: str | Path,
    metric_name: str,
    metric_value: float,
) -> Path:
    """Guarda un checkpoint de entrenamiento con torch.save."""
    torch = import_torch()
    checkpoint_path = Path(path)
    ensure_dir(checkpoint_path.parent)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
            "model_config": model_config,
            "label_mapping": label_mapping,
            "tokenizer_path": str(tokenizer_path),
            "metric_name": metric_name,
            "metric_value": metric_value,
        },
        checkpoint_path,
    )
    return checkpoint_path


def plot_training_curves(history: dict[str, Any]) -> dict[str, str]:
    """Genera curvas de perdida, accuracy y macro-F1."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ensure_dir(config.FIGURES_DIR)
    epochs = [item["epoch"] for item in history["epochs"]]
    generated: dict[str, str] = {}

    if epochs:
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, [item["train_loss"] for item in history["epochs"]], label="train")
        plt.plot(epochs, [item["val_loss"] for item in history["epochs"]], label="val")
        plt.xlabel("Epoca")
        plt.ylabel("Loss")
        plt.title("Curva de perdida")
        plt.legend()
        plt.tight_layout()
        plt.savefig(config.TRAINING_LOSS_FIGURE_PATH, dpi=150)
        plt.close()
        generated["loss"] = str(relative_to_project(config.TRAINING_LOSS_FIGURE_PATH))

        plt.figure(figsize=(8, 5))
        plt.plot(
            epochs,
            [item["train_accuracy"] for item in history["epochs"]],
            label="train",
        )
        plt.plot(
            epochs,
            [item["val_accuracy"] for item in history["epochs"]],
            label="val",
        )
        plt.xlabel("Epoca")
        plt.ylabel("Accuracy")
        plt.title("Curva de accuracy")
        plt.legend()
        plt.tight_layout()
        plt.savefig(config.TRAINING_ACC_FIGURE_PATH, dpi=150)
        plt.close()
        generated["accuracy"] = str(relative_to_project(config.TRAINING_ACC_FIGURE_PATH))

        if any(item.get("val_macro_f1") is not None for item in history["epochs"]):
            plt.figure(figsize=(8, 5))
            plt.plot(
                epochs,
                [item.get("val_macro_f1") for item in history["epochs"]],
                label="val_macro_f1",
            )
            plt.xlabel("Epoca")
            plt.ylabel("Macro F1")
            plt.title("Curva de F1 macro en validacion")
            plt.legend()
            plt.tight_layout()
            plt.savefig(config.TRAINING_F1_FIGURE_PATH, dpi=150)
            plt.close()
            generated["macro_f1"] = str(relative_to_project(config.TRAINING_F1_FIGURE_PATH))

    return generated


def save_training_history(
    history: dict[str, Any],
    args: argparse.Namespace,
    model_config: dict[str, Any],
    output_path: str | Path,
) -> Path:
    """Guarda historial de entrenamiento en JSON."""
    payload = {
        "args": vars(args),
        "model_config": model_config,
        "history": history["epochs"],
        "best_epoch": history.get("best_epoch"),
        "best_metric_name": history.get("best_metric_name"),
        "best_metric_value": history.get("best_metric_value"),
        "best_checkpoint": history.get("best_checkpoint"),
        "last_checkpoint": history.get("last_checkpoint"),
        "generated_figures": history.get("generated_figures", {}),
    }
    return save_json(payload, output_path)


def generate_training_summary(
    history: dict[str, Any],
    args: argparse.Namespace,
    model_config: dict[str, Any],
    output_path: str | Path,
) -> Path:
    """Genera resumen Markdown del entrenamiento."""
    lines: list[str] = []
    reduced_run = args.max_train_batches is not None or args.max_val_batches is not None

    lines.append("# Resumen de entrenamiento del Mini-GPT")
    lines.append("")
    lines.append("## Configuracion del entrenamiento")
    lines.append("")
    training_rows = [
        ("epochs", args.epochs),
        ("batch_size", args.batch_size),
        ("learning_rate", args.learning_rate),
        ("weight_decay", args.weight_decay),
        ("max_length", args.max_length),
        ("grad_clip", args.grad_clip),
        ("use_class_weights", args.use_class_weights),
        ("patience", args.patience),
        ("max_train_batches", args.max_train_batches),
        ("max_val_batches", args.max_val_batches),
    ]
    lines.extend(markdown_table(["Parametro", "Valor"], training_rows))
    lines.append("")

    lines.append("## Configuracion del modelo")
    lines.append("")
    lines.extend(markdown_table(["Parametro", "Valor"], model_config.items()))
    lines.append("")

    lines.append("## Dataset utilizado")
    lines.append("")
    lines.append("- Entrenamiento: `data/processed/train.csv`.")
    lines.append("- Validacion: `data/processed/val.csv`.")
    lines.append("- `test.csv` se reserva para la evaluacion final posterior.")
    lines.append("- El vocabulario fue construido solo con `train.csv`.")
    lines.append("")

    lines.append("## Metrica principal de seleccion")
    lines.append("")
    lines.append(f"- Metrica: `{history.get('best_metric_name')}`.")
    lines.append(f"- Mejor valor: {history.get('best_metric_value')}.")
    lines.append(f"- Mejor epoca: {history.get('best_epoch')}.")
    lines.append("")

    lines.append("## Resultados por epoca")
    lines.append("")
    rows = [
        (
            item["epoch"],
            round(item["train_loss"], 6),
            round(item["train_accuracy"], 6),
            round(item["val_loss"], 6),
            round(item["val_accuracy"], 6),
            None if item.get("val_macro_f1") is None else round(item["val_macro_f1"], 6),
        )
        for item in history["epochs"]
    ]
    lines.extend(
        markdown_table(
            ["Epoca", "Train loss", "Train acc", "Val loss", "Val acc", "Val macro F1"],
            rows,
        )
    )
    lines.append("")

    lines.append("## Mejor checkpoint")
    lines.append("")
    lines.append(f"- Mejor checkpoint: `{history.get('best_checkpoint')}`.")
    lines.append(f"- Ultimo checkpoint: `{history.get('last_checkpoint')}`.")
    lines.append("")

    lines.append("## Observaciones para el informe academico")
    lines.append("")
    lines.append("- El entrenamiento usa `train.csv` y selecciona checkpoint con `val.csv`.")
    lines.append("- La evaluacion final sobre `test.csv` se deja para la siguiente fase.")
    if reduced_run:
        lines.append(
            "- Esta corrida uso `max-train-batches` o `max-val-batches`; por tanto "
            "es una corrida reducida de prueba y no el entrenamiento final."
        )
    lines.append("")

    lines.append("## Limitaciones del entrenamiento")
    lines.append("")
    lines.append("- El modelo es deliberadamente pequeno para fines didacticos.")
    lines.append("- Las metricas de validacion no sustituyen la evaluacion final en test.")
    lines.append("- Si se ejecuta en CPU, el entrenamiento completo puede tardar mas.")
    lines.append("")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def main() -> int:
    """Ejecuta entrenamiento completo o reducido segun argumentos."""
    args = parse_args()

    try:
        check_training_requirements()

        torch = import_torch()
        from src.dataset import create_dataloaders, load_processed_split
        from src.model import create_model_from_config
        from src.tokenizer import SimpleTokenizer
        from src.utils import get_device

        set_seed(config.RANDOM_SEED)
        tokenizer = SimpleTokenizer.load(config.TOKENIZER_PATH)
        label_mapping = load_json(config.PROCESSED_DATA_DIR / "label_mapping.json")
        num_classes = len(label_mapping)

        dataloaders = create_dataloaders(
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            max_length=args.max_length,
            processed_dir=config.PROCESSED_DATA_DIR,
            num_workers=args.num_workers,
            include_test=False,
        )

        model = create_model_from_config(
            vocab_size=tokenizer.vocab_size,
            num_classes=num_classes,
        )
        device = get_device()
        model.to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )

        train_df_for_weights = load_processed_split("train", processed_dir=config.PROCESSED_DATA_DIR)
        class_weights = None
        if args.use_class_weights or args.use_focal_loss:
            class_weights = compute_class_weights(train_df_for_weights, num_classes=num_classes).to(device)

        if args.use_focal_loss:
            criterion = FocalLoss(
                gamma=args.focal_gamma,
                alpha=class_weights if args.focal_alpha is None else args.focal_alpha,
                label_smoothing=args.label_smoothing,
            )
        elif args.use_class_weights:
            criterion = torch.nn.CrossEntropyLoss(
                weight=class_weights,
                label_smoothing=args.label_smoothing,
            )
        else:
            criterion = torch.nn.CrossEntropyLoss(
                label_smoothing=args.label_smoothing,
            )

        num_train_batches = limited_total_batches(dataloaders["train"], args.max_train_batches)
        num_training_steps = num_train_batches * args.epochs
        scheduler = create_scheduler(
            optimizer=optimizer,
            num_training_steps=num_training_steps,
            warmup_ratio=args.warmup_ratio,
        )

        scaler = None
        if args.use_amp and device.type in ("cuda", "cpu"):
            scaler = torch.amp.GradScaler(device_type=device.type)

        model_config = model.get_model_config()
        best_checkpoint_path = resolve_checkpoint_path(args.checkpoint_name)
        last_checkpoint_path = resolve_checkpoint_path(args.last_checkpoint_name)
        history: dict[str, Any] = {
            "epochs": [],
            "best_epoch": None,
            "best_metric_name": None,
            "best_metric_value": None,
            "best_checkpoint": str(relative_to_project(best_checkpoint_path)),
            "last_checkpoint": str(relative_to_project(last_checkpoint_path)),
            "generated_figures": {},
            "device": str(device),
        }

        best_metric_value = float("-inf")
        epochs_without_improvement = 0

        for epoch in range(1, args.epochs + 1):
            print(f"\nEpoca {epoch}/{args.epochs}")
            train_metrics = train_one_epoch(
                model=model,
                dataloader=dataloaders["train"],
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                grad_clip=args.grad_clip,
                max_batches=args.max_train_batches,
                scheduler=scheduler,
                scaler=scaler,
            )
            val_metrics = validate_one_epoch(
                model=model,
                dataloader=dataloaders["val"],
                criterion=criterion,
                device=device,
                max_batches=args.max_val_batches,
            )

            metric_name = (
                "val_macro_f1" if val_metrics.get("macro_f1") is not None else "val_accuracy"
            )
            metric_value = (
                float(val_metrics["macro_f1"])
                if val_metrics.get("macro_f1") is not None
                else float(val_metrics["accuracy"])
            )

            epoch_record = {
                "epoch": epoch,
                "train_loss": float(train_metrics["loss"]),
                "train_accuracy": float(train_metrics["accuracy"]),
                "train_batches": int(train_metrics["batches"]),
                "val_loss": float(val_metrics["loss"]),
                "val_accuracy": float(val_metrics["accuracy"]),
                "val_macro_f1": (
                    None if val_metrics["macro_f1"] is None else float(val_metrics["macro_f1"])
                ),
                "val_batches": int(val_metrics["batches"]),
                "selection_metric_name": metric_name,
                "selection_metric_value": metric_value,
            }
            history["epochs"].append(epoch_record)

            save_checkpoint(
                path=last_checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                history=history,
                model_config=model_config,
                label_mapping=label_mapping,
                tokenizer_path=config.TOKENIZER_PATH,
                metric_name=metric_name,
                metric_value=metric_value,
            )

            improved = metric_value > best_metric_value
            if improved:
                best_metric_value = metric_value
                history["best_epoch"] = epoch
                history["best_metric_name"] = metric_name
                history["best_metric_value"] = metric_value
                epochs_without_improvement = 0
                save_checkpoint(
                    path=best_checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    history=history,
                    model_config=model_config,
                    label_mapping=label_mapping,
                    tokenizer_path=config.TOKENIZER_PATH,
                    metric_name=metric_name,
                    metric_value=metric_value,
                )
            else:
                epochs_without_improvement += 1

            print(
                "Resumen epoca "
                f"{epoch}: train_loss={train_metrics['loss']:.4f}, "
                f"train_acc={train_metrics['accuracy']:.4f}, "
                f"val_loss={val_metrics['loss']:.4f}, "
                f"val_acc={val_metrics['accuracy']:.4f}, "
                f"val_macro_f1={val_metrics['macro_f1']}"
            )

            if epochs_without_improvement >= args.patience:
                print(
                    f"Early stopping: {args.patience} epocas sin mejora en {metric_name}."
                )
                break

        history["generated_figures"] = plot_training_curves(history)
        save_training_history(
            history=history,
            args=args,
            model_config=model_config,
            output_path=config.TRAINING_HISTORY_PATH,
        )
        generate_training_summary(
            history=history,
            args=args,
            model_config=model_config,
            output_path=config.TRAINING_SUMMARY_PATH,
        )

    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print("")
    print("Entrenamiento completado.")
    print(f"- Device usado: {history['device']}")
    print(f"- Mejor epoca: {history['best_epoch']}")
    print(f"- Mejor metrica: {history['best_metric_name']}={history['best_metric_value']}")
    print(f"- Mejor checkpoint: {history['best_checkpoint']}")
    print(f"- Ultimo checkpoint: {history['last_checkpoint']}")
    print(f"- Historial: {relative_to_project(config.TRAINING_HISTORY_PATH)}")
    print(f"- Resumen: {relative_to_project(config.TRAINING_SUMMARY_PATH)}")
    for figure_path in history["generated_figures"].values():
        print(f"- Figura: {figure_path}")
    return 0


def compute_macro_f1(labels: list[int], predictions: list[int]) -> float | None:
    """Calcula macro-F1 si scikit-learn esta disponible."""
    if not labels:
        return None
    try:
        from sklearn.metrics import f1_score
    except ImportError:
        return None

    return float(f1_score(labels, predictions, average="macro", zero_division=0))


def import_torch() -> Any:
    """Importa PyTorch con mensaje claro si falta."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch no esta instalado. Ejecuta: pip install -r requirements.txt") from exc

    return torch


def get_tqdm() -> Any:
    """Retorna tqdm si esta disponible; si no, usa el iterable sin barra."""
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return lambda iterable, **_: iterable

    return tqdm


def limited_total_batches(dataloader: Any, max_batches: int | None) -> int | None:
    """Calcula total de batches para tqdm respetando limites opcionales."""
    try:
        total = len(dataloader)
    except TypeError:
        return max_batches

    if max_batches is None:
        return total
    return min(total, max_batches)


def resolve_checkpoint_path(name_or_path: str | Path) -> Path:
    """Resuelve un nombre de checkpoint dentro de models/ si no es ruta absoluta."""
    path = Path(name_or_path)
    if path.is_absolute():
        return path
    if path.parent != Path("."):
        return config.PROJECT_ROOT / path
    return config.MODELS_DIR / path


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


if __name__ == "__main__":
    raise SystemExit(main())
