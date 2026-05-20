"""Preparacion del dataset para entrenamiento.

Esta fase toma el CSV crudo, confirma columnas indicadas por argumentos,
normaliza textos y etiquetas, aplica filtros reproducibles y guarda particiones
train/validation/test listas para la fase de tokenizacion.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from src import config
from src.utils import ensure_dir, save_json, set_seed


BINARY_LABEL_MAPPING = {"negative": 0, "positive": 1}
MULTICLASS_LABEL_MAPPING = {"negative": 0, "neutral": 1, "positive": 2}


def find_csv_files(raw_data_dir: str | Path) -> list[Path]:
    """Busca archivos CSV dentro de data/raw/."""
    raw_dir = Path(raw_data_dir)
    csv_files = sorted(path for path in raw_dir.glob("*.csv") if path.is_file())

    if not csv_files:
        raise FileNotFoundError(
            "No se encontraron archivos CSV en data/raw/. Descarga manualmente "
            "el dataset desde Kaggle y ubica el archivo .csv en data/raw/."
        )

    return csv_files


def resolve_input_file(input_path: str | Path | None = None) -> Path:
    """Resuelve que archivo CSV se usara como entrada."""
    if input_path is not None:
        path = Path(input_path)
        if not path.is_absolute():
            path = config.PROJECT_ROOT / path

        if not path.exists():
            raise FileNotFoundError(f"No existe el archivo indicado: {path}")

        if not path.is_file() or path.suffix.lower() != ".csv":
            raise ValueError(f"La ruta indicada no corresponde a un CSV: {path}")

        return path

    csv_files = find_csv_files(config.RAW_DATA_DIR)
    if len(csv_files) > 1:
        print("Se encontraron varios archivos CSV en data/raw/:")
        for index, csv_file in enumerate(csv_files, start=1):
            print(f"  {index}. {relative_to_project(csv_file)}")
        print(f"Se usara el primero: {relative_to_project(csv_files[0])}")

    return csv_files[0]


def load_dataset(csv_path: str | Path) -> pd.DataFrame:
    """Carga el CSV con pandas e intenta alternativas de encoding."""
    path = Path(csv_path)

    try:
        return pd.read_csv(path, low_memory=False)
    except UnicodeDecodeError:
        errors: list[str] = []
        for encoding in ("utf-8", "latin-1"):
            try:
                return pd.read_csv(path, encoding=encoding, low_memory=False)
            except Exception as exc:
                errors.append(f"{encoding}: {exc}")

        raise ValueError(
            "No se pudo leer el CSV por problemas de codificacion. "
            f"Intentos realizados: {' | '.join(errors)}"
        ) from None
    except Exception as exc:
        raise ValueError(f"No se pudo cargar el CSV con pandas: {exc}") from None


def clean_text(text: Any) -> str:
    """Limpia texto de forma moderada para preservar informacion emocional."""
    try:
        if pd.isna(text):
            return ""
    except (TypeError, ValueError):
        pass

    cleaned = str(text).lower()
    cleaned = cleaned.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    # Se conserva puntuacion basica porque puede aportar enfasis emocional.
    cleaned = re.sub(r"[^\w\s.,!?;:'\"()\-/%]", " ", cleaned, flags=re.UNICODE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def normalize_label_from_label(value: Any) -> str | None:
    """Normaliza etiquetas provenientes de una columna de labels.

    Para valores numericos 1, 2 y 3 se usa la codificacion frecuente
    negativo/neutro/positivo. En build_labels esta hipotesis se compara contra
    ratings_overall cuando esa columna existe, porque no debe asumirse sin
    validacion sobre el dataset real.
    """
    if pd.isna(value):
        return None

    text = str(value).strip().lower()
    if not text:
        return None

    direct_mapping = {
        "positive": "positive",
        "pos": "positive",
        "positivo": "positive",
        "negative": "negative",
        "neg": "negative",
        "negativo": "negative",
        "neutral": "neutral",
        "neu": "neutral",
        "neutro": "neutral",
    }
    if text in direct_mapping:
        return direct_mapping[text]

    numeric_mapping = {1: "negative", 2: "neutral", 3: "positive"}
    try:
        numeric_value = float(text)
    except ValueError:
        return None

    if numeric_value.is_integer():
        return numeric_mapping.get(int(numeric_value))

    return None


def normalize_label_from_rating(value: Any) -> str | None:
    """Convierte ratings 1-5 a sentimiento."""
    if pd.isna(value):
        return None

    try:
        rating = float(str(value).strip())
    except ValueError:
        return None

    rounded_rating = int(rating)
    if rating != rounded_rating:
        return None

    if rounded_rating in {1, 2}:
        return "negative"
    if rounded_rating == 3:
        return "neutral"
    if rounded_rating in {4, 5}:
        return "positive"
    return None


def build_labels(
    df: pd.DataFrame,
    label_column: str | None = None,
    rating_column: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Construye la columna sentiment a partir de labels o rating."""
    prepared_df = df.copy()
    total_rows = int(len(prepared_df))
    stats: dict[str, Any] = {
        "label_column": label_column,
        "rating_column": rating_column,
        "source_used": None,
        "valid_labels": 0,
        "invalid_labels": total_rows,
        "label_distribution": {},
        "label_column_valid": 0,
        "rating_column_valid": 0,
        "disagreements_with_rating": None,
    }

    label_sentiment: pd.Series | None = None
    rating_sentiment: pd.Series | None = None

    if label_column and label_column in prepared_df.columns:
        label_sentiment = prepared_df[label_column].apply(normalize_label_from_label)
        prepared_df["sentiment_from_label"] = label_sentiment
        stats["label_column_valid"] = int(label_sentiment.notna().sum())
    elif label_column:
        stats["label_column_missing"] = True

    if rating_column and rating_column in prepared_df.columns:
        rating_sentiment = prepared_df[rating_column].apply(normalize_label_from_rating)
        prepared_df["sentiment_from_rating"] = rating_sentiment
        stats["rating_column_valid"] = int(rating_sentiment.notna().sum())
    elif rating_column:
        stats["rating_column_missing"] = True

    if label_sentiment is not None and rating_sentiment is not None:
        comparable = label_sentiment.notna() & rating_sentiment.notna()
        disagreements = (label_sentiment[comparable] != rating_sentiment[comparable]).sum()
        stats["disagreements_with_rating"] = int(disagreements)
        stats["comparable_labels_with_rating"] = int(comparable.sum())

    # Se prefiere labels si produce etiquetas validas para una parte suficiente
    # del dataset. Si no, se usa rating como fuente alternativa.
    minimum_valid_ratio = 0.5
    label_is_usable = (
        label_sentiment is not None
        and stats["label_column_valid"] > 0
        and stats["label_column_valid"] / max(total_rows, 1) >= minimum_valid_ratio
    )
    rating_is_usable = rating_sentiment is not None and stats["rating_column_valid"] > 0

    if label_is_usable:
        prepared_df["sentiment"] = label_sentiment
        stats["source_used"] = label_column
    elif rating_is_usable:
        prepared_df["sentiment"] = rating_sentiment
        stats["source_used"] = rating_column
    else:
        raise ValueError(
            "No se pudo construir la columna sentiment. Revisa --label-column "
            "y --rating-column, o confirma manualmente el significado de las etiquetas."
        )

    valid_final = int(prepared_df["sentiment"].notna().sum())
    stats["valid_labels"] = valid_final
    stats["invalid_labels"] = int(total_rows - valid_final)
    stats["label_distribution"] = counts_as_dict(prepared_df["sentiment"])
    return prepared_df, stats


def filter_dataset(
    df: pd.DataFrame,
    text_column: str,
    task: str,
    min_text_length: int,
    max_text_length: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Limpia textos, filtra filas invalidas y asigna label_id."""
    if text_column not in df.columns:
        available_columns = ", ".join(str(column) for column in df.columns)
        raise ValueError(
            f"No existe la columna de texto '{text_column}'. "
            f"Columnas disponibles: {available_columns}"
        )

    filtered_df = df.copy()
    stats: dict[str, Any] = {"initial_rows": int(len(filtered_df))}

    if text_column != "text":
        filtered_df["text"] = filtered_df[text_column]

    filtered_df["text_clean"] = filtered_df[text_column].apply(clean_text)
    before = len(filtered_df)
    filtered_df = filtered_df[filtered_df["text_clean"] != ""].copy()
    stats["removed_empty_text"] = int(before - len(filtered_df))

    # min_text_length se mide en palabras para descartar entradas sin contexto.
    filtered_df["word_count"] = filtered_df["text_clean"].str.split().str.len()
    before = len(filtered_df)
    filtered_df = filtered_df[filtered_df["word_count"] >= min_text_length].copy()
    stats["removed_short_text"] = int(before - len(filtered_df))

    # max_text_length se mide en caracteres porque el tokenizer definitivo aun no
    # existe y este limite evita textos atipicamente largos.
    filtered_df["char_count"] = filtered_df["text_clean"].str.len()
    before = len(filtered_df)
    filtered_df = filtered_df[filtered_df["char_count"] <= max_text_length].copy()
    stats["removed_long_text"] = int(before - len(filtered_df))

    before = len(filtered_df)
    filtered_df = filtered_df[filtered_df["sentiment"].notna()].copy()
    stats["removed_missing_sentiment"] = int(before - len(filtered_df))

    if task == "binary":
        allowed_labels = {"negative", "positive"}
        label_mapping = BINARY_LABEL_MAPPING
    elif task == "multiclass":
        allowed_labels = {"negative", "neutral", "positive"}
        label_mapping = MULTICLASS_LABEL_MAPPING
    else:
        raise ValueError("task debe ser 'binary' o 'multiclass'.")

    before = len(filtered_df)
    filtered_df = filtered_df[filtered_df["sentiment"].isin(allowed_labels)].copy()
    stats["removed_by_task"] = int(before - len(filtered_df))
    stats["excluded_neutral_for_binary"] = (
        stats["removed_by_task"] if task == "binary" else 0
    )

    before = len(filtered_df)
    filtered_df = filtered_df.drop_duplicates(subset=["text_clean", "sentiment"]).copy()
    stats["removed_duplicate_text_sentiment"] = int(before - len(filtered_df))

    filtered_df["label_id"] = filtered_df["sentiment"].map(label_mapping).astype(int)
    stats["final_rows"] = int(len(filtered_df))
    stats["class_distribution"] = counts_as_dict(filtered_df["sentiment"])
    stats["label_mapping"] = label_mapping

    if stats["final_rows"] < 100:
        stats["warning"] = "Quedaron menos de 100 filas despues de los filtros."

    return filtered_df, stats


def apply_sampling_and_balancing(
    df: pd.DataFrame,
    max_samples: int | None = None,
    balance_strategy: str = "none",
    random_seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Aplica muestreo opcional y balanceo por downsampling."""
    sampled_df = df.copy()
    stats: dict[str, Any] = {
        "initial_rows": int(len(sampled_df)),
        "max_samples": max_samples,
        "balance_strategy": balance_strategy,
        "rows_after_max_samples": int(len(sampled_df)),
        "rows_after_balancing": int(len(sampled_df)),
        "class_distribution_before": counts_as_dict(sampled_df["sentiment"]),
    }

    if max_samples is not None and max_samples > 0 and max_samples < len(sampled_df):
        sampled_df, stratified = stratified_sample(
            sampled_df,
            sample_size=max_samples,
            random_seed=random_seed,
        )
        stats["max_samples_applied"] = True
        stats["max_samples_stratified"] = stratified
    else:
        stats["max_samples_applied"] = False
        stats["max_samples_stratified"] = False

    stats["rows_after_max_samples"] = int(len(sampled_df))
    stats["class_distribution_after_max_samples"] = counts_as_dict(
        sampled_df["sentiment"]
    )

    if balance_strategy == "downsample":
        class_counts = sampled_df["sentiment"].value_counts()
        if not class_counts.empty:
            minority_count = int(class_counts.min())
            class_samples = (
                group.sample(n=minority_count, random_state=random_seed)
                for _, group in sampled_df.groupby("sentiment")
            )
            sampled_df = pd.concat(class_samples, ignore_index=True)
            sampled_df = sampled_df.sample(frac=1.0, random_state=random_seed).reset_index(
                drop=True
            )
            stats["downsample_target_per_class"] = minority_count
    elif balance_strategy != "none":
        raise ValueError("balance_strategy debe ser 'none' o 'downsample'.")

    stats["rows_after_balancing"] = int(len(sampled_df))
    stats["class_distribution_after_balancing"] = counts_as_dict(sampled_df["sentiment"])
    return sampled_df.reset_index(drop=True), stats


def split_dataset(
    df: pd.DataFrame,
    test_size: float,
    val_size: float,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Divide el dataset en train, validation y test con estratificacion."""
    if test_size <= 0 or val_size <= 0 or test_size + val_size >= 1:
        raise ValueError("test_size y val_size deben ser positivos y sumar menos de 1.")

    temp_size = test_size + val_size
    train_df, temp_df = safe_train_test_split(
        df,
        test_size=temp_size,
        random_seed=random_seed,
    )

    relative_test_size = test_size / temp_size
    val_df, test_df = safe_train_test_split(
        temp_df,
        test_size=relative_test_size,
        random_seed=random_seed,
    )

    output_columns = select_output_columns(df)
    return (
        train_df[output_columns].reset_index(drop=True),
        val_df[output_columns].reset_index(drop=True),
        test_df[output_columns].reset_index(drop=True),
    )


def save_processed_data(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_mapping: dict[str, int],
    stats: dict[str, Any],
) -> dict[str, str]:
    """Guarda particiones y archivos auxiliares en data/processed/."""
    output_dir = ensure_dir(config.PROCESSED_DATA_DIR)
    paths = {
        "train_csv": output_dir / "train.csv",
        "val_csv": output_dir / "val.csv",
        "test_csv": output_dir / "test.csv",
        "label_mapping_json": output_dir / "label_mapping.json",
        "preprocessing_report_json": output_dir / "preprocessing_report.json",
    }

    train_df.to_csv(paths["train_csv"], index=False)
    val_df.to_csv(paths["val_csv"], index=False)
    test_df.to_csv(paths["test_csv"], index=False)
    save_json(label_mapping, paths["label_mapping_json"])
    save_json(stats, paths["preprocessing_report_json"])

    return {key: str(relative_to_project(path)) for key, path in paths.items()}


def generate_markdown_summary(stats: dict[str, Any], output_path: str | Path) -> None:
    """Genera un resumen en Markdown para el informe academico."""
    lines: list[str] = []
    generated_files = stats.get("generated_files", {})

    lines.append("# Resumen de preparacion del dataset")
    lines.append("")
    lines.append("## Archivo utilizado")
    lines.append("")
    lines.append(f"- Archivo: `{stats['input_file']}`")
    lines.append(f"- Filas originales: {stats['raw_rows']}")
    lines.append("")

    lines.append("## Columnas utilizadas")
    lines.append("")
    lines.append(f"- Texto: `{stats['text_column']}`")
    lines.append(f"- Etiqueta: `{stats['label_column']}`")
    lines.append(f"- Rating: `{stats['rating_column']}`")
    lines.append(f"- Fuente final de sentimiento: `{stats['label_stats']['source_used']}`")
    lines.append("")

    lines.append("## Tarea configurada")
    lines.append("")
    lines.append(f"- Tipo de tarea: `{stats['task']}`")
    lines.append(f"- Semilla aleatoria: {stats['random_seed']}")
    lines.append(f"- Test size final: {stats['test_size']}")
    lines.append(f"- Validation size final: {stats['val_size']}")
    lines.append("")

    lines.append("## Limpieza aplicada")
    lines.append("")
    lines.append("- Conversion a minusculas.")
    lines.append("- Eliminacion de saltos de linea y espacios repetidos.")
    lines.append("- Conservacion de letras, numeros, puntuacion basica y espacios.")
    lines.append("- No se aplico limpieza agresiva ni stemming.")
    lines.append("")

    lines.append("## Mapeo de etiquetas")
    lines.append("")
    lines.extend(markdown_table(["Sentimiento", "label_id"], stats["label_mapping"].items()))
    lines.append("")
    lines.append(
        f"- Etiquetas validas construidas: {stats['label_stats']['valid_labels']}"
    )
    lines.append(
        f"- Etiquetas invalidas o no interpretables: {stats['label_stats']['invalid_labels']}"
    )
    disagreements = stats["label_stats"].get("disagreements_with_rating")
    if disagreements is not None:
        comparable = stats["label_stats"].get("comparable_labels_with_rating", 0)
        lines.append(
            f"- Desacuerdos entre labels y ratings_overall: {disagreements} "
            f"de {comparable} casos comparables."
        )
    lines.append("")

    lines.append("## Filtros aplicados")
    lines.append("")
    filter_stats = stats["filter_stats"]
    filter_rows = [
        ("Textos vacios", filter_stats["removed_empty_text"]),
        ("Textos cortos", filter_stats["removed_short_text"]),
        ("Textos largos", filter_stats["removed_long_text"]),
        ("Sin sentimiento", filter_stats["removed_missing_sentiment"]),
        ("Excluidos por tipo de tarea", filter_stats["removed_by_task"]),
        (
            "Duplicados por texto limpio y sentimiento",
            filter_stats["removed_duplicate_text_sentiment"],
        ),
    ]
    lines.extend(markdown_table(["Filtro", "Filas removidas"], filter_rows))
    lines.append("")

    lines.append("## Distribucion final de clases")
    lines.append("")
    lines.extend(
        markdown_table(
            ["Clase", "Cantidad"],
            stats["final_class_distribution"].items(),
        )
    )
    lines.append("")

    lines.append("## Division del dataset")
    lines.append("")
    split_rows = [
        ("train", stats["split_stats"]["train_rows"]),
        ("val", stats["split_stats"]["val_rows"]),
        ("test", stats["split_stats"]["test_rows"]),
    ]
    lines.extend(markdown_table(["Particion", "Filas"], split_rows))
    lines.append("")

    lines.append("## Archivos generados")
    lines.append("")
    for path in generated_files.values():
        lines.append(f"- `{path}`")
    lines.append("")

    lines.append("## Observaciones para el informe academico")
    lines.append("")
    lines.append(
        "- La configuracion principal del proyecto usa clasificacion binaria, "
        "excluyendo la clase neutral."
    )
    lines.append(
        "- La seleccion de columnas se basa en el EDA previo y queda registrada "
        "para trazabilidad."
    )
    lines.append(
        "- La columna de texto limpio queda preparada para la tokenizacion de la "
        "siguiente fase."
    )
    lines.append(
        "- Si se usa modo multiclass, se debe reportar explicitamente la clase "
        "neutral en metodologia y resultados."
    )
    lines.append("")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    """Punto de entrada de la preparacion de datos."""
    parser = argparse.ArgumentParser(
        description="Prepara el dataset para entrenamiento de sentimiento."
    )
    parser.add_argument("--input", default=None, help="Ruta opcional del CSV crudo.")
    parser.add_argument("--text-column", default="text", help="Columna de texto.")
    parser.add_argument("--label-column", default="labels", help="Columna de etiqueta.")
    parser.add_argument(
        "--rating-column",
        default="ratings_overall",
        help="Columna numerica de rating usada como respaldo.",
    )
    parser.add_argument(
        "--task",
        choices=("binary", "multiclass"),
        default="binary",
        help="Tipo de clasificacion a preparar.",
    )
    parser.add_argument("--test-size", type=float, default=config.TEST_SIZE)
    parser.add_argument("--val-size", type=float, default=config.VAL_SIZE)
    parser.add_argument("--min-text-length", type=int, default=3)
    parser.add_argument("--max-text-length", type=int, default=512)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--balance-strategy",
        choices=("none", "downsample"),
        default="none",
        help="Estrategia opcional de balanceo.",
    )
    parser.add_argument("--random-seed", type=int, default=config.RANDOM_SEED)
    args = parser.parse_args()

    try:
        set_seed(args.random_seed)
        csv_path = resolve_input_file(args.input)
        print(f"Archivo seleccionado: {relative_to_project(csv_path)}")

        df = load_dataset(csv_path)
        raw_rows = int(len(df))
        raw_columns = [str(column) for column in df.columns]
        print(f"Dataset cargado: {len(df)} filas, {len(df.columns)} columnas.")

        df, label_stats = build_labels(
            df,
            label_column=args.label_column,
            rating_column=args.rating_column,
        )
        df, filter_stats = filter_dataset(
            df,
            text_column=args.text_column,
            task=args.task,
            min_text_length=args.min_text_length,
            max_text_length=args.max_text_length,
        )
        df, sampling_stats = apply_sampling_and_balancing(
            df,
            max_samples=args.max_samples,
            balance_strategy=args.balance_strategy,
            random_seed=args.random_seed,
        )

        if len(df) < 100:
            print("Advertencia: despues de filtros y muestreo quedaron menos de 100 filas.")

        train_df, val_df, test_df = split_dataset(
            df,
            test_size=args.test_size,
            val_size=args.val_size,
            random_seed=args.random_seed,
        )

        label_mapping = (
            BINARY_LABEL_MAPPING if args.task == "binary" else MULTICLASS_LABEL_MAPPING
        )
        split_stats = {
            "train_rows": int(len(train_df)),
            "val_rows": int(len(val_df)),
            "test_rows": int(len(test_df)),
            "train_distribution": counts_as_dict(train_df["sentiment"]),
            "val_distribution": counts_as_dict(val_df["sentiment"]),
            "test_distribution": counts_as_dict(test_df["sentiment"]),
        }
        stats: dict[str, Any] = {
            "input_file": str(relative_to_project(csv_path)),
            "raw_rows": raw_rows,
            "raw_columns": raw_columns,
            "text_column": args.text_column,
            "label_column": args.label_column,
            "rating_column": args.rating_column,
            "task": args.task,
            "test_size": args.test_size,
            "val_size": args.val_size,
            "min_text_length_words": args.min_text_length,
            "max_text_length_characters": args.max_text_length,
            "max_samples": args.max_samples,
            "balance_strategy": args.balance_strategy,
            "random_seed": args.random_seed,
            "label_mapping": label_mapping,
            "label_stats": label_stats,
            "filter_stats": filter_stats,
            "sampling_stats": sampling_stats,
            "split_stats": split_stats,
            "final_rows_before_split": int(len(df)),
            "final_class_distribution": counts_as_dict(df["sentiment"]),
        }

        generated_files = save_processed_data(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            label_mapping=label_mapping,
            stats=stats,
        )
        summary_path = config.REPORTS_DIR / "preprocessing_summary.md"
        stats["generated_files"] = {
            **generated_files,
            "preprocessing_summary_md": str(relative_to_project(summary_path)),
        }
        save_json(stats, config.PROCESSED_DATA_DIR / "preprocessing_report.json")
        generate_markdown_summary(stats, summary_path)

    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print("")
    print("Preparacion de datos completada.")
    print(f"- Archivo utilizado: {stats['input_file']}")
    print(f"- Tarea: {args.task}")
    print(f"- Fuente de etiqueta usada: {label_stats['source_used']}")
    if label_stats.get("disagreements_with_rating") is not None:
        print(
            "- Desacuerdos labels vs rating: "
            f"{label_stats['disagreements_with_rating']} de "
            f"{label_stats.get('comparable_labels_with_rating', 0)}"
        )
    print(f"- Train: {split_stats['train_rows']} filas")
    print(f"- Val: {split_stats['val_rows']} filas")
    print(f"- Test: {split_stats['test_rows']} filas")
    print(f"- Distribucion final: {stats['final_class_distribution']}")
    print("- Archivos generados:")
    for path in stats["generated_files"].values():
        print(f"  - {path}")
    return 0


def stratified_sample(
    df: pd.DataFrame,
    sample_size: int,
    random_seed: int,
) -> tuple[pd.DataFrame, bool]:
    """Toma una muestra estratificada cuando es posible."""
    try:
        sample_df, _ = train_test_split(
            df,
            train_size=sample_size,
            stratify=df["label_id"],
            random_state=random_seed,
        )
        return sample_df.reset_index(drop=True), True
    except ValueError:
        return df.sample(n=sample_size, random_state=random_seed).reset_index(drop=True), False


def safe_train_test_split(
    df: pd.DataFrame,
    test_size: float,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Divide con estratificacion y cae a division simple si no es posible."""
    stratify = df["label_id"] if df["label_id"].nunique() > 1 else None
    try:
        return train_test_split(
            df,
            test_size=test_size,
            stratify=stratify,
            random_state=random_seed,
        )
    except ValueError:
        return train_test_split(df, test_size=test_size, random_state=random_seed)


def select_output_columns(df: pd.DataFrame) -> list[str]:
    """Selecciona columnas minimas y columnas originales utiles para trazabilidad."""
    desired_columns = [
        "text",
        "text_clean",
        "sentiment",
        "label_id",
        "ratings_overall",
        "labels",
    ]
    return [column for column in desired_columns if column in df.columns]


def counts_as_dict(series: pd.Series) -> dict[str, int]:
    """Convierte conteos de pandas a diccionario con claves de texto."""
    counts = {
        str(key): int(value)
        for key, value in series.value_counts(dropna=False).items()
    }
    return dict(sorted(counts.items(), key=lambda item: item[0]))


def markdown_table(headers: list[str], rows: Any) -> list[str]:
    """Construye una tabla Markdown sencilla."""
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
    """Escapa contenido basico para celdas Markdown."""
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
