"""Analisis exploratorio del dataset crudo.

Este modulo inspecciona archivos CSV ubicados en data/raw/ y genera reportes
utiles para documentar el dataset antes de la fase de preparacion de datos.
No descarga datos, no modifica data/raw/ y no asume nombres definitivos de
columnas.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import seaborn as sns
except ImportError:  # pragma: no cover - depende del entorno local.
    sns = None

from pandas.api.types import is_numeric_dtype, is_object_dtype, is_string_dtype

from src import config
from src.utils import ensure_dir


TEXT_NAME_HINTS = (
    "review",
    "text",
    "comment",
    "comments",
    "description",
    "content",
    "feedback",
    "translated",
    "english",
    "bangla",
    "banglish",
)

STRONG_TEXT_NAME_HINTS = (
    "text",
    "comment",
    "comments",
    "description",
    "content",
    "feedback",
    "translated",
    "english",
    "bangla",
    "banglish",
)

NON_TEXT_NAME_HINTS = (
    "id",
    "uuid",
    "name",
    "date",
    "time",
    "created",
    "updated",
    "number",
    "url",
    "link",
    "phone",
    "email",
    "city",
    "latitude",
    "longitude",
)

LABEL_NAME_HINTS = (
    "sentiment",
    "label",
    "class",
    "polarity",
    "target",
    "rating",
    "ratings",
    "score",
    "overall",
)

RATING_NAME_HINTS = ("rating", "ratings", "score", "overall")


def find_csv_files(raw_data_dir: str | Path) -> list[Path]:
    """Busca archivos CSV en el directorio de datos crudos."""
    raw_dir = Path(raw_data_dir)
    csv_files = sorted(path for path in raw_dir.glob("*.csv") if path.is_file())

    if not csv_files:
        raise FileNotFoundError(
            "No se encontraron archivos CSV en data/raw/. Descarga manualmente "
            "el dataset desde Kaggle y ubica el archivo .csv en data/raw/."
        )

    return csv_files


def resolve_input_file(input_path: str | Path | None = None) -> Path:
    """Resuelve el archivo CSV a analizar."""
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
    """Carga un CSV con pandas e intenta alternativas de encoding si fallan."""
    path = Path(csv_path)

    try:
        return pd.read_csv(path, low_memory=False)
    except UnicodeDecodeError:
        encoding_errors: list[str] = []
        for encoding in ("utf-8", "latin-1"):
            try:
                return pd.read_csv(path, encoding=encoding, low_memory=False)
            except UnicodeDecodeError as exc:
                encoding_errors.append(f"{encoding}: {exc}")
            except Exception as exc:
                encoding_errors.append(f"{encoding}: {exc}")

        raise ValueError(
            "No se pudo leer el CSV por problemas de codificacion. "
            f"Intentos realizados: {' | '.join(encoding_errors)}"
        ) from None
    except Exception as exc:
        raise ValueError(f"No se pudo cargar el CSV con pandas: {exc}") from None


def basic_profile(df: pd.DataFrame) -> dict[str, Any]:
    """Genera un perfil basico del DataFrame."""
    missing_values = df.isna().sum()
    missing_percent = (missing_values / max(len(df), 1)) * 100
    memory_usage_mb = df.memory_usage(deep=True).sum() / (1024**2)

    return {
        "num_rows": int(df.shape[0]),
        "num_columns": int(df.shape[1]),
        "columns": [str(column) for column in df.columns],
        "dtypes": {str(column): str(dtype) for column, dtype in df.dtypes.items()},
        "missing_values": {
            str(column): int(value) for column, value in missing_values.items()
        },
        "missing_percent": {
            str(column): round(float(value), 4)
            for column, value in missing_percent.items()
        },
        "duplicate_rows": int(df.duplicated().sum()),
        "memory_usage_mb": round(float(memory_usage_mb), 4),
    }


def detect_text_columns(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Detecta columnas candidatas de texto sin asumir una seleccion definitiva."""
    candidates: list[tuple[float, dict[str, Any]]] = []

    for column in df.columns:
        series = df[column].dropna()
        if series.empty:
            continue

        column_name = str(column)
        lower_name = column_name.lower()
        name_matches = sum(hint in lower_name for hint in TEXT_NAME_HINTS)
        strong_name_match = any(hint in lower_name for hint in STRONG_TEXT_NAME_HINTS)
        non_text_name_match = any(hint in lower_name for hint in NON_TEXT_NAME_HINTS)
        dtype_is_text = is_object_dtype(series) or is_string_dtype(series)

        if not dtype_is_text and name_matches == 0:
            continue

        if non_text_name_match and not strong_name_match:
            continue

        text_values = series.astype(str).str.strip()
        text_values = text_values[text_values != ""]
        if text_values.empty:
            continue

        lengths = text_values.str.len()
        avg_length = float(lengths.mean())
        unique_ratio = float(text_values.nunique() / max(len(text_values), 1))

        if name_matches == 0 and (avg_length < 30 or unique_ratio < 0.20):
            continue

        score = (name_matches * 10) + min(avg_length / 20, 10) + unique_ratio
        sample_values = [
            truncate_text(value)
            for value in text_values.drop_duplicates().head(5).tolist()
        ]

        candidates.append(
            (
                score,
                {
                    "column": column_name,
                    "non_null_count": int(text_values.shape[0]),
                    "avg_length": round(avg_length, 4),
                    "min_length": int(lengths.min()),
                    "max_length": int(lengths.max()),
                    "sample_values": sample_values,
                },
            )
        )

    candidates.sort(key=lambda item: item[0], reverse=True)
    return [candidate for _, candidate in candidates]


def detect_label_columns(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Detecta columnas candidatas de etiqueta, sentimiento o rating."""
    candidates: list[tuple[float, dict[str, Any]]] = []

    for column in df.columns:
        series = df[column].dropna()
        if series.empty:
            continue

        column_name = str(column)
        lower_name = column_name.lower()
        name_matches = sum(hint in lower_name for hint in LABEL_NAME_HINTS)
        sentiment_name_match = "sentiment" in lower_name
        label_name_match = any(
            hint in lower_name for hint in ("label", "class", "polarity", "target")
        )
        rating_name_match = any(hint in lower_name for hint in RATING_NAME_HINTS)
        unique_count = int(series.nunique(dropna=True))
        dtype_name = str(df[column].dtype)
        numeric = bool(is_numeric_dtype(series))
        low_cardinality = unique_count <= 20 and unique_count < max(len(series), 1)
        categorical_like = (
            (is_object_dtype(series) or is_string_dtype(series)) and low_cardinality
        )
        sentiment_values_match = looks_like_sentiment_values(series)

        if name_matches == 0 and not sentiment_values_match:
            continue

        score = 0.0
        if sentiment_name_match:
            score += 45
        if label_name_match:
            score += 40
        if rating_name_match:
            score += 20
        if name_matches:
            score += 3
        if low_cardinality:
            score += 5
        if sentiment_values_match:
            score += 6
        if numeric and rating_name_match:
            score += 3

        value_counts = series.value_counts(dropna=False).head(20)
        candidate: dict[str, Any] = {
            "column": column_name,
            "dtype": dtype_name,
            "unique_count": unique_count,
            "value_counts_top": {
                stringify_value(index): int(value)
                for index, value in value_counts.items()
            },
        }

        if numeric and rating_name_match:
            candidate["possible_rating_mapping"] = (
                "Hipotesis pendiente de confirmar: rating 1-2 negativo, "
                "3 neutro, 4-5 positivo."
            )

        candidates.append((score, candidate))

    candidates.sort(key=lambda item: (item[0], -item[1]["unique_count"]), reverse=True)
    return [candidate for _, candidate in candidates]


def analyze_text_lengths(df: pd.DataFrame, text_column: str | None) -> dict[str, Any]:
    """Calcula estadisticas de longitud de texto en caracteres y palabras."""
    if not text_column or text_column not in df.columns:
        print("Advertencia: no hay columna de texto seleccionable para analizar longitudes.")
        return {}

    text_values = df[text_column].dropna().astype(str).str.strip()
    text_values = text_values[text_values != ""]
    if text_values.empty:
        print("Advertencia: la columna de texto seleccionada no tiene textos validos.")
        return {}

    char_lengths = text_values.str.len()
    word_lengths = text_values.str.split().str.len()

    return {
        "text_column": text_column,
        "num_texts": int(text_values.shape[0]),
        "characters": describe_numeric_series(char_lengths),
        "words": describe_numeric_series(word_lengths),
    }


def plot_missing_values(df: pd.DataFrame, output_dir: str | Path) -> Path | None:
    """Grafica las 20 columnas con mas valores nulos."""
    output_path = Path(output_dir) / "missing_values_top20.png"
    missing_values = df.isna().sum().sort_values(ascending=False).head(20)
    missing_values = missing_values[missing_values > 0]

    if missing_values.empty:
        print("No se detectaron valores nulos; se omite missing_values_top20.png.")
        return None

    plt.figure(figsize=(12, 6))
    if sns is not None:
        sns.barplot(x=missing_values.values, y=missing_values.index, color="#4C78A8")
    else:
        plt.barh(missing_values.index.astype(str), missing_values.values, color="#4C78A8")
        plt.gca().invert_yaxis()

    plt.title("Columnas con mas valores nulos")
    plt.xlabel("Cantidad de valores nulos")
    plt.ylabel("Columna")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def plot_text_length_distribution(
    df: pd.DataFrame, text_column: str | None, output_dir: str | Path
) -> Path | None:
    """Grafica la distribucion de longitud de textos en caracteres."""
    if not text_column or text_column not in df.columns:
        print("No hay columna de texto para graficar longitudes.")
        return None

    output_path = Path(output_dir) / "text_length_distribution.png"
    lengths = df[text_column].dropna().astype(str).str.strip().str.len()
    lengths = lengths[lengths > 0]

    if lengths.empty:
        print("No hay textos validos para graficar longitudes.")
        return None

    plt.figure(figsize=(10, 6))
    if sns is not None:
        sns.histplot(lengths, bins=50, color="#59A14F")
    else:
        plt.hist(lengths, bins=50, color="#59A14F", edgecolor="white")

    plt.title(f"Distribucion de longitud de textos: {text_column}")
    plt.xlabel("Cantidad de caracteres")
    plt.ylabel("Frecuencia")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def plot_label_distribution(
    df: pd.DataFrame, label_column: str | None, output_dir: str | Path
) -> Path | None:
    """Grafica la distribucion de clases para una columna candidata."""
    if not label_column or label_column not in df.columns:
        print("No hay columna de etiqueta para graficar distribucion.")
        return None

    unique_count = int(df[label_column].nunique(dropna=True))
    if unique_count > 50:
        print(
            "Advertencia: la columna candidata de etiqueta tiene demasiados "
            f"valores unicos ({unique_count}); probablemente no es una clase."
        )
        return None

    output_path = Path(output_dir) / "label_distribution.png"
    value_counts = df[label_column].value_counts(dropna=False).head(50)

    plt.figure(figsize=(10, 6))
    if sns is not None:
        sns.barplot(
            x=value_counts.index.astype(str),
            y=value_counts.values,
            color="#F28E2B",
        )
    else:
        plt.bar(value_counts.index.astype(str), value_counts.values, color="#F28E2B")

    plt.title(f"Distribucion de clases candidata: {label_column}")
    plt.xlabel("Valor")
    plt.ylabel("Cantidad")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def plot_rating_distribution(
    df: pd.DataFrame, rating_column: str | None, output_dir: str | Path
) -> Path | None:
    """Grafica la distribucion de una columna numerica de rating."""
    if not rating_column or rating_column not in df.columns:
        print("No hay columna numerica de rating para graficar distribucion.")
        return None

    rating_values = pd.to_numeric(df[rating_column], errors="coerce").dropna()
    if rating_values.empty:
        print("La columna candidata de rating no contiene valores numericos validos.")
        return None

    output_path = Path(output_dir) / "rating_distribution.png"
    value_counts = rating_values.value_counts().sort_index()

    plt.figure(figsize=(8, 5))
    if sns is not None:
        sns.barplot(
            x=value_counts.index.astype(str),
            y=value_counts.values,
            color="#E15759",
        )
    else:
        plt.bar(value_counts.index.astype(str), value_counts.values, color="#E15759")

    plt.title(f"Distribucion de rating candidato: {rating_column}")
    plt.xlabel("Rating")
    plt.ylabel("Cantidad")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def generate_markdown_report(
    profile: dict[str, Any],
    text_candidates: list[dict[str, Any]],
    label_candidates: list[dict[str, Any]],
    selected_text_column: str | None,
    selected_label_column: str | None,
    output_path: str | Path,
) -> None:
    """Genera un reporte Markdown claro y util para el informe academico."""
    output = Path(output_path)
    lines: list[str] = []
    sample_rows = profile.get("sample_rows", [])
    text_length_stats = profile.get("text_length_stats", {})

    lines.append("# Analisis exploratorio del dataset")
    lines.append("")
    lines.append("## Archivo analizado")
    lines.append("")
    lines.append(f"- Archivo: `{profile.get('archivo_analizado', 'No disponible')}`")
    lines.append(
        "- Nota: la seleccion final de columnas de texto y etiqueta se "
        "confirmara en la fase de preparacion de datos."
    )
    lines.append("")

    lines.append("## Dimensiones del dataset")
    lines.append("")
    lines.append(f"- Filas: {profile['num_rows']}")
    lines.append(f"- Columnas: {profile['num_columns']}")
    lines.append(f"- Memoria aproximada: {profile['memory_usage_mb']} MB")
    lines.append("")

    lines.append("## Columnas disponibles")
    lines.append("")
    for column in profile["columns"]:
        lines.append(f"- `{column}`")
    lines.append("")

    lines.append("## Tipos de datos")
    lines.append("")
    lines.extend(markdown_table(["Columna", "Tipo"], profile["dtypes"].items()))
    lines.append("")

    lines.append("## Valores nulos")
    lines.append("")
    missing_rows = [
        (
            column,
            profile["missing_values"][column],
            f"{profile['missing_percent'][column]:.4f}%",
        )
        for column in profile["columns"]
        if profile["missing_values"][column] > 0
    ]
    if missing_rows:
        lines.extend(markdown_table(["Columna", "Nulos", "Porcentaje"], missing_rows))
    else:
        lines.append("No se detectaron valores nulos en el dataset cargado.")
    lines.append("")

    lines.append("## Filas duplicadas")
    lines.append("")
    lines.append(f"- Filas duplicadas detectadas: {profile['duplicate_rows']}")
    lines.append("")

    lines.append("## Posibles columnas de texto")
    lines.append("")
    if text_candidates:
        rows = [
            (
                candidate["column"],
                candidate["non_null_count"],
                candidate["avg_length"],
                candidate["min_length"],
                candidate["max_length"],
                " | ".join(candidate["sample_values"][:3]),
            )
            for candidate in text_candidates
        ]
        lines.extend(
            markdown_table(
                [
                    "Columna",
                    "No nulos",
                    "Longitud media",
                    "Min",
                    "Max",
                    "Muestras",
                ],
                rows,
            )
        )
        lines.append("")
        lines.append(f"- Columna de texto tentativa: `{selected_text_column}`")
    else:
        lines.append("No se pudo detectar automaticamente una columna de texto.")
    lines.append("")

    if text_length_stats:
        lines.append("### Estadisticas de longitud de la columna tentativa")
        lines.append("")
        lines.append(f"- Columna: `{text_length_stats['text_column']}`")
        lines.append(f"- Textos validos: {text_length_stats['num_texts']}")
        lines.append("")
        lines.append("Caracteres:")
        lines.extend(
            markdown_table(
                ["Metrica", "Valor"],
                text_length_stats["characters"].items(),
            )
        )
        lines.append("")
        lines.append("Palabras aproximadas:")
        lines.extend(
            markdown_table(["Metrica", "Valor"], text_length_stats["words"].items())
        )
        lines.append("")

    lines.append("## Posibles columnas de etiqueta o rating")
    lines.append("")
    if label_candidates:
        rows = [
            (
                candidate["column"],
                candidate["dtype"],
                candidate["unique_count"],
                " | ".join(
                    f"{key}: {value}"
                    for key, value in list(candidate["value_counts_top"].items())[:5]
                ),
                candidate.get("possible_rating_mapping", "Sin hipotesis automatica"),
            )
            for candidate in label_candidates
        ]
        lines.extend(
            markdown_table(
                ["Columna", "Tipo", "Valores unicos", "Valores frecuentes", "Nota"],
                rows,
            )
        )
        lines.append("")
        lines.append(f"- Columna de etiqueta/rating tentativa: `{selected_label_column}`")
    else:
        lines.append(
            "No se pudo detectar automaticamente una columna de sentimiento, "
            "etiqueta o rating. Se recomienda revisar manualmente el CSV."
        )
    lines.append("")

    lines.append("## Ejemplos del dataset")
    lines.append("")
    if sample_rows:
        sample_columns = profile["columns"][: min(6, len(profile["columns"]))]
        rows = [
            tuple(truncate_text(row.get(column, "")) for column in sample_columns)
            for row in sample_rows
        ]
        lines.extend(markdown_table(sample_columns, rows))
    else:
        lines.append("No hay ejemplos disponibles para mostrar.")
    lines.append("")

    lines.append("## Problemas potenciales encontrados")
    lines.append("")
    lines.extend(build_potential_issues(profile, text_candidates, label_candidates))
    lines.append("")

    lines.append("## Recomendaciones para la siguiente fase")
    lines.append("")
    lines.append(
        "- Confirmar manualmente la columna de texto y la columna objetivo antes "
        "de limpiar o transformar datos."
    )
    lines.append(
        "- Definir si la tarea sera binaria o multiclase segun las etiquetas "
        "reales disponibles."
    )
    lines.append(
        "- Si solo existe rating numerico, validar academicamente el mapeo de "
        "rating a sentimiento antes de usarlo."
    )
    lines.append(
        "- Revisar textos vacios, duplicados, ruido, errores ortograficos y "
        "posibles sesgos antes del entrenamiento."
    )
    lines.append(
        "- Guardar una version procesada en data/processed/ durante la Fase 3."
    )
    lines.append("")

    output.write_text("\n".join(lines), encoding="utf-8")


def save_json_profile(profile_data: dict[str, Any], output_path: str | Path) -> None:
    """Guarda el perfil estructurado en formato JSON."""
    output = Path(output_path)
    output.write_text(
        json.dumps(make_json_safe(profile_data), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> int:
    """Ejecuta el analisis exploratorio desde consola."""
    parser = argparse.ArgumentParser(
        description="Analisis exploratorio del CSV ubicado en data/raw/."
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Ruta opcional del archivo CSV a analizar.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=5,
        help="Cantidad de filas de ejemplo para incluir en el reporte.",
    )
    args = parser.parse_args()

    try:
        csv_path = resolve_input_file(args.input)
        print(f"Archivo seleccionado: {relative_to_project(csv_path)}")
        df = load_dataset(csv_path)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    reports_dir = ensure_dir(config.REPORTS_DIR)
    figures_dir = ensure_dir(config.FIGURES_DIR)

    profile = basic_profile(df)
    text_candidates = detect_text_columns(df)
    label_candidates = detect_label_columns(df)

    selected_text_column = (
        text_candidates[0]["column"] if text_candidates else None
    )
    selected_label_column = (
        label_candidates[0]["column"] if label_candidates else None
    )
    selected_rating_column = choose_rating_column(label_candidates)

    text_length_stats = analyze_text_lengths(df, selected_text_column)

    generated_figures: dict[str, str] = {}
    for name, path in {
        "missing_values_top20": plot_missing_values(df, figures_dir),
        "text_length_distribution": plot_text_length_distribution(
            df, selected_text_column, figures_dir
        ),
        "label_distribution": plot_label_distribution(
            df, selected_label_column, figures_dir
        ),
        "rating_distribution": plot_rating_distribution(
            df, selected_rating_column, figures_dir
        ),
    }.items():
        if path is not None:
            generated_figures[name] = str(relative_to_project(path))

    sample_rows = get_sample_rows(df, max(args.sample_size, 0))

    enriched_profile = {
        **profile,
        "archivo_analizado": str(relative_to_project(csv_path)),
        "sample_rows": sample_rows,
        "text_length_stats": text_length_stats,
        "generated_figures": generated_figures,
    }

    markdown_path = reports_dir / "dataset_summary.md"
    json_path = reports_dir / "dataset_profile.json"

    generate_markdown_report(
        enriched_profile,
        text_candidates,
        label_candidates,
        selected_text_column,
        selected_label_column,
        markdown_path,
    )

    save_json_profile(
        {
            "archivo_analizado": str(relative_to_project(csv_path)),
            "basic_profile": profile,
            "text_candidates": text_candidates,
            "label_candidates": label_candidates,
            "selected_text_column": selected_text_column,
            "selected_label_column": selected_label_column,
            "selected_rating_column": selected_rating_column,
            "text_length_stats": text_length_stats,
            "sample_rows": sample_rows,
            "generated_reports": {
                "markdown": str(relative_to_project(markdown_path)),
                "json": str(relative_to_project(json_path)),
                "figures": generated_figures,
            },
        },
        json_path,
    )

    print("")
    print("Analisis exploratorio completado.")
    print(f"- Archivo analizado: {relative_to_project(csv_path)}")
    print(f"- Filas: {profile['num_rows']}")
    print(f"- Columnas: {profile['num_columns']}")
    print(
        "- Posibles columnas de texto: "
        + format_candidate_names(text_candidates)
    )
    print(
        "- Posibles columnas de etiqueta/rating: "
        + format_candidate_names(label_candidates)
    )
    print(f"- Reporte Markdown: {relative_to_project(markdown_path)}")
    print(f"- Perfil JSON: {relative_to_project(json_path)}")
    if generated_figures:
        print("- Figuras generadas:")
        for figure_path in generated_figures.values():
            print(f"  - {figure_path}")
    else:
        print("- Figuras generadas: ninguna aplicable con las columnas detectadas.")

    return 0


def describe_numeric_series(series: pd.Series) -> dict[str, float | int]:
    """Resume una serie numerica con estadisticas descriptivas."""
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return {}

    return {
        "mean": round(float(values.mean()), 4),
        "median": round(float(values.median()), 4),
        "min": int(values.min()),
        "max": int(values.max()),
        "p25": round(float(np.percentile(values, 25)), 4),
        "p75": round(float(np.percentile(values, 75)), 4),
        "p90": round(float(np.percentile(values, 90)), 4),
        "p95": round(float(np.percentile(values, 95)), 4),
    }


def choose_rating_column(label_candidates: list[dict[str, Any]]) -> str | None:
    """Elige una columna numerica de rating si existe entre las candidatas."""
    for candidate in label_candidates:
        column = candidate["column"]
        lower_name = column.lower()
        if candidate.get("possible_rating_mapping") and any(
            hint in lower_name for hint in RATING_NAME_HINTS
        ):
            return column
    return None


def looks_like_sentiment_values(series: pd.Series) -> bool:
    """Detecta valores que parecen etiquetas de sentimiento."""
    if int(series.nunique(dropna=True)) > 20:
        return False

    values = series.dropna().drop_duplicates().head(50)
    normalized_values = {str(value).strip().lower() for value in values}
    sentiment_words = {
        "positive",
        "negative",
        "neutral",
        "pos",
        "neg",
        "neu",
        "good",
        "bad",
        "mixed",
        "positivo",
        "negativo",
        "neutro",
    }

    if normalized_values & sentiment_words:
        return True

    for value in normalized_values:
        if len(value) > 30:
            continue
        if any(word in value for word in sentiment_words):
            return True

    return False


def get_sample_rows(df: pd.DataFrame, sample_size: int) -> list[dict[str, str]]:
    """Obtiene filas de ejemplo en formato seguro para Markdown y JSON."""
    if sample_size <= 0 or df.empty:
        return []

    rows = df.head(sample_size).replace({np.nan: None}).to_dict(orient="records")
    return [
        {str(key): stringify_value(value) for key, value in row.items()}
        for row in rows
    ]


def build_potential_issues(
    profile: dict[str, Any],
    text_candidates: list[dict[str, Any]],
    label_candidates: list[dict[str, Any]],
) -> list[str]:
    """Construye observaciones basadas solo en datos detectados."""
    issues: list[str] = []

    missing_columns = [
        column
        for column, value in profile["missing_values"].items()
        if int(value) > 0
    ]
    if missing_columns:
        issues.append(
            "- Existen valores nulos en algunas columnas; deben tratarse antes "
            "de entrenar el modelo."
        )
    else:
        issues.append("- No se detectaron valores nulos en esta carga del CSV.")

    if profile["duplicate_rows"] > 0:
        issues.append(
            f"- Se detectaron {profile['duplicate_rows']} filas duplicadas; "
            "conviene revisar si deben eliminarse."
        )
    else:
        issues.append("- No se detectaron filas duplicadas exactas.")

    if not text_candidates:
        issues.append(
            "- No se detecto automaticamente una columna de texto; se requiere "
            "revision manual."
        )
    else:
        text_stats = profile.get("text_length_stats", {})
        word_stats = text_stats.get("words", {}) if text_stats else {}
        if word_stats and word_stats.get("median", 0) <= 3:
            issues.append(
                "- La mediana de palabras de la columna tentativa es baja; "
                "pueden existir textos muy cortos."
            )
        issues.append(
            "- La columna de texto detectada es tentativa; debe confirmarse con "
            "los ejemplos reales."
        )

    if not label_candidates:
        issues.append(
            "- No se detecto automaticamente una columna de etiqueta o rating; "
            "la variable objetivo debe definirse manualmente."
        )
    else:
        first_candidate = label_candidates[0]
        counts = list(first_candidate["value_counts_top"].values())
        if counts and sum(counts) > 0 and max(counts) / sum(counts) >= 0.6:
            issues.append(
                "- La columna candidata principal puede estar desbalanceada; "
                "conviene revisar la distribucion completa de clases."
            )
        if first_candidate.get("possible_rating_mapping"):
            issues.append(
                "- Existe una hipotesis de convertir rating numerico a "
                "sentimiento, pero debe validarse antes de aplicarla."
            )
        issues.append(
            "- Las clases disponibles deben confirmarse antes de la preparacion "
            "de datos."
        )

    issues.append(
        "- Los problemas de ruido, ortografia, idioma y sesgos deben evaluarse "
        "con una inspeccion cualitativa de ejemplos."
    )
    return issues


def markdown_table(headers: list[str], rows: Any) -> list[str]:
    """Crea una tabla Markdown sencilla."""
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
    """Normaliza valores para celdas Markdown."""
    text = stringify_value(value)
    return text.replace("\n", " ").replace("\r", " ").replace("|", "\\|")


def stringify_value(value: Any) -> str:
    """Convierte valores de pandas/numpy a texto legible."""
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)


def truncate_text(value: Any, max_length: int = 160) -> str:
    """Recorta texto largo para reportes compactos."""
    text = stringify_value(value).replace("\n", " ").replace("\r", " ").strip()
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."


def relative_to_project(path: str | Path) -> Path:
    """Devuelve una ruta relativa al proyecto cuando es posible."""
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(config.PROJECT_ROOT)
    except ValueError:
        return resolved


def format_candidate_names(candidates: list[dict[str, Any]]) -> str:
    """Formatea nombres de candidatas para consola."""
    if not candidates:
        return "ninguna detectada"
    return ", ".join(candidate["column"] for candidate in candidates)


def make_json_safe(value: Any) -> Any:
    """Convierte objetos comunes de pandas/numpy a estructuras JSON serializables."""
    if isinstance(value, dict):
        return {str(key): make_json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [make_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [make_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
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
