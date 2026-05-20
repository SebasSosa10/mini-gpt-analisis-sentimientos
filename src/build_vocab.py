"""Construccion del vocabulario desde el split de entrenamiento."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src import config
from src.tokenizer import SimpleTokenizer
from src.utils import ensure_dir, save_json


def load_train_data(path: str | Path) -> pd.DataFrame:
    """Carga train.csv y valida columnas minimas."""
    train_path = Path(path)
    if not train_path.is_absolute():
        train_path = config.PROJECT_ROOT / train_path

    if not train_path.exists():
        raise FileNotFoundError(
            f"No existe el archivo de entrenamiento: {train_path}. "
            "Ejecuta primero python -m src.prepare_data."
        )

    train_df = pd.read_csv(train_path)
    required_columns = {"text_clean", "label_id"}
    missing_columns = required_columns - set(train_df.columns)
    if missing_columns:
        raise ValueError(
            "El archivo de entrenamiento no tiene las columnas requeridas: "
            f"{sorted(missing_columns)}"
        )

    return train_df


def build_tokenizer(
    train_df: pd.DataFrame,
    text_column: str,
    max_vocab_size: int,
    min_freq: int,
) -> SimpleTokenizer:
    """Crea y ajusta el tokenizer solo con textos de entrenamiento."""
    if text_column not in train_df.columns:
        raise ValueError(
            f"No existe la columna de texto '{text_column}' en train.csv. "
            f"Columnas disponibles: {list(train_df.columns)}"
        )

    tokenizer = SimpleTokenizer(
        max_vocab_size=max_vocab_size,
        min_freq=min_freq,
        lowercase=True,
    )
    tokenizer.fit(train_df[text_column].fillna("").astype(str))
    return tokenizer


def analyze_tokenized_lengths(
    tokenizer: SimpleTokenizer,
    texts: pd.Series,
    max_context_length: int = config.MAX_CONTEXT_LENGTH,
) -> dict[str, Any]:
    """Calcula estadisticas de longitud tokenizada con tokens especiales."""
    lengths = np.array(
        [
            len(
                tokenizer.encode(
                    text,
                    add_special_tokens=True,
                    padding=False,
                    truncation=False,
                )
            )
            for text in texts.fillna("").astype(str)
        ],
        dtype=np.int64,
    )

    if lengths.size == 0:
        return {}

    exceeds = lengths > max_context_length
    return {
        "num_texts": int(lengths.size),
        "mean": round(float(np.mean(lengths)), 4),
        "median": round(float(np.median(lengths)), 4),
        "min": int(np.min(lengths)),
        "max": int(np.max(lengths)),
        "p25": round(float(np.percentile(lengths, 25)), 4),
        "p75": round(float(np.percentile(lengths, 75)), 4),
        "p90": round(float(np.percentile(lengths, 90)), 4),
        "p95": round(float(np.percentile(lengths, 95)), 4),
        "max_context_length": int(max_context_length),
        "texts_over_max_context": int(np.sum(exceeds)),
        "percent_over_max_context": round(float(np.mean(exceeds) * 100), 4),
    }


def generate_tokenization_report(
    train_path: str | Path,
    tokenizer: SimpleTokenizer,
    length_stats: dict[str, Any],
    examples: list[dict[str, Any]],
    max_context_length: int,
    output_path: str | Path,
) -> None:
    """Genera reporte Markdown para documentar tokenizacion y vocabulario."""
    lines: list[str] = []
    lines.append("# Resumen de tokenizacion y vocabulario")
    lines.append("")
    lines.append("## Archivo de entrenamiento usado")
    lines.append("")
    lines.append(f"- Archivo: `{relative_to_project(train_path)}`")
    lines.append(
        "- El vocabulario se construyo solo con `train.csv` para evitar fuga de "
        "informacion desde validacion o prueba."
    )
    lines.append("")

    lines.append("## Tipo de tokenizacion")
    lines.append("")
    lines.append(
        "- Tokenizador didactico basado en expresiones regulares, con separacion "
        "de palabras, numeros y puntuacion basica."
    )
    lines.append("- No usa tokenizadores externos como HuggingFace.")
    lines.append("")

    lines.append("## Tokens especiales")
    lines.append("")
    special_rows = [
        (tokenizer.pad_token, tokenizer.pad_token_id),
        (tokenizer.unk_token, tokenizer.unk_token_id),
        (tokenizer.bos_token, tokenizer.bos_token_id),
        (tokenizer.eos_token, tokenizer.eos_token_id),
    ]
    lines.extend(markdown_table(["Token", "ID"], special_rows))
    lines.append("")

    lines.append("## Tamano del vocabulario")
    lines.append("")
    lines.append(f"- Tamano final: {tokenizer.vocab_size}")
    lines.append(f"- Tamano maximo configurado: {tokenizer.max_vocab_size}")
    lines.append("")

    lines.append("## Frecuencia minima")
    lines.append("")
    lines.append(f"- Frecuencia minima incluida: {tokenizer.min_freq}")
    lines.append("")

    lines.append("## Longitud maxima de contexto")
    lines.append("")
    lines.append(f"- MAX_CONTEXT_LENGTH: {max_context_length}")
    lines.append(
        "- Las secuencias mas largas se truncaran en el Dataset; las mas cortas "
        "recibiran padding."
    )
    lines.append("")

    lines.append("## Estadisticas de longitud tokenizada")
    lines.append("")
    lines.extend(markdown_table(["Metrica", "Valor"], length_stats.items()))
    lines.append("")

    lines.append("## Ejemplos de tokenizacion")
    lines.append("")
    example_rows = [
        (
            example["text"],
            " ".join(example["tokens"]),
            example["input_ids"],
            example["decoded"],
        )
        for example in examples
    ]
    lines.extend(markdown_table(["Texto", "Tokens", "IDs", "Decodificado"], example_rows))
    lines.append("")

    lines.append("## Observaciones para el informe academico")
    lines.append("")
    lines.append(
        "- El token `<PAD>` permite formar batches de longitud fija y se marca "
        "con 0 en `attention_mask`."
    )
    lines.append(
        "- Los tokens `<BOS>` y `<EOS>` delimitan el inicio y fin de cada "
        "secuencia."
    )
    lines.append(
        "- Los tokens fuera del vocabulario se reemplazan por `<UNK>`."
    )
    lines.append(
        "- La siguiente fase puede usar `input_ids` para embeddings de tokens y "
        "`attention_mask` para ignorar padding."
    )
    lines.append("")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def save_tokenizer_outputs(
    tokenizer: SimpleTokenizer,
    report_data: dict[str, Any],
    markdown_report_path: str | Path,
) -> dict[str, str]:
    """Guarda tokenizer y reportes de tokenizacion."""
    ensure_dir(config.PROCESSED_DATA_DIR)
    ensure_dir(config.REPORTS_DIR)

    tokenizer.save(config.TOKENIZER_PATH)
    save_json(report_data, config.TOKENIZATION_REPORT_PATH)

    return {
        "tokenizer_json": str(relative_to_project(config.TOKENIZER_PATH)),
        "tokenization_report_json": str(
            relative_to_project(config.TOKENIZATION_REPORT_PATH)
        ),
        "tokenization_summary_md": str(relative_to_project(markdown_report_path)),
    }


def main() -> int:
    """Punto de entrada para construir el vocabulario."""
    parser = argparse.ArgumentParser(
        description="Construye el vocabulario desde data/processed/train.csv."
    )
    parser.add_argument(
        "--train",
        default=str(config.PROCESSED_DATA_DIR / "train.csv"),
        help="Ruta de train.csv procesado.",
    )
    parser.add_argument("--text-column", default="text_clean")
    parser.add_argument("--max-vocab-size", type=int, default=config.MAX_VOCAB_SIZE)
    parser.add_argument("--min-freq", type=int, default=config.MIN_TOKEN_FREQ)
    parser.add_argument(
        "--max-context-length",
        type=int,
        default=config.MAX_CONTEXT_LENGTH,
    )
    parser.add_argument(
        "--sample-text",
        default=None,
        help="Texto opcional para mostrar una tokenizacion de ejemplo.",
    )
    args = parser.parse_args()

    try:
        train_df = load_train_data(args.train)
        tokenizer = build_tokenizer(
            train_df=train_df,
            text_column=args.text_column,
            max_vocab_size=args.max_vocab_size,
            min_freq=args.min_freq,
        )
        length_stats = analyze_tokenized_lengths(
            tokenizer=tokenizer,
            texts=train_df[args.text_column],
            max_context_length=args.max_context_length,
        )
        examples = build_examples(
            tokenizer=tokenizer,
            train_df=train_df,
            text_column=args.text_column,
            sample_text=args.sample_text,
            max_context_length=args.max_context_length,
        )

        report_data = {
            "train_file": str(relative_to_project(args.train)),
            "text_column": args.text_column,
            "tokenizer_type": "regex_word_punctuation",
            "vocab_size": tokenizer.vocab_size,
            "max_vocab_size": tokenizer.max_vocab_size,
            "min_freq": tokenizer.min_freq,
            "lowercase": tokenizer.lowercase,
            "max_context_length": args.max_context_length,
            "special_tokens": {
                tokenizer.pad_token: tokenizer.pad_token_id,
                tokenizer.unk_token: tokenizer.unk_token_id,
                tokenizer.bos_token: tokenizer.bos_token_id,
                tokenizer.eos_token: tokenizer.eos_token_id,
            },
            "length_stats": length_stats,
            "top_tokens": tokenizer.get_token_frequencies_summary(top_n=30),
            "examples": examples,
        }

        generate_tokenization_report(
            train_path=args.train,
            tokenizer=tokenizer,
            length_stats=length_stats,
            examples=examples,
            max_context_length=args.max_context_length,
            output_path=config.TOKENIZATION_SUMMARY_PATH,
        )
        generated_paths = save_tokenizer_outputs(
            tokenizer=tokenizer,
            report_data={**report_data, "generated_files": {}},
            markdown_report_path=config.TOKENIZATION_SUMMARY_PATH,
        )
        report_data["generated_files"] = generated_paths
        save_json(report_data, config.TOKENIZATION_REPORT_PATH)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print("")
    print("Vocabulario construido correctamente.")
    print(f"- Tamano del vocabulario: {tokenizer.vocab_size}")
    print(
        "- Tokens especiales: "
        f"{tokenizer.pad_token}={tokenizer.pad_token_id}, "
        f"{tokenizer.unk_token}={tokenizer.unk_token_id}, "
        f"{tokenizer.bos_token}={tokenizer.bos_token_id}, "
        f"{tokenizer.eos_token}={tokenizer.eos_token_id}"
    )
    if examples:
        print(f"- Ejemplo: {examples[0]['text']}")
        print(f"- Tokens: {examples[0]['tokens']}")
        print(f"- IDs: {examples[0]['input_ids']}")
    print("- Archivos generados:")
    for path in generated_paths.values():
        print(f"  - {path}")
    return 0


def build_examples(
    tokenizer: SimpleTokenizer,
    train_df: pd.DataFrame,
    text_column: str,
    sample_text: str | None,
    max_context_length: int,
) -> list[dict[str, Any]]:
    """Construye ejemplos compactos de tokenizacion."""
    texts: list[str] = []
    if sample_text:
        texts.append(sample_text)
    texts.extend(train_df[text_column].dropna().astype(str).head(2).tolist())

    examples: list[dict[str, Any]] = []
    for text in texts[:3]:
        tokens = tokenizer.basic_tokenize(text)
        input_ids = tokenizer.encode(
            text,
            max_length=max_context_length,
            add_special_tokens=True,
            padding=False,
            truncation=True,
        )
        examples.append(
            {
                "text": text,
                "tokens": tokens,
                "input_ids": input_ids[:40],
                "decoded": tokenizer.decode(input_ids),
            }
        )
    return examples


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
    """Escapa valores para tablas Markdown."""
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
