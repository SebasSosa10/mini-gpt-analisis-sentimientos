"""Pruebas unitarias para utilidades de evaluacion."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluate import (
    analyze_errors,
    compute_metrics,
    get_id_to_label,
    save_predictions_csv,
)


def test_get_id_to_label() -> None:
    """Valida inversion de label_mapping."""
    mapping = {"negative": 0, "positive": 1}

    assert get_id_to_label(mapping) == {0: "negative", 1: "positive"}


def test_compute_metrics_binary() -> None:
    """Valida metricas binarias sobre arrays pequenos."""
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 1]

    metrics = compute_metrics(y_true, y_pred, ["negative", "positive"])

    assert metrics["accuracy"] == 0.75
    assert "precision_macro" in metrics
    assert "recall_macro" in metrics
    assert "f1_macro" in metrics
    assert metrics["confusion_matrix"] == [[1, 1], [0, 2]]


def test_save_predictions_csv(tmp_path) -> None:
    """Valida creacion del CSV de predicciones."""
    split_df = pd.DataFrame(
        {
            "text": ["bad food", "good food"],
            "text_clean": ["bad food", "good food"],
            "labels": [1, 3],
            "ratings_overall": [1, 5],
        }
    )
    output_path = tmp_path / "predictions.csv"

    predictions_df = save_predictions_csv(
        split_df=split_df,
        y_true=[0, 1],
        y_pred=[0, 1],
        probabilities=[[0.9, 0.1], [0.2, 0.8]],
        id_to_label={0: "negative", 1: "positive"},
        output_path=output_path,
    )

    assert output_path.exists()
    assert "prob_negative" in predictions_df.columns
    assert "prob_positive" in predictions_df.columns
    assert predictions_df["is_correct"].tolist() == [True, True]


def test_analyze_errors_creates_file(tmp_path) -> None:
    """Valida que el analisis de errores guarde archivo."""
    predictions_df = pd.DataFrame(
        {
            "text_clean": ["bad food", "good food", "ok"],
            "true_label": ["negative", "positive", "negative"],
            "predicted_label": ["positive", "positive", "negative"],
            "confidence": [0.95, 0.80, 0.70],
            "prob_negative": [0.05, 0.20, 0.70],
            "prob_positive": [0.95, 0.80, 0.30],
            "is_correct": [False, True, True],
        }
    )
    output_path = tmp_path / "errors.csv"

    stats = analyze_errors(predictions_df, output_path)

    assert output_path.exists()
    assert stats["total_errors"] == 1
    assert stats["false_positives"] == 1
    assert stats["false_negatives"] == 0
