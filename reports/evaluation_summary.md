# Evaluacion final del Mini-GPT

## Checkpoint evaluado

- Checkpoint: `models/mini_gpt_sentiment_best_weighted.pt`
- Epoca del checkpoint: 5
- Metrica de seleccion: val_macro_f1
- Valor de seleccion: 0.897870175080072

## Dataset usado para evaluacion

- Split evaluado: `test`
- `test.csv` se usa solo para evaluacion final.
- El modelo fue seleccionado con validacion, no con test.

## Metricas generales

| Metrica | Valor |
| --- | --- |
| accuracy | 0.9068152749862963 |
| precision_macro | 0.9017496891969856 |
| recall_macro | 0.8990101255187404 |
| f1_macro | 0.9003373251387814 |
| f1_weighted | 0.9066558796603781 |

## Metricas por clase

| Clase | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| negative | 0.9211415877921576 | 0.9303970344356648 | 0.9257461781121087 | 20502.0 |
| positive | 0.8823577906018136 | 0.8676232166018158 | 0.8749284721654541 | 12336.0 |

## Matriz de confusion

Filas = etiqueta real; columnas = prediccion.
| Real/Pred | negative | positive |
| --- | --- | --- |
| negative | 19075 | 1427 |
| positive | 1633 | 10703 |

## Analisis de errores

- Total de errores: 3060
- Falsos positivos: 1427
- Falsos negativos: 1633
- Errores en textos muy cortos: 199
- Errores con alta confianza: 1254

## Interpretacion de resultados

- Macro-F1 es importante porque el dataset binario esta desbalanceado.
- La matriz de confusion permite observar si el modelo confunde mas positivos con negativos o negativos con positivos.

## Limitaciones

- Los errores pueden deberse a textos ambiguos, sarcasmo, reseñas muy cortas, vocabulario poco frecuente o ruido del dataset.
- Esta evaluacion no modifica el checkpoint ni realiza nuevo entrenamiento.

## Archivos generados

- `reports/evaluation_metrics.json`
- `reports/test_predictions.csv`
- `reports/figures/confusion_matrix_test.png`
- `reports/figures/final_metrics_barplot.png`
- `reports/error_analysis.csv`
- `reports/evaluation_summary.md`
