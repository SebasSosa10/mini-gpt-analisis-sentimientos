# Mini-GPT para análisis de sentimiento en reseñas de comida

Proyecto académico que implementa un Mini-GPT didáctico para clasificar reseñas de comida como `positive` o `negative`, usando el dataset **BD Food Review Dataset** de Kaggle.

## Descripción general

Este proyecto construye un flujo completo de análisis de sentimiento con Python, PyTorch y NLP. El sistema carga un dataset real, explora sus columnas, limpia textos, prepara etiquetas binarias, construye un vocabulario, tokeniza reseñas, entrena una arquitectura reducida tipo GPT, evalúa el modelo sobre `test.csv` y permite predecir frases nuevas desde consola.

El modelo implementado es un Mini-GPT didáctico: usa componentes de una arquitectura tipo Transformer decoder, pero no es un GPT grande ni un modelo generativo completo. En este proyecto se adapta a clasificación supervisada de sentimiento.

La tarea final es binaria:

- `negative`
- `positive`

El dataset y los ejemplos usados están principalmente en inglés o en texto normalizado en inglés.

## Objetivo académico

El objetivo es aplicar conceptos centrales de Procesamiento de Lenguaje Natural y modelos Transformer:

- Representación de texto.
- Tokenización.
- Construcción de vocabulario.
- Embeddings de tokens.
- Embeddings posicionales.
- Atención causal.
- Bloques Transformer decoder.
- Clasificación supervisada.
- Entrenamiento y evaluación de modelos en PyTorch.

El repositorio `LLMs-from-scratch` de Sebastian Raschka se usó únicamente como referencia conceptual. La estructura, los archivos y la adaptación a clasificación de sentimiento fueron implementados de forma propia para este proyecto.

## Integrantes

| Campo | Valor |
| --- | --- |
| Integrantes | Pablo Garcés Hoyos, Alejandra Mejía Patiño, Joan Sebastián Sosa Bedoya |

## Dataset

| Campo | Valor |
| --- | --- |
| Dataset | BD Food Review Dataset |
| Fuente | Kaggle |
| Archivo esperado | `data/raw/BDFoodSent-334k.csv` |
| Filas originales | 334119 |
| Columnas | 29 |
| Columna de texto | `text` |
| Columna de etiqueta | `labels` |
| Columna comparada | `ratings_overall` |
| Desacuerdos `labels` vs `ratings_overall` | 0 |
| Tarea final | Clasificación binaria `positive` / `negative` |

El CSV debe descargarse manualmente desde Kaggle y ubicarse en:

```text
data/raw/BDFoodSent-334k.csv
```

Enlace del dataset:

```text
https://www.kaggle.com/datasets/sanjidh090/bd-food-review-dataset
```

Distribución final después de la preparación binaria:

| Clase | Cantidad |
| --- | ---: |
| negative | 136676 |
| positive | 82239 |
| Total | 218915 |

Splits procesados:

| Split | Filas | negative | positive |
| --- | ---: | ---: | ---: |
| train | 153240 | 95673 | 57567 |
| val | 32837 | 20501 | 12336 |
| test | 32838 | 20502 | 12336 |

## Resultados principales

Evaluación final sobre `data/processed/test.csv` usando el checkpoint `models/mini_gpt_sentiment_best_weighted.pt`:

| Métrica | Valor |
| --- | ---: |
| Accuracy | 0.906815 |
| Precision macro | 0.901750 |
| Recall macro | 0.899010 |
| F1 macro | 0.900337 |
| F1 weighted | 0.906656 |
| Errores | 3060 |

Otros resultados relevantes:

| Elemento | Valor |
| --- | --- |
| Mejor `val_macro_f1` | 0.897870175080072 |
| Mejor época | 5 |
| Mejor checkpoint | `models/mini_gpt_sentiment_best_weighted.pt` |
| Pruebas | `26 passed` |

## Arquitectura del modelo

El modelo es una arquitectura Mini-GPT para clasificación de sentimiento. Sus componentes principales son:

- Embedding de tokens.
- Embedding posicional aprendido.
- Causal self-attention multi-cabeza.
- Bloques Transformer decoder.
- FeedForward.
- LayerNorm.
- Conexiones residuales.
- Dropout.
- Cabeza final de clasificación.
- Uso del último token válido según `attention_mask`.

Hiperparámetros del modelo final:

| Hiperparámetro | Valor |
| --- | ---: |
| vocab_size | 16612 |
| max_context_length | 128 |
| embed_dim | 128 |
| num_heads | 4 |
| num_layers | 2 |
| dropout | 0.1 |
| num_classes | 2 |
| Parámetros totales | 2539778 |
| Parámetros entrenables | 2539778 |

## Estructura del proyecto

```text
mini-gpt-sentiment/
├── data/
│   ├── raw/
│   └── processed/
├── docs/
├── models/
├── reports/
│   └── figures/
├── src/
│   ├── explore_data.py
│   ├── prepare_data.py
│   ├── tokenizer.py
│   ├── build_vocab.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── tests/
├── requirements.txt
└── README.md
```

## Requisitos

Se recomienda usar Python 3.9 o superior.

Dependencias principales:

- PyTorch
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- tqdm
- pytest

## Instalación

Mac/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows PowerShell:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Preparación del dataset

1. Descargar manualmente el dataset desde Kaggle.
2. Copiar el CSV en `data/raw/`.
3. Verificar que exista el archivo:

```text
data/raw/BDFoodSent-334k.csv
```

Ejecutar exploración del dataset:

```bash
python -m src.explore_data
```

Preparar datos procesados:

```bash
python -m src.prepare_data
```

Archivos generados principales:

- `data/processed/train.csv`
- `data/processed/val.csv`
- `data/processed/test.csv`
- `data/processed/label_mapping.json`
- `data/processed/preprocessing_report.json`
- `reports/preprocessing_summary.md`

## Construcción del vocabulario

```bash
python -m src.build_vocab --max-vocab-size 20000 --min-freq 2 --max-context-length 128
```

El vocabulario se construye únicamente con `train.csv` para evitar fuga de información desde validación o prueba.

Resultados del tokenizer:

| Elemento | Valor |
| --- | ---: |
| Tokenizer | `SimpleTokenizer` basado en expresiones regulares |
| Tamaño final del vocabulario | 16612 |
| `<PAD>` | 0 |
| `<UNK>` | 1 |
| `<BOS>` | 2 |
| `<EOS>` | 3 |
| MAX_CONTEXT_LENGTH | 128 |

Archivos generados:

- `data/processed/tokenizer.json`
- `data/processed/tokenization_report.json`
- `reports/tokenization_summary.md`

## Inspección del modelo

Inspeccionar un batch tokenizado:

```bash
python -m src.inspect_batch --batch-size 4 --max-length 128
```

Inspeccionar un forward pass del modelo:

```bash
python -m src.inspect_model --batch-size 4 --max-length 128
```

Estos comandos no entrenan el modelo. Sirven para validar que tokenizer, dataset, dataloader y arquitectura funcionan correctamente.

## Entrenamiento

Entrenamiento rápido de prueba:

```bash
python -m src.train --epochs 1 --max-train-batches 20 --max-val-batches 10
```

Entrenamiento final usado:

```bash
python3 -m src.train \
  --epochs 5 \
  --batch-size 32 \
  --learning-rate 3e-4 \
  --use-class-weights \
  --checkpoint-name mini_gpt_sentiment_best_weighted.pt \
  --last-checkpoint-name mini_gpt_sentiment_last_weighted.pt
```

Configuración final:

| Elemento | Valor |
| --- | --- |
| Device usado | `mps` |
| Épocas | 5 |
| Batch size | 32 |
| Learning rate | 3e-4 |
| Class weights | Sí |
| Mejor época | 5 |
| Mejor `val_macro_f1` | 0.897870175080072 |

Archivos generados:

- `models/mini_gpt_sentiment_best_weighted.pt`
- `models/mini_gpt_sentiment_last_weighted.pt`
- `reports/training_history.json`
- `reports/training_summary.md`
- `reports/figures/training_loss_curve.png`
- `reports/figures/training_accuracy_curve.png`
- `reports/figures/training_f1_curve.png`

## Evaluación

Evaluación final sobre `test.csv`:

```bash
python -m src.evaluate --checkpoint models/mini_gpt_sentiment_best_weighted.pt
```

Evaluación rápida:

```bash
python -m src.evaluate --checkpoint models/mini_gpt_sentiment_best_weighted.pt --max-batches 20
```

La evaluación final usa `test.csv`. Este split no se usa para entrenamiento ni para selección del mejor checkpoint.

Archivos generados:

- `reports/evaluation_metrics.json`
- `reports/evaluation_summary.md`
- `reports/test_predictions.csv`
- `reports/error_analysis.csv`
- `reports/figures/confusion_matrix_test.png`
- `reports/figures/final_metrics_barplot.png`

## Predicción de frases nuevas

Ejemplo positivo:

```bash
python -m src.predict --text "The food was delicious and the service was excellent" --show-probs
```

Ejemplo negativo:

```bash
python -m src.predict --text "The food was terrible and the delivery was very late" --show-probs
```

Modo interactivo:

```bash
python -m src.predict --interactive
```

El script:

- Carga el checkpoint entrenado.
- Carga el tokenizer guardado.
- Tokeniza la frase.
- Pasa el texto por el Mini-GPT.
- Devuelve etiqueta, confianza y probabilidades por clase.

## Ejemplos de salida

| Texto | Predicción | Confianza |
| --- | --- | ---: |
| The food was delicious and the service was excellent | positive | 0.9983 |
| The food was terrible and the delivery was very late | negative | 0.9931 |
| I will order again because everything was perfect | positive | 0.9971 |

## Pruebas

Ejecutar la suite de pruebas:

```bash
pytest
```

Resultado confirmado:

```text
26 passed
```

Las pruebas cubren tokenizer, dataset, modelo, entrenamiento, evaluación y predicción.

## Limitaciones

- El modelo fue entrenado principalmente con textos en inglés o normalizados en inglés.
- Las predicciones en español no son completamente confiables sin traducción o entrenamiento adicional.
- Se trabajó con clasificación binaria, excluyendo la clase neutral.
- El tokenizador es didáctico y no usa subpalabras.
- El modelo es pequeño y no fue preentrenado en grandes corpus.
- Puede fallar con ironía, ambigüedad, textos mixtos o ruido en etiquetas.

## Informe académico

El informe final está disponible en:

```text
docs/final_report.md
```

Reportes técnicos:

```text
reports/
```

Figuras:

```text
reports/figures/
```

## Referencias

- Kaggle. BD Food Review Dataset. https://www.kaggle.com/datasets/sanjidh090/bd-food-review-dataset
- Raschka, S. LLMs-from-scratch. https://github.com/rasbt/LLMs-from-scratch
- Vaswani, A., et al. (2017). Attention Is All You Need. https://arxiv.org/abs/1706.03762
- Radford, A., et al. (2018). Improving Language Understanding by Generative Pre-Training. https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf
- PyTorch Documentation. https://pytorch.org/docs/stable/index.html
