# Resumen de entrenamiento del Mini-GPT

## Configuracion del entrenamiento

| Parametro | Valor |
| --- | --- |
| epochs | 1 |
| batch_size | 32 |
| learning_rate | 0.0003 |
| weight_decay | 0.01 |
| max_length | 128 |
| grad_clip | 1.0 |
| use_class_weights | True |
| patience | 3 |
| max_train_batches | 5 |
| max_val_batches | 5 |

## Configuracion del modelo

| Parametro | Valor |
| --- | --- |
| vocab_size | 16612 |
| max_context_length | 128 |
| embed_dim | 192 |
| num_heads | 6 |
| num_layers | 3 |
| num_classes | 2 |
| dropout | 0.1 |
| pad_token_id | 0 |
| pooling | mean |

## Dataset utilizado

- Entrenamiento: `data/processed/train.csv`.
- Validacion: `data/processed/val.csv`.
- `test.csv` se reserva para la evaluacion final posterior.
- El vocabulario fue construido solo con `train.csv`.

## Metrica principal de seleccion

- Metrica: `val_macro_f1`.
- Mejor valor: 0.638942617666022.
- Mejor epoca: 1.

## Resultados por epoca

| Epoca | Train loss | Train acc | Val loss | Val acc | Val macro F1 |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.687037 | 0.525 | 0.670211 | 0.65 | 0.638943 |

## Mejor checkpoint

- Mejor checkpoint: `models\mini_gpt_v2_test.pt`.
- Ultimo checkpoint: `models\mini_gpt_v2_test_last.pt`.

## Observaciones para el informe academico

- El entrenamiento usa `train.csv` y selecciona checkpoint con `val.csv`.
- La evaluacion final sobre `test.csv` se deja para la siguiente fase.
- Esta corrida uso `max-train-batches` o `max-val-batches`; por tanto es una corrida reducida de prueba y no el entrenamiento final.

## Limitaciones del entrenamiento

- El modelo es deliberadamente pequeno para fines didacticos.
- Las metricas de validacion no sustituyen la evaluacion final en test.
- Si se ejecuta en CPU, el entrenamiento completo puede tardar mas.
