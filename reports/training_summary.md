# Resumen de entrenamiento del Mini-GPT

## Configuracion del entrenamiento

| Parametro | Valor |
| --- | --- |
| epochs | 5 |
| batch_size | 32 |
| learning_rate | 0.0003 |
| weight_decay | 0.01 |
| max_length | 128 |
| grad_clip | 1.0 |
| use_class_weights | True |
| patience | 3 |
| max_train_batches | None |
| max_val_batches | None |

## Configuracion del modelo

| Parametro | Valor |
| --- | --- |
| vocab_size | 16612 |
| max_context_length | 128 |
| embed_dim | 128 |
| num_heads | 4 |
| num_layers | 2 |
| num_classes | 2 |
| dropout | 0.1 |
| pad_token_id | 0 |

## Dataset utilizado

- Entrenamiento: `data/processed/train.csv`.
- Validacion: `data/processed/val.csv`.
- `test.csv` se reserva para la evaluacion final posterior.
- El vocabulario fue construido solo con `train.csv`.

## Metrica principal de seleccion

- Metrica: `val_macro_f1`.
- Mejor valor: 0.897870175080072.
- Mejor epoca: 5.

## Resultados por epoca

| Epoca | Train loss | Train acc | Val loss | Val acc | Val macro F1 |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.367039 | 0.844819 | 0.309396 | 0.875019 | 0.86911 |
| 2 | 0.292118 | 0.884978 | 0.270826 | 0.895118 | 0.887999 |
| 3 | 0.269013 | 0.894499 | 0.2635 | 0.897037 | 0.890826 |
| 4 | 0.254365 | 0.901938 | 0.263375 | 0.902671 | 0.89635 |
| 5 | 0.242688 | 0.90648 | 0.25527 | 0.904315 | 0.89787 |

## Mejor checkpoint

- Mejor checkpoint: `models/mini_gpt_sentiment_best_weighted.pt`.
- Ultimo checkpoint: `models/mini_gpt_sentiment_last_weighted.pt`.

## Observaciones para el informe academico

- El entrenamiento usa `train.csv` y selecciona checkpoint con `val.csv`.
- La evaluacion final sobre `test.csv` se deja para la siguiente fase.

## Limitaciones del entrenamiento

- El modelo es deliberadamente pequeno para fines didacticos.
- Las metricas de validacion no sustituyen la evaluacion final en test.
- Si se ejecuta en CPU, el entrenamiento completo puede tardar mas.
