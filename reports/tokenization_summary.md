# Resumen de tokenizacion y vocabulario

## Archivo de entrenamiento usado

- Archivo: `data\processed\train.csv`
- El vocabulario se construyo solo con `train.csv` para evitar fuga de informacion desde validacion o prueba.

## Tipo de tokenizacion

- Tokenizador didactico basado en expresiones regulares, con separacion de palabras, numeros y puntuacion basica.
- No usa tokenizadores externos como HuggingFace.

## Tokens especiales

| Token | ID |
| --- | --- |
| <PAD> | 0 |
| <UNK> | 1 |
| <BOS> | 2 |
| <EOS> | 3 |

## Tamano del vocabulario

- Tamano final: 16612
- Tamano maximo configurado: 20000

## Frecuencia minima

- Frecuencia minima incluida: 2

## Longitud maxima de contexto

- MAX_CONTEXT_LENGTH: 128
- Las secuencias mas largas se truncaran en el Dataset; las mas cortas recibiran padding.

## Estadisticas de longitud tokenizada

| Metrica | Valor |
| --- | --- |
| num_texts | 153240 |
| mean | 16.0216 |
| median | 13.0 |
| min | 5 |
| max | 65 |
| p25 | 8.0 |
| p75 | 21.0 |
| p90 | 32.0 |
| p95 | 37.0 |
| max_context_length | 128 |
| texts_over_max_context | 0 |
| percent_over_max_context | 0.0 |

## Ejemplos de tokenizacion

| Texto | Tokens | IDs | Decodificado |
| --- | --- | --- | --- |
| the meal was disappointing | the meal was disappointing | [2, 4, 125, 5, 158, 3] | the meal was disappointing |
| very very disgusting i dont even want to rate | very very disgusting i dont even want to rate | [2, 15, 15, 160, 7, 535, 112, 329, 17, 847, 3] | very very disgusting i dont even want to rate |

## Observaciones para el informe academico

- El token `<PAD>` permite formar batches de longitud fija y se marca con 0 en `attention_mask`.
- Los tokens `<BOS>` y `<EOS>` delimitan el inicio y fin de cada secuencia.
- Los tokens fuera del vocabulario se reemplazan por `<UNK>`.
- La siguiente fase puede usar `input_ids` para embeddings de tokens y `attention_mask` para ignorar padding.
