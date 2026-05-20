# Analisis exploratorio del dataset

## Archivo analizado

- Archivo: `data/raw/BDFoodSent-334k.csv`
- Nota: la seleccion final de columnas de texto y etiqueta se confirmara en la fase de preparacion de datos.

## Dimensiones del dataset

- Filas: 334119
- Columnas: 29
- Memoria aproximada: 561.6596 MB

## Columnas disponibles

- `uuid`
- `createdAt`
- `updatedAt`
- `text`
- `isAnonymous`
- `reviewerName`
- `reviewerId`
- `ratings_overall`
- `labels`
- `ratings`
- `type`
- `likeCount`
- `isLiked`
- `code`
- `name`
- `address`
- `city`
- `post_code`
- `latitude`
- `longitude`
- `primary_cuisine`
- `primary_cuisine_id`
- `cuisines`
- `review_number`
- `restaurant_overall_rating`
- `hero_image`
- `hero_listing_image`
- `vertical_type_ids`
- `web_path`

## Tipos de datos

| Columna | Tipo |
| --- | --- |
| uuid | object |
| createdAt | object |
| updatedAt | object |
| text | object |
| isAnonymous | bool |
| reviewerName | object |
| reviewerId | object |
| ratings_overall | int64 |
| labels | int64 |
| ratings | object |
| type | object |
| likeCount | int64 |
| isLiked | bool |
| code | object |
| name | object |
| address | object |
| city | object |
| post_code | object |
| latitude | float64 |
| longitude | float64 |
| primary_cuisine | object |
| primary_cuisine_id | int64 |
| cuisines | object |
| review_number | int64 |
| restaurant_overall_rating | float64 |
| hero_image | object |
| hero_listing_image | object |
| vertical_type_ids | object |
| web_path | object |

## Valores nulos

No se detectaron valores nulos en el dataset cargado.

## Filas duplicadas

- Filas duplicadas detectadas: 0

## Posibles columnas de texto

| Columna | No nulos | Longitud media | Min | Max | Muestras |
| --- | --- | --- | --- | --- | --- |
| text | 334119 | 57.5104 | 1 | 365 | too much small amount \| very small in amount \| food average but their desert booter halwa and pudding is very good |

- Columna de texto tentativa: `text`

### Estadisticas de longitud de la columna tentativa

- Columna: `text`
- Textos validos: 334119

Caracteres:
| Metrica | Valor |
| --- | --- |
| mean | 57.5104 |
| median | 41.0 |
| min | 1 |
| max | 365 |
| p25 | 18.0 |
| p75 | 81.0 |
| p90 | 140.0 |
| p95 | 178.0 |

Palabras aproximadas:
| Metrica | Valor |
| --- | --- |
| mean | 10.9826 |
| median | 8.0 |
| min | 1 |
| max | 72 |
| p25 | 3.0 |
| p75 | 15.0 |
| p90 | 27.0 |
| p95 | 33.0 |

## Posibles columnas de etiqueta o rating

| Columna | Tipo | Valores unicos | Valores frecuentes | Nota |
| --- | --- | --- | --- | --- |
| labels | int64 | 3 | 1: 161275 \| 3: 133529 \| 2: 39315 | Sin hipotesis automatica |
| ratings_overall | int64 | 5 | 1: 127409 \| 5: 94199 \| 4: 39330 \| 3: 39315 \| 2: 33866 | Hipotesis pendiente de confirmar: rating 1-2 negativo, 3 neutro, 4-5 positivo. |
| restaurant_overall_rating | float64 | 42 | 4.0: 42825 \| 4.1: 38019 \| 3.9: 36508 \| 4.2: 36216 \| 3.8: 26696 | Hipotesis pendiente de confirmar: rating 1-2 negativo, 3 neutro, 4-5 positivo. |
| ratings | object | 30 | [{'topic': 'overall', 'score': 1}, {'topic': 'restaurant_food', 'score': 1}]: 53317 \| [{'topic': 'overall', 'score': 5}, {'topic': 'restaurant_food', 'score': 5}, {'topic': 'rider', 'score': 5}]: 49322 \| [{'topic': 'overall', 'score': 5}, {'topic': 'restaurant_food', 'score': 5}]: 38748 \| [{'topic': 'overall', 'score': 1}, {'topic': 'restaurant_food', 'score': 1}, {'topic': 'rider', 'score': 5}]: 29990 \| [{'topic': 'overall', 'score': 1}, {'topic': 'restaurant_food', 'score': 1}, {'topic': 'rider', 'score': 1}]: 21523 | Sin hipotesis automatica |

- Columna de etiqueta/rating tentativa: `labels`

## Ejemplos del dataset

| uuid | createdAt | updatedAt | text | isAnonymous | reviewerName |
| --- | --- | --- | --- | --- | --- |
| 68e8f769-a0f0-460c-bfb9-16662fcc1e17 | 2023-11-02T06:48:08Z | 2023-11-02T06:48:08Z | too much small amount | False | MOON |
| 706a1e43-7475-4d7f-8488-cb8ed2ea8991 | 2023-10-27T07:35:53Z | 2023-10-27T07:35:53Z | very small in amount | False | MOON |
| 16cd99a3-7295-432e-b193-1551df62d255 | 2023-10-25T18:09:17Z | 2023-10-25T18:09:17Z | food average but their desert booter halwa and pudding is very good | False | SALMAN |
| 61a9fdf3-ad6b-4d23-9436-1f2f596bbcff | 2023-10-25T07:48:39Z | 2023-10-25T07:48:39Z | fresh and tasty | False | GOLAM |
| a7835cdf-2c1f-4cf0-b9d5-2777d904d746 | 2023-10-15T13:37:20Z | 2023-10-15T13:37:20Z | everything i ordered was good | False | FERZANA |
| 799d5071-bd5d-417d-ba92-56aaf4d528c8 | 2023-10-14T09:58:14Z | 2023-10-14T09:58:14Z | kalo jira vhorta balu vhorta not recommended baki shob valo chilo | False | FERZANA |
| 547974ca-f846-47e3-a5a8-3834c40764be | 2023-10-11T14:04:13Z | 2023-10-11T14:04:13Z | quantity of dal should be little more otherwise everything was good specially beef | False | FERZANA |
| f4d7d419-e569-4b7e-b097-adfdefeb62ad | 2023-10-01T06:32:07Z | 2023-10-01T06:32:07Z | beef was too poor in size quantity aganist price but test was good | False | DEDAR |
| 79e75563-ac7f-4084-99be-913e929902cb | 2023-09-29T08:34:28Z | 2023-09-29T08:34:28Z | chicken was not up to the mark and lotpoti was full of potato alu bhorta didn t seem fresh | False | IFTEKHER |
| 12854429-fed4-480f-b996-b4bc64a259af | 2023-09-25T08:58:18Z | 2023-09-25T08:58:18Z | lot of sand in kali jeers bortha | False | MOHAMMAD |

## Problemas potenciales encontrados

- No se detectaron valores nulos en esta carga del CSV.
- No se detectaron filas duplicadas exactas.
- La columna de texto detectada es tentativa; debe confirmarse con los ejemplos reales.
- Las clases disponibles deben confirmarse antes de la preparacion de datos.
- Los problemas de ruido, ortografia, idioma y sesgos deben evaluarse con una inspeccion cualitativa de ejemplos.

## Recomendaciones para la siguiente fase

- Confirmar manualmente la columna de texto y la columna objetivo antes de limpiar o transformar datos.
- Definir si la tarea sera binaria o multiclase segun las etiquetas reales disponibles.
- Si solo existe rating numerico, validar academicamente el mapeo de rating a sentimiento antes de usarlo.
- Revisar textos vacios, duplicados, ruido, errores ortograficos y posibles sesgos antes del entrenamiento.
- Guardar una version procesada en data/processed/ durante la Fase 3.
