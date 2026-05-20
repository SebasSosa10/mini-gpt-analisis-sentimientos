# Resumen de preparacion del dataset

## Archivo utilizado

- Archivo: `data/raw/BDFoodSent-334k.csv`
- Filas originales: 334119

## Columnas utilizadas

- Texto: `text`
- Etiqueta: `labels`
- Rating: `ratings_overall`
- Fuente final de sentimiento: `labels`

## Tarea configurada

- Tipo de tarea: `binary`
- Semilla aleatoria: 42
- Test size final: 0.15
- Validation size final: 0.15

## Limpieza aplicada

- Conversion a minusculas.
- Eliminacion de saltos de linea y espacios repetidos.
- Conservacion de letras, numeros, puntuacion basica y espacios.
- No se aplico limpieza agresiva ni stemming.

## Mapeo de etiquetas

| Sentimiento | label_id |
| --- | --- |
| negative | 0 |
| positive | 1 |

- Etiquetas validas construidas: 334119
- Etiquetas invalidas o no interpretables: 0
- Desacuerdos entre labels y ratings_overall: 0 de 334119 casos comparables.

## Filtros aplicados

| Filtro | Filas removidas |
| --- | --- |
| Textos vacios | 0 |
| Textos cortos | 60980 |
| Textos largos | 0 |
| Sin sentimiento | 0 |
| Excluidos por tipo de tarea | 34172 |
| Duplicados por texto limpio y sentimiento | 20052 |

## Distribucion final de clases

| Clase | Cantidad |
| --- | --- |
| negative | 136676 |
| positive | 82239 |

## Division del dataset

| Particion | Filas |
| --- | --- |
| train | 153240 |
| val | 32837 |
| test | 32838 |

## Archivos generados

- `data/processed/train.csv`
- `data/processed/val.csv`
- `data/processed/test.csv`
- `data/processed/label_mapping.json`
- `data/processed/preprocessing_report.json`
- `reports/preprocessing_summary.md`

## Observaciones para el informe academico

- La configuracion principal del proyecto usa clasificacion binaria, excluyendo la clase neutral.
- La seleccion de columnas se basa en el EDA previo y queda registrada para trazabilidad.
- La columna de texto limpio queda preparada para la tokenizacion de la siguiente fase.
- Si se usa modo multiclass, se debe reportar explicitamente la clase neutral en metodologia y resultados.
