"""Configuracion inicial del proyecto Mini-GPT Sentiment."""

from pathlib import Path


# Raiz del proyecto. Se calcula desde este archivo para evitar rutas absolutas.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Directorios principales de datos y artefactos.
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TOKENIZER_PATH = PROCESSED_DATA_DIR / "tokenizer.json"
TOKENIZATION_REPORT_PATH = PROCESSED_DATA_DIR / "tokenization_report.json"
TOKENIZATION_SUMMARY_PATH = REPORTS_DIR / "tokenization_summary.md"
MODEL_CHECKPOINT_PATH = MODELS_DIR / "mini_gpt_sentiment.pt"
MODEL_CONFIG_PATH = MODELS_DIR / "model_config.json"
BEST_MODEL_PATH = MODELS_DIR / "mini_gpt_sentiment_best.pt"
LAST_MODEL_PATH = MODELS_DIR / "mini_gpt_sentiment_last.pt"
TRAINING_HISTORY_PATH = REPORTS_DIR / "training_history.json"
TRAINING_SUMMARY_PATH = REPORTS_DIR / "training_summary.md"
TRAINING_LOSS_FIGURE_PATH = FIGURES_DIR / "training_loss_curve.png"
TRAINING_ACC_FIGURE_PATH = FIGURES_DIR / "training_accuracy_curve.png"
TRAINING_F1_FIGURE_PATH = FIGURES_DIR / "training_f1_curve.png"
EVALUATION_METRICS_PATH = REPORTS_DIR / "evaluation_metrics.json"
EVALUATION_SUMMARY_PATH = REPORTS_DIR / "evaluation_summary.md"
TEST_PREDICTIONS_PATH = REPORTS_DIR / "test_predictions.csv"
ERROR_ANALYSIS_PATH = REPORTS_DIR / "error_analysis.csv"
CONFUSION_MATRIX_FIGURE_PATH = FIGURES_DIR / "confusion_matrix_test.png"
FINAL_METRICS_FIGURE_PATH = FIGURES_DIR / "final_metrics_barplot.png"

# Semilla para reproducibilidad en particiones, entrenamiento y evaluacion.
RANDOM_SEED = 42

# Proporciones iniciales para dividir el dataset.
TEST_SIZE = 0.15
VAL_SIZE = 0.15

# Hiperparametros didacticos para tokenizacion y contexto.
MAX_VOCAB_SIZE = 20000
MAX_CONTEXT_LENGTH = 128
MIN_TOKEN_FREQ = 2

# Hiperparametros iniciales de entrenamiento.
BATCH_SIZE = 32
DEFAULT_EPOCHS = 5
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
PATIENCE = 3
WARMUP_RATIO = 0.1
LABEL_SMOOTHING = 0.1

# Configuracion del Mini-GPT.
EMBED_DIM = 192
NUM_HEADS = 6
NUM_LAYERS = 3
DROPOUT = 0.1

# Numero inicial de clases. Se ajustara si el dataset confirma una clase neutra.
NUM_CLASSES = 2
