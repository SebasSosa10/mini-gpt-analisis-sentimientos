"""Pruebas basicas de estructura del proyecto."""

from pathlib import Path


def test_config_importable() -> None:
    """Valida que la configuracion principal se pueda importar."""
    import src.config as config

    assert config.PROJECT_ROOT.exists()


def test_expected_directories_exist() -> None:
    """Valida que existan las carpetas principales del proyecto."""
    project_root = Path(__file__).resolve().parents[1]

    expected_directories = [
        project_root / "data",
        project_root / "src",
        project_root / "models",
        project_root / "reports",
    ]

    for directory in expected_directories:
        assert directory.exists()
        assert directory.is_dir()


def test_get_device_returns_valid_device() -> None:
    """Valida que get_device retorne un dispositivo reconocido por PyTorch."""
    from src.utils import get_device

    device = get_device()
    assert device.type in {"cpu", "cuda", "mps"}


def test_clean_text_removes_repeated_spaces() -> None:
    """Valida limpieza basica de espacios repetidos."""
    from src.prepare_data import clean_text

    assert clean_text("  Food   was\n delicious  ") == "food was delicious"


def test_normalize_label_from_rating() -> None:
    """Valida el mapeo de ratings a sentimiento."""
    from src.prepare_data import normalize_label_from_rating

    assert normalize_label_from_rating(1) == "negative"
    assert normalize_label_from_rating(2) == "negative"
    assert normalize_label_from_rating(3) == "neutral"
    assert normalize_label_from_rating(4) == "positive"
    assert normalize_label_from_rating(5) == "positive"


def test_normalize_label_from_label() -> None:
    """Valida la normalizacion de etiquetas textuales."""
    from src.prepare_data import normalize_label_from_label

    assert normalize_label_from_label("positive") == "positive"
    assert normalize_label_from_label("negative") == "negative"
    assert normalize_label_from_label("neutral") == "neutral"
