"""Funciones auxiliares compartidas del proyecto."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np


def set_seed(seed: int) -> None:
    """Configura semillas para mejorar la reproducibilidad."""
    random.seed(seed)
    np.random.seed(seed)

    torch = import_torch_if_available()
    if torch is not None:
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)


def ensure_dir(path: str | Path) -> Path:
    """Crea un directorio si no existe y devuelve su ruta como Path."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_json(data: Any, path: str | Path) -> Path:
    """Guarda datos en JSON con indentacion legible."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(make_json_safe(data), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path


def load_json(path: str | Path) -> Any:
    """Carga un archivo JSON desde disco."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def get_device() -> Any:
    """Devuelve el dispositivo disponible para PyTorch."""
    torch = import_torch_if_available()
    if torch is None:
        return SimpleNamespace(type="cpu")

    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def import_torch_if_available() -> Any:
    """Importa PyTorch solo cuando se necesita."""
    try:
        import torch
    except ImportError:
        return None

    return torch


def make_json_safe(value: Any) -> Any:
    """Convierte objetos comunes de numpy y pathlib a valores serializables."""
    if isinstance(value, dict):
        return {str(key): make_json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [make_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [make_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value
