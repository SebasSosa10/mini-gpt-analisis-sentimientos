"""Tokenizer didactico basado en expresiones regulares."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence


class SimpleTokenizer:
    """Tokenizer simple para el Mini-GPT de sentimiento.

    El vocabulario se construye con textos de entrenamiento y usa tokens
    especiales para padding, desconocidos, inicio y fin de secuencia.
    """

    pad_token = "<PAD>"
    unk_token = "<UNK>"
    bos_token = "<BOS>"
    eos_token = "<EOS>"

    pad_token_id = 0
    unk_token_id = 1
    bos_token_id = 2
    eos_token_id = 3

    _token_pattern = re.compile(r"\w+|[.,!?;:'\"()\-/%%]", flags=re.UNICODE)

    def __init__(
        self,
        max_vocab_size: int = 20000,
        min_freq: int = 2,
        lowercase: bool = True,
    ) -> None:
        if max_vocab_size < 4:
            raise ValueError("max_vocab_size debe ser al menos 4.")
        if min_freq < 1:
            raise ValueError("min_freq debe ser al menos 1.")

        self.max_vocab_size = max_vocab_size
        self.min_freq = min_freq
        self.lowercase = lowercase
        self.token_to_id: dict[str, int] = {}
        self.id_to_token: dict[int, str] = {}
        self.fitted = False
        self.top_tokens: list[dict[str, int | str]] = []
        self._initialize_special_tokens()

    @property
    def vocab_size(self) -> int:
        """Retorna el tamano actual del vocabulario."""
        return len(self.token_to_id)

    def basic_tokenize(self, text: Any) -> list[str]:
        """Tokeniza palabras, numeros y puntuacion basica."""
        if text is None:
            return []

        try:
            if text != text:  # Detecta NaN sin depender de pandas.
                return []
        except TypeError:
            pass

        normalized_text = str(text).strip()
        if not normalized_text:
            return []

        if self.lowercase:
            normalized_text = normalized_text.lower()

        return self._token_pattern.findall(normalized_text)

    def fit(self, texts: Iterable[Any]) -> "SimpleTokenizer":
        """Construye el vocabulario usando solo textos de entrenamiento."""
        counter: Counter[str] = Counter()
        for text in texts:
            counter.update(self.basic_tokenize(text))

        self._initialize_special_tokens()
        max_regular_tokens = self.max_vocab_size - len(self.token_to_id)
        sorted_tokens = sorted(counter.items(), key=lambda item: (-item[1], item[0]))

        for token, frequency in sorted_tokens:
            if frequency < self.min_freq:
                continue
            if len(self.token_to_id) >= self.max_vocab_size:
                break
            self.token_to_id[token] = len(self.token_to_id)

        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}
        self.top_tokens = [
            {"token": token, "frequency": int(frequency)}
            for token, frequency in sorted_tokens[: min(100, len(sorted_tokens))]
        ]
        self.fitted = True

        # max_regular_tokens queda calculado para dejar claro que el limite
        # incluye tokens especiales, aunque el corte real se hace arriba.
        _ = max_regular_tokens
        return self

    def encode(
        self,
        text: Any,
        max_length: int | None = None,
        add_special_tokens: bool = True,
        padding: bool = False,
        truncation: bool = False,
    ) -> list[int]:
        """Convierte texto a IDs de tokens."""
        self._require_fitted()

        token_ids = [
            self.token_to_id.get(token, self.unk_token_id)
            for token in self.basic_tokenize(text)
        ]

        if add_special_tokens:
            token_ids = [self.bos_token_id] + token_ids + [self.eos_token_id]

        if max_length is not None and len(token_ids) > max_length:
            if truncation:
                token_ids = token_ids[:max_length]
                if add_special_tokens and token_ids[-1] != self.eos_token_id:
                    token_ids[-1] = self.eos_token_id
            else:
                raise ValueError(
                    "La secuencia supera max_length y truncation=False."
                )

        if padding:
            if max_length is None:
                raise ValueError("padding=True requiere max_length.")
            pad_length = max_length - len(token_ids)
            if pad_length > 0:
                token_ids = token_ids + [self.pad_token_id] * pad_length

        return token_ids

    def encode_plus(
        self,
        text: Any,
        max_length: int,
        add_special_tokens: bool = True,
        padding: bool = True,
        truncation: bool = True,
    ) -> dict[str, list[int]]:
        """Codifica texto y retorna input_ids junto con attention_mask."""
        input_ids = self.encode(
            text,
            max_length=max_length,
            add_special_tokens=add_special_tokens,
            padding=False,
            truncation=truncation,
        )

        attention_mask = [1] * len(input_ids)
        if padding:
            pad_length = max_length - len(input_ids)
            if pad_length < 0:
                raise ValueError(
                    "La secuencia supera max_length y no se pudo truncar."
                )
            input_ids = input_ids + [self.pad_token_id] * pad_length
            attention_mask = attention_mask + [0] * pad_length

        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def decode(
        self,
        token_ids: Sequence[int],
        skip_special_tokens: bool = True,
    ) -> str:
        """Reconstruye texto legible desde IDs de tokens."""
        self._require_fitted()
        special_tokens = {
            self.pad_token,
            self.bos_token,
            self.eos_token,
        }

        tokens: list[str] = []
        for token_id in token_ids:
            token = self.id_to_token.get(int(token_id), self.unk_token)
            if skip_special_tokens and token in special_tokens:
                continue
            tokens.append(token)

        return self._join_tokens(tokens)

    def save(self, path: str | Path) -> None:
        """Guarda el tokenizer en formato JSON."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "token_to_id": self.token_to_id,
            "id_to_token": {str(idx): token for idx, token in self.id_to_token.items()},
            "max_vocab_size": self.max_vocab_size,
            "min_freq": self.min_freq,
            "lowercase": self.lowercase,
            "special_tokens": {
                "pad_token": self.pad_token,
                "unk_token": self.unk_token,
                "bos_token": self.bos_token,
                "eos_token": self.eos_token,
                "pad_token_id": self.pad_token_id,
                "unk_token_id": self.unk_token_id,
                "bos_token_id": self.bos_token_id,
                "eos_token_id": self.eos_token_id,
            },
            "fitted": self.fitted,
            "top_tokens": self.top_tokens,
        }
        output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "SimpleTokenizer":
        """Carga un tokenizer guardado previamente."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        tokenizer = cls(
            max_vocab_size=int(payload["max_vocab_size"]),
            min_freq=int(payload["min_freq"]),
            lowercase=bool(payload["lowercase"]),
        )
        tokenizer.token_to_id = {
            str(token): int(token_id)
            for token, token_id in payload["token_to_id"].items()
        }
        tokenizer.id_to_token = {
            int(token_id): str(token)
            for token_id, token in payload["id_to_token"].items()
        }
        tokenizer.fitted = bool(payload.get("fitted", True))
        tokenizer.top_tokens = list(payload.get("top_tokens", []))
        return tokenizer

    def get_vocab(self) -> dict[str, int]:
        """Retorna una copia del vocabulario."""
        return dict(self.token_to_id)

    def get_token_frequencies_summary(self, top_n: int = 30) -> list[dict[str, int | str]]:
        """Retorna los tokens mas frecuentes registrados durante fit."""
        return self.top_tokens[:top_n]

    def _initialize_special_tokens(self) -> None:
        self.token_to_id = {
            self.pad_token: self.pad_token_id,
            self.unk_token: self.unk_token_id,
            self.bos_token: self.bos_token_id,
            self.eos_token: self.eos_token_id,
        }
        self.id_to_token = {
            self.pad_token_id: self.pad_token,
            self.unk_token_id: self.unk_token,
            self.bos_token_id: self.bos_token,
            self.eos_token_id: self.eos_token,
        }

    def _require_fitted(self) -> None:
        if not self.fitted:
            raise RuntimeError("El tokenizer debe ajustarse con fit() antes de usarse.")

    @staticmethod
    def _join_tokens(tokens: Sequence[str]) -> str:
        """Une tokens evitando espacios antes de puntuacion basica."""
        text = " ".join(tokens)
        text = re.sub(r"\s+([.,!?;:%/)])", r"\1", text)
        text = re.sub(r"([(])\s+", r"\1", text)
        text = text.replace(" - ", "-")
        return text.strip()
