"""Arquitectura Mini-GPT para clasificacion de sentimiento."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from src import config


class CausalSelfAttention(nn.Module):
    """Atencion causal multi-cabeza para bloques tipo decoder."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float,
        max_context_length: int,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim debe ser divisible entre num_heads.")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.max_context_length = max_context_length

        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        causal_mask = torch.tril(
            torch.ones(max_context_length, max_context_length, dtype=torch.bool)
        )
        self.register_buffer(
            "causal_mask",
            causal_mask.view(1, 1, max_context_length, max_context_length),
            persistent=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Aplica self-attention causal respetando padding opcional."""
        batch_size, seq_len, _ = x.shape
        if seq_len > self.max_context_length:
            raise ValueError(
                f"seq_len={seq_len} supera max_context_length={self.max_context_length}."
            )

        qkv = self.qkv_proj(x)
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask_value = torch.finfo(scores.dtype).min

        causal_mask = self.causal_mask[:, :, :seq_len, :seq_len]
        scores = scores.masked_fill(~causal_mask, mask_value)

        if attention_mask is not None:
            key_mask = attention_mask[:, None, None, :seq_len].to(dtype=torch.bool)
            scores = scores.masked_fill(~key_mask, mask_value)

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        attn_weights = self.attn_dropout(attn_weights)

        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, seq_len, self.embed_dim)

        output = self.out_proj(context)
        output = self.resid_dropout(output)
        return output


class FeedForward(nn.Module):
    """MLP interna de un bloque Transformer."""

    def __init__(self, embed_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """Bloque decoder tipo GPT con pre-normalizacion."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float,
        max_context_length: int,
    ) -> None:
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        self.attention = CausalSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            max_context_length=max_context_length,
        )
        self.layer_norm_2 = nn.LayerNorm(embed_dim)
        self.feed_forward = FeedForward(embed_dim=embed_dim, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.attention(self.layer_norm_1(x), attention_mask=attention_mask)
        x = x + self.feed_forward(self.layer_norm_2(x))
        return x


class MiniGPTForSentiment(nn.Module):
    """Mini-GPT reducido adaptado a clasificacion de sentimiento."""

    def __init__(
        self,
        vocab_size: int,
        max_context_length: int,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        num_classes: int,
        dropout: float,
        pad_token_id: int = 0,
        pooling: str = "mean",
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim debe ser divisible entre num_heads.")
        if pooling not in ("mean", "last"):
            raise ValueError("pooling debe ser 'mean' o 'last'.")

        self.vocab_size = vocab_size
        self.max_context_length = max_context_length
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout = dropout
        self.pad_token_id = pad_token_id
        self.pooling = pooling

        self.token_embedding = nn.Embedding(
            vocab_size,
            embed_dim,
            padding_idx=pad_token_id,
        )
        self.position_embedding = nn.Embedding(max_context_length, embed_dim)
        self.embedding_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    max_context_length=max_context_length,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_layer_norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        """Ejecuta un forward pass y calcula loss opcional."""
        if input_ids.ndim != 2:
            raise ValueError("input_ids debe tener forma [batch_size, seq_len].")

        batch_size, seq_len = input_ids.shape
        if seq_len > self.max_context_length:
            raise ValueError(
                f"seq_len={seq_len} supera max_context_length={self.max_context_length}."
            )

        if attention_mask is None:
            attention_mask = (input_ids != self.pad_token_id).long()
        else:
            attention_mask = attention_mask.to(device=input_ids.device)

        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        token_embeddings = self.token_embedding(input_ids)
        position_embeddings = self.position_embedding(position_ids)
        x = self.embedding_dropout(token_embeddings + position_embeddings)

        for block in self.blocks:
            x = block(x, attention_mask=attention_mask)

        x = self.final_layer_norm(x)

        if self.pooling == "mean":
            mask_expanded = attention_mask.unsqueeze(-1).float()
            pooled = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1e-9)
        else:
            valid_lengths = attention_mask.long().sum(dim=1)
            last_indices = (valid_lengths - 1).clamp(min=0, max=seq_len - 1)
            batch_indices = torch.arange(batch_size, device=input_ids.device)
            pooled = x[batch_indices, last_indices]

        logits = self.classifier(pooled)
        loss = F.cross_entropy(logits, labels) if labels is not None else None
        return {"logits": logits, "loss": loss}

    def count_parameters(self) -> dict[str, int]:
        """Retorna parametros totales y entrenables."""
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(
            parameter.numel() for parameter in self.parameters() if parameter.requires_grad
        )
        return {"total": int(total), "trainable": int(trainable)}

    def get_model_config(self) -> dict[str, Any]:
        """Retorna la configuracion principal del modelo."""
        return {
            "vocab_size": self.vocab_size,
            "max_context_length": self.max_context_length,
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "num_classes": self.num_classes,
            "dropout": self.dropout,
            "pad_token_id": self.pad_token_id,
            "pooling": self.pooling,
        }


def create_model_from_config(
    vocab_size: int | None = None,
    num_classes: int | None = None,
) -> MiniGPTForSentiment:
    """Crea el modelo usando config.py y artefactos procesados si existen."""
    if vocab_size is None:
        vocab_size = infer_vocab_size(config.TOKENIZER_PATH)

    if num_classes is None:
        num_classes = infer_num_classes(config.PROCESSED_DATA_DIR / "label_mapping.json")

    return MiniGPTForSentiment(
        vocab_size=vocab_size,
        max_context_length=config.MAX_CONTEXT_LENGTH,
        embed_dim=config.EMBED_DIM,
        num_heads=config.NUM_HEADS,
        num_layers=config.NUM_LAYERS,
        num_classes=num_classes,
        dropout=config.DROPOUT,
        pad_token_id=0,
        pooling="mean",
    )


def infer_vocab_size(tokenizer_path: str | Path) -> int:
    """Obtiene vocab_size desde tokenizer.json o usa MAX_VOCAB_SIZE."""
    path = Path(tokenizer_path)
    if not path.exists():
        return config.MAX_VOCAB_SIZE

    payload = json.loads(path.read_text(encoding="utf-8"))
    return int(len(payload["token_to_id"]))


def infer_num_classes(label_mapping_path: str | Path) -> int:
    """Obtiene numero de clases desde label_mapping.json o usa config.NUM_CLASSES."""
    path = Path(label_mapping_path)
    if not path.exists():
        return config.NUM_CLASSES

    payload = json.loads(path.read_text(encoding="utf-8"))
    return int(len(payload))
