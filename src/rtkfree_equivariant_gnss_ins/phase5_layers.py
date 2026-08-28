"""Typed scalar/vector neural building blocks for strict Phase 5 SO(2)."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class So2VectorLinear(nn.Module):
    """Bias-free linear mixing of horizontal vectors that commutes with SO(2)."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError("SO(2) vector channel counts must be positive")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight_identity = nn.Parameter(torch.empty(out_channels, in_channels))
        self.weight_quarter_turn = nn.Parameter(torch.empty(out_channels, in_channels))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1.0 / math.sqrt(2.0 * self.in_channels)
        nn.init.uniform_(self.weight_identity, -bound, bound)
        nn.init.uniform_(self.weight_quarter_turn, -bound, bound)

    def forward(self, vectors: Tensor) -> Tensor:
        if vectors.ndim < 2 or vectors.shape[-2:] != (self.in_channels, 2):
            raise ValueError("SO(2) vectors must end in the configured channels by two")
        quarter_turned = torch.stack((-vectors[..., 1], vectors[..., 0]), dim=-1)
        direct = torch.einsum("oi,...id->...od", self.weight_identity, vectors)
        quadrature = torch.einsum(
            "oi,...id->...od",
            self.weight_quarter_turn,
            quarter_turned,
        )
        return direct + quadrature

    def extra_repr(self) -> str:
        return f"in_channels={self.in_channels}, out_channels={self.out_channels}, bias=False"


def radial_tanh(vectors: Tensor) -> Tensor:
    """Bound each horizontal vector norm without changing its direction."""

    if vectors.ndim < 2 or vectors.shape[-1] != 2:
        raise ValueError("radial nonlinearity expects horizontal vector channels")
    if not torch.is_floating_point(vectors):
        raise TypeError("radial nonlinearity expects floating-point vectors")
    norm = torch.linalg.vector_norm(vectors, dim=-1, keepdim=True)
    denominator = norm.clamp_min(torch.finfo(vectors.dtype).eps)
    return vectors * (torch.tanh(norm) / denominator)


def apply_scalar_vector_gate(vectors: Tensor, invariant_gate_logits: Tensor) -> Tensor:
    """Scale each vector channel with one rotation-invariant sigmoid gate."""

    if vectors.ndim < 2 or vectors.shape[-1] != 2:
        raise ValueError("gated values must be horizontal vector channels")
    if invariant_gate_logits.shape != vectors.shape[:-1]:
        raise ValueError("one invariant gate is required per vector channel")
    return vectors * torch.sigmoid(invariant_gate_logits).unsqueeze(-1)
