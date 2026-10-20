"""Stage 2 - Initial embeddings and the tabular Transformer (FT-Transformer style).

How a flow becomes tokens
-------------------------
A preprocessed flow has F numeric features x'_j (already standardised) and C
categorical fields (here: protocol). Each feature becomes ONE d-dimensional token:

    numeric j :  z_j = x'_j * W_j + b_j        W_j, b_j in R^d  (learned, per feature)
    categorical:  z_c = Embedding_c[value]      (learned lookup table, per column)

A learned [CLS] token is prepended. Result: tensor (N, 1 + F + C, d).
Self-attention then lets every feature token look at every other token; the
final [CLS] vector is the flow representation fed to the classifier head.

This is the feature-tokenizer idea of FT-Transformer (Gorishniy et al. 2021),
not the original TabTransformer, which feeds continuous features straight to
the MLP. Verify against the paper before using either name in the report.

All "initial embeddings" are random at start: W, b and [CLS] ~ uniform(-1/sqrt(d),
1/sqrt(d)); the protocol table uses nn.Embedding's default N(0, 1). They are learned
jointly with the network during initial training. Nothing is pretrained in this baseline.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class FeatureTokenizer(nn.Module):
    def __init__(self, n_num: int, d: int, cat_cardinalities: list[int] | None = None):
        super().__init__()
        cat_cardinalities = cat_cardinalities or []
        bound = 1.0 / math.sqrt(d)
        self.W = nn.Parameter(torch.empty(n_num, d).uniform_(-bound, bound))
        self.b = nn.Parameter(torch.empty(n_num, d).uniform_(-bound, bound))
        self.cat_emb = nn.ModuleList([nn.Embedding(card, d) for card in cat_cardinalities])
        self.cls = nn.Parameter(torch.empty(1, 1, d).uniform_(-bound, bound))

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor | None = None) -> torch.Tensor:
        tokens = x_num.unsqueeze(-1) * self.W + self.b          # (N, F, d)
        parts = [self.cls.expand(x_num.shape[0], -1, -1), tokens]
        for i, emb in enumerate(self.cat_emb):
            parts.append(emb(x_cat[:, i]).unsqueeze(1))          # (N, 1, d)
        return torch.cat(parts, dim=1)                           # (N, 1+F+C, d)


class TabTransformer(nn.Module):
    def __init__(self, n_num: int, n_classes: int, cat_cardinalities=None,
                 d: int = 32, heads: int = 4, layers: int = 3, dropout: float = 0.1):
        super().__init__()
        assert d % heads == 0, f"d={d} must be divisible by heads={heads}"
        self.tokenizer = FeatureTokenizer(n_num, d, cat_cardinalities)
        block = nn.TransformerEncoderLayer(
            d_model=d, nhead=heads, dim_feedforward=4 * d, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(block, num_layers=layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, n_classes)

    def embed(self, x_num, x_cat=None):
        """Flow-level representation = final [CLS] token, shape (N, d)."""
        h = self.encoder(self.tokenizer(x_num, x_cat))
        return self.norm(h[:, 0])

    def forward(self, x_num, x_cat=None):
        return self.head(self.embed(x_num, x_cat))

    @torch.no_grad()
    def expand_head(self, new_n_classes: int) -> None:
        """Grow the classifier for new classes; old output rows are copied unchanged."""
        old = self.head
        if new_n_classes <= old.out_features:
            return
        new = nn.Linear(old.in_features, new_n_classes).to(old.weight.device)
        new.weight[: old.out_features] = old.weight
        new.bias[: old.out_features] = old.bias
        self.head = new


if __name__ == "__main__":
    torch.manual_seed(0)
    N, F = 8, 54
    m = TabTransformer(n_num=F, n_classes=3, cat_cardinalities=[4], d=32, heads=4, layers=2).eval()
    x, c = torch.randn(N, F), torch.randint(0, 4, (N, 1))
    tok = m.tokenizer(x, c)
    print("token tensor:", tuple(tok.shape), "(N, 1 CLS + F numeric + C categorical, d)")
    print("flow embedding:", tuple(m.embed(x, c).shape))
    before = m(x, c).detach().clone()
    m.expand_head(5)
    after = m(x, c).detach()
    assert torch.allclose(before, after[:, :3], atol=1e-6), "old logits changed!"
    print("logits:", tuple(after.shape), "| old-class logits preserved after head expansion: OK")
    print("parameters:", sum(p.numel() for p in m.parameters()))
