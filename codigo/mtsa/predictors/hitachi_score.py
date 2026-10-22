"""Adapta a reconstrução do Hitachi ao StandardPredictor do framework."""
import torch
from torch import nn


class HitachiScore(nn.Module):
    """MSE por janela: maior = mais anômalo.

    StandardPredictor entrega (batch, tempo, variáveis, 1). O Hitachi
    reconstrói a última dimensão de (batch, tempo, variáveis).
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        if x.ndim == 4 and x.shape[-1] == 1:
            x = x.squeeze(-1)
        if x.ndim != 3 or x.shape[-1] != self.model.input_dim:
            raise ValueError('Esperado (batch, tempo, input_dim), com canal singleton opcional.')
        error = torch.square(x - self.model(x))
        return error.mean(dim=(1, 2))
