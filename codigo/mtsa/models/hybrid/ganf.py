import time
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.nn.utils import clip_grad_value_
from tqdm import tqdm
from torch.nn.init import xavier_uniform_

from mtsa.models.gnn import GNN
from mtsa.models.components.ganf.NF import MAF, RealNVP
from mtsa.utils.data import get_dataloader
from mtsa.utils.trcvae_data_utils import create_dataLoader

"""
    GRAPH-AUGMENTED NORMALIZING FLOWS

    This class implements the model described in the paper:
        GRAPH-AUGMENTED NORMALIZING FLOWS FOR ANOMALY DETECTION OF MULTIPLE TIME SERIES

    Key concepts and techniques from the paper have been adapted and implemented here.
    Any deviations or modifications from the original methodology are noted in the implementation details.

    Reference:
    [1] X. Wu, Y. Park, S. Tirthapura, "GRAPH-AUGMENTED NORMALIZING FLOWS FOR ANOMALY DETECTION OF MULTIPLE TIME SERIES,"
    AAAI, vol. 36, no. 8, 2022.  DOI: 10.1609/aaai.v36i8.20829
"""


class GANF(nn.Module):
    def __init__(
        self,
        n_blocks=6,
        input_features=1,
        hidden_size=32,
        n_hidden=1,
        model="MAF",
        device=None,
        adjacent_matrix=None,
        time_series_amount=2,
    ):
        super().__init__()
        self.scaler = None

        self.is_fitted_ = False
        self.input_features = input_features

        self.device = (
            torch.device(f"cuda:{0!s}" if torch.cuda.is_available() else "cpu")
            if device is None
            else device
        )
        self.adjacency_matrix = (
            self.__init_adjacent_matrix(time_series_amount, self.device)
            if adjacent_matrix is None
            else adjacent_matrix
        )

        self.rnn = nn.LSTM(
            input_size=input_features,
            hidden_size=hidden_size,
            batch_first=True,
            dropout=0.0,
            device=self.device,
        )

        self.gcn = GNN(input_size=hidden_size, hidden_size=hidden_size)

        self.nf = self._get_probability_model(
            n_blocks, input_features, hidden_size, n_hidden, model
        )

        self.to(device=self.device)

    @property
    def name(self):
        return "GANF"

    @property
    def adjacency_matrix(self):
        return self._adjacency_matrix

    @adjacency_matrix.setter
    def adjacency_matrix(self, adjacent_matrix):
        self._adjacency_matrix = adjacent_matrix

    def __coerce_data(self, X):
        # B x W x D to B x D x W x input_features
        X = X.reshape(
            (X.shape[0], X.shape[2], X.shape[1], self.input_features))
        return X.to(torch.float32)

    def forward(self, X, A=None):
        X = self.__coerce_data(X)

        if A is None:
            A = self.adjacency_matrix

        full_shape = X.shape
        norm_A = self.__apply_row_normalization(A)

        X = X.reshape((X.shape[0] * X.shape[1], X.shape[2], X.shape[3]))
        h, _ = self.rnn(X)

        h = h.reshape((full_shape[0], full_shape[1], h.shape[1], h.shape[2]))

        h = self.gcn(h, norm_A)

        h = h.reshape((-1, h.shape[3]))
        X = X.reshape((-1, full_shape[3]))

        log_prob = self.nf.log_prob(X, h).reshape([full_shape[0], -1])
        log_prob = log_prob.mean(dim=1)
        log_prob_x = log_prob.mean()
        return -log_prob_x

    def __apply_row_normalization(self, adjacent_matrix, eps=1e-8):
        adjacent_matrix_norm = torch.divide(
            adjacent_matrix.T, adjacent_matrix.sum(dim=1).detach() + eps
        ).T

        return adjacent_matrix_norm

    def _get_probability_model(
        self, n_blocks, input_features, hidden_size, n_hidden, model
    ):
        if model == "MAF":
            # Masked auto regressive flow
            return MAF(
                n_blocks,
                input_features,
                hidden_size,
                n_hidden,
                cond_label_size=hidden_size,
                batch_norm=False,
                activation="tanh",
            )
        else:
            # real-valued non-volume preserving (real NVP)
            return RealNVP(
                n_blocks,
                input_features,
                hidden_size,
                n_hidden,
                cond_label_size=hidden_size,
                batch_norm=False,
            )

    def __init_adjacent_matrix(self,time_series_amount, device):
        init = torch.zeros(
            [time_series_amount, time_series_amount], device=device)
        init = xavier_uniform_(init).abs()
        init = init.fill_diagonal_(0.0)
        return init.requires_grad_(True)
