import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from mtsa.loss import Loss
from mtsa.predictor import BasePredictor
from mtsa.utils.data import build_window_tensor, coerce_2d_array, compute_window_starts


class StandardPredictor(BasePredictor):
    def __init__(self):
        super().__init__()

    def predict(
        self, function: nn.Module | Loss, dataloader: DataLoader, pointwise: bool, aggregation="max", device=None
    ):

        self.device = self._resolve_device(device)

        if dataloader is None:
            raise ValueError("Dataloader is none")

        if type(function) is nn.Module:
            function.eval()

        y_hat = None
        y_true = None

        with torch.no_grad():
            if pointwise:
                y_hat, y_true = self.__predict_points(
                    function=function, dataloader=dataloader, aggregation=aggregation)
            else:
                y_hat, y_true = self.__predict_windows(function, dataloader)

        y_hat = np.array(y_hat).reshape(-1,1)
        y_true = np.array(y_true).reshape(-1,1)
        return y_hat, y_true

    def __predict_windows(self, function, dataloader):
        y_hat_list = [], y_true_list = []
        for x_batch, y_batch in dataloader:
            y_hat = function(x_batch)
            y_hat_list.append(y_hat)
            y_true_list.append(y_batch)

        return y_hat_list, y_true_list

    def __predict_points(self, function: nn.Module, dataloader: DataLoader, aggregation="max"):
        X = dataloader.dataset.X
        X = coerce_2d_array(X)
        y = dataloader.dataset.y

        effective_window_size, starts = compute_window_starts(
            X.shape[0], dataloader.dataset.window_size, dataloader.dataset.stride, ensure_full_coverage=True
        )

        if aggregation not in {"max", "mean"}:
            raise ValueError(
                f"Unknown aggregation: {aggregation}. Expected one of ['max', 'mean'].")

        if aggregation == "max":
            point_scores = np.full(X.shape[0], -np.inf, dtype=float)
        else:
            point_scores = np.zeros(X.shape[0], dtype=float)
            counts = np.zeros(X.shape[0], dtype=float)

        with torch.no_grad():
            for start in starts:
                x_window = build_window_tensor(
                    X, start, effective_window_size).unsqueeze(0).to(self.device)
                window_score = float(function(x_window))
                end = start + effective_window_size
                if aggregation == "max":
                    point_scores[start:end] = np.maximum(
                        point_scores[start:end], window_score)
                else:
                    point_scores[start:end] += window_score
                    counts[start:end] += 1.0

        if aggregation == "mean":
            point_scores = point_scores / counts

        y_hat = point_scores.tolist()
        return y_hat, y

    @staticmethod
    def _resolve_device(device):
        if isinstance(device, torch.device):
            return device
        if isinstance(device, int):
            if torch.cuda.is_available():
                return torch.device(f"cuda:{device}")
            return torch.device("cpu")
        if device is None:
            return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

