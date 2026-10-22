import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np

def get_dataloader(
    X,
    y,
    window_size=20,
    stride_size=20,
    batch_size=32,
    shuffle=False,
    num_workers=0,
    persistent_workers=False,
    drop_last=True,
):
    Xy_dataloader = DataLoader(
        Data(X=X, y=y, window_size=window_size, stride_size=stride_size),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        drop_last=drop_last,
    )
    return Xy_dataloader


def build_window_tensor(X, start, window_size):
    start = int(start)
    end = start + int(window_size)
    x_window = X[start:end].reshape(int(window_size), -1, 1)
    return torch.from_numpy(x_window.copy()).float()

def coerce_2d_array(X):
    X = X.reshape((X.shape[0]* X.shape[1], X.shape[2]))
    if isinstance(X, list):
        if len(X) == 1:
            X = X[0]
        else:
            X = np.asarray(X)
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError('X should be a 2D array')
    return X

def compute_window_starts(n_rows, window_size, stride_size, ensure_full_coverage=False):
    n_rows = int(n_rows)
    window_size = int(window_size)
    stride_size = int(stride_size)

    if n_rows <= 0:
        raise ValueError('n_rows must be positive')
    if window_size <= 0:
        raise ValueError('window_size must be positive')
    if stride_size <= 0:
        raise ValueError('stride_size must be positive')

    effective_window_size = min(window_size, n_rows)
    starts = np.arange(0, n_rows - effective_window_size + 1, stride_size, dtype=int)
    if starts.size == 0:
        starts = np.array([0], dtype=int)

    if ensure_full_coverage:
        if stride_size > effective_window_size:
            raise ValueError(
                'Point-level scoring requires stride_size <= effective_window_size so every timestamp is covered.'
            )
        last_start = n_rows - effective_window_size
        if starts[-1] != last_start:
            starts = np.unique(np.append(starts, last_start))

    return effective_window_size, starts


class Data(Dataset):
    def __init__(self, X, y, window_size=20, stride_size=20):
        self.coerce(X, y)
        self.window_size = window_size
        self.stride = stride_size
        self.n_files = self.X.shape[0]  # number F of data matrix TxD
        self.n_datapoints = self.X.shape[1]  # T
        self.n_features = self.X.shape[2]  # D

        self.windows_per_file = 0
        start = 0
        while start + window_size <= self.n_datapoints:
            self.windows_per_file += 1
            start += stride_size
        self.total_windows = self.n_files * self.windows_per_file

    def coerce(self, X, y):
        if X.ndim > 3:
            raise ValueError("X should be 2D or 3D array")

        elif X.ndim == 2:
            X = X.reshape(-1, X.shape[0], X.shape[1])
            y = y.reshape(-1, y.shape[0])
        elif y.shape[0] == X.shape[0]:
            y = np.array([np.full(X.shape[1], y_i) for y_i in y])

        if (y.shape[0] > 1 and y.shape[0]*y.shape[1] != X.shape[0]*X.shape[1]) or (y.shape[0] == 1 and X.shape[1] != y.shape[1]):
            raise ValueError("X and y must have the same number of rows.")

        self.X = X
        self.y = y

    def __len__(self):
        return self.total_windows

    def __getitem__(self, idx):
        file_idx = idx // self.windows_per_file
        window_idx = idx % self.windows_per_file
        start = window_idx * self.stride
        end = start + self.window_size
        if end > self.n_datapoints:
            raise IndexError("It was not possible create a window: end > n_datapoints")
        X_windowed = self.X[file_idx][start:end]
        y_windowed = self.y[file_idx][start:end]
        return X_windowed, y_windowed
