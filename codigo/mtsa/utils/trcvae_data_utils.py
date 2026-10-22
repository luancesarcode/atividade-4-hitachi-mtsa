import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.nn.init import xavier_uniform_
from torch.utils.data import DataLoader, Dataset


def create_dataLoader(X, y, model, batch_size=32, window_size=12, stride_size=1, drop_last = True):

    X = coerce_2d_array(X)
    effective_window_size, _ = compute_window_starts(X.shape[0], window_size, stride_size, ensure_full_coverage=False)
    if effective_window_size < window_size:
        window_size = effective_window_size
        drop_last = False
    else:
        window_size = effective_window_size

    if y is None:
        y = np.zeros((X.shape[0], 1), dtype=float)
    else:
        y = np.asarray(y)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        elif y.ndim != 2:
            raise ValueError('y should be 1D or 2D array')

    if X.shape[0] != y.shape[0]:
        raise ValueError('X and y must have the same number of rows.')

    X = normalize_array(X, model)
    dataset = Data(X, y, window_size=window_size, stride_size=stride_size)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        persistent_workers=False,
        drop_last=drop_last,
    )


def coerce_2d_array(X):
    if isinstance(X, list):
        if len(X) == 1:
            X = X[0]
        else:
            X = np.asarray(X)
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError('X should be a 2D array')
    return X


def normalize_array(X, model):
    X = np.asarray(X, dtype=np.float32)
    if model.scaler is None:
        model.scaler = StandardScaler()
        X = model.scaler.fit_transform(X)
    else:
        X = model.scaler.transform(X)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X.astype(np.float32, copy=False)


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


def build_window_tensor(X, start, window_size):
    start = int(start)
    end = start + int(window_size)
    x_window = X[start:end].reshape(int(window_size), -1, 1)
    return torch.from_numpy(x_window.copy()).float().transpose(0, 1)


def init_adjacent_matrix(X, device):
    X = np.asarray(X)
    init = torch.zeros([X.shape[-1], X.shape[-1]], device=device)
    init = xavier_uniform_(init).abs()
    init = init.fill_diagonal_(0.0)
    return init.requires_grad_(True)


class Data(Dataset):
    def __init__(self, X, y, window_size=12, stride_size=1):
        super().__init__()

        self.X = np.asarray(X, dtype=np.float32)
        self.y = np.asarray(y).reshape(-1)
        self.window_size = int(window_size)
        self.stride_size = int(stride_size)

        if self.window_size <= 0:
            raise ValueError('window_size must be positive')
        if self.stride_size <= 0:
            raise ValueError('stride_size must be positive')
        if self.X.shape[0] < self.window_size:
            raise ValueError('window_size cannot exceed the number of rows in X')

        _, starts = compute_window_starts(
            len(self.X),
            self.window_size,
            self.stride_size,
            ensure_full_coverage=False,
        )
        self.idx = starts

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, index):
        start = self.idx[index]
        end = start + self.window_size

        x_window = self.X[start:end].reshape(self.window_size, -1, 1)
        x_window = torch.from_numpy(x_window.copy()).float().transpose(0, 1)

        y_window = self.y[start:end]
        y_label = float(np.max(y_window))

        return x_window, torch.tensor(y_label, dtype=torch.float32)
