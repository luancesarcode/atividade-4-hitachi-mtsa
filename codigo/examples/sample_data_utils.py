"""Dados do tutorial; 0 = normal, 1 = anômalo. Anomalias são sintéticas."""
from pathlib import Path
from typing import Any, Literal, overload

import numpy as np
import pandas as pd


def get_acoustic_sample_data():
    from mtsa import files_train_test_split
    path_input_1 = Path(__file__).resolve().parent / 'sample_data' / 'machine_type_1' / 'id_00'
    X_train, X_test, y_train, y_test = files_train_test_split(str(path_input_1))
    return X_train, y_train, X_test, y_test


def set_abnormal_data(x, step, y_test, features=None, magnitude=2, start_point=None):
    """Quatro intervalos de duração step, in-place; início zero é válido.

    Correção: o fim do primeiro intervalo é início + duração, não step.
    """
    first_start = step if start_point is None else start_point
    intervals = [
        (first_start, first_start + step),
        (2 * step, 3 * step),
        (4 * step, 5 * step),
        (len(x) - (100 + step), len(x) - 100),
    ]
    if step <= 0 or len(y_test) != len(x) or any(
        not 0 <= start < end <= len(x) for start, end in intervals
    ):
        raise ValueError('Intervalos devem ser não vazios e estar dentro dos dados.')
    if features is None:
        features = slice(None)
    anomaly = magnitude * np.sin(np.pi / 4)
    for start, end in intervals:
        x[start:end, features] += anomaly
        y_test[start:end] = 1
    return x, y_test


@overload
def get_tabular_sample_data(*, split_mode: str = 'chronological', return_metadata: Literal[True]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]: ...


@overload
def get_tabular_sample_data(*, split_mode: str = 'chronological', return_metadata: Literal[False] = False) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: ...


def get_tabular_sample_data(*, split_mode='chronological', return_metadata=False):
    """Retorna X_train, y_train, X_test, y_test.

    chronological: primeiros 60% normais para desenvolvimento; últimos 40%
    para teste, com anomalias injetadas apenas no teste.
    legacy: montagem por classe do tutorial, apenas para comparação.
    """
    data_path = Path(__file__).resolve().parent / 'linear_ts_with_confounder_n5000_vars6_lag2_gaussian.csv'
    df = pd.read_csv(data_path)
    df = df.drop(columns=["U"])
    df = df.drop(columns=["time"])

    X = df.to_numpy()
    y = np.zeros(X.shape[0])

    if split_mode == 'chronological':
        cut = int(0.6 * len(X))
        X_train, X_test = X[:cut].copy(), X[cut:].copy()
        y_train = np.zeros(cut, dtype=int)
        y_test = np.zeros(len(X_test), dtype=int)
        train_indices, test_indices = np.arange(cut), np.arange(cut, len(X))
        X_test, y_test = set_abnormal_data(X_test, 200, y_test, features=[4, 5], magnitude=2, start_point=150)
        X_test, y_test = set_abnormal_data(X_test, 13, y_test, features=[1, 0], magnitude=7, start_point=200)
        result = (X_train, y_train, X_test, y_test)
        if return_metadata:
            return (*result, {'split_mode': split_mode, 'train_indices': train_indices.tolist(),
                             'test_indices': test_indices.tolist(), 'features': list(df.columns)})
        return result
    if split_mode != 'legacy':
        raise ValueError("split_mode deve ser 'chronological' ou 'legacy'.")

    X, y = set_abnormal_data(X, 200, y, features=[4, 5], magnitude=2, start_point=150)
    X, y = set_abnormal_data(X, 13, y, features=[1, 0], magnitude=7, start_point=200)
    mask_normal = y == 0
    mask_abnormal = y == 1

    X_normal = X[mask_normal]
    X_abnormal = X[mask_abnormal]

    anomaly_quantity = len(X_abnormal)

    X_test = np.vstack([X_normal[:anomaly_quantity], X_abnormal])

    X_train = X_normal[anomaly_quantity:]

    y_test = np.concatenate([np.zeros(anomaly_quantity, dtype=int), np.ones(anomaly_quantity, dtype=int)])

    y_train = np.zeros_like(X_train[:, 0])

    if return_metadata:
        normal, abnormal = np.flatnonzero(mask_normal), np.flatnonzero(mask_abnormal)
        return X_train, y_train, X_test, y_test, {
            'split_mode': split_mode, 'train_indices': normal[anomaly_quantity:].tolist(),
            'test_indices': np.concatenate([normal[:anomaly_quantity], abnormal]).tolist(),
            'features': list(df.columns)}
    return X_train, y_train, X_test, y_test
