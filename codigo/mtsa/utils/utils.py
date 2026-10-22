from sklearn.base import BaseEstimator, TransformerMixin
import librosa as lib
from functools import reduce
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import FeatureUnion
#import torch
#from torch.nn.init import xavier_uniform_

NORMAL = 0
ABNORMAL = 1
class Wav2Array(nn.Module):
    def __init__(self, 
                 sampling_rate=None,
                 mono=True
                 ):
        
        super().__init__()
        self.sampling_rate = sampling_rate
        self.mono = mono

    def forward(self, X, y=None, **fit_params):
        def get_array(f):
            if self.sampling_rate:
                s, _ = lib.load(f, sr=self.sampling_rate, mono=self.mono)
                return s
            else:
                s, _ = lib.load(f, mono=self.mono)
                return s
        np.random.seed(0)
        Xt = np.array(list(map(get_array, X)))
        return Xt

class Demux2Array(BaseEstimator, TransformerMixin):
    def __init__(self, 
                 channel=0
                 ):
        self.channel = channel

    def fit(self, root, y=None, **fit_params):
        self.is_fitted_ = True
        return self

    def transform(self, X, y=None, **fit_params):
        def get_array(Xi):
            # multi_channel_data, sr = file_load(wav_name)
            if Xi.ndim <= 1:
                return Xi
            return np.array(Xi)[self.channel, :]
        
        Xt = np.array(list(map(get_array, X)))
        return Xt

from sklearn.model_selection import BaseShuffleSplit
from sklearn.utils.validation import check_random_state 
from sklearn.model_selection._split import _validate_shuffle_split
import numpy as np
class AbnormalSplit(BaseShuffleSplit):

    def __init__(
        self, n_splits=10, random_state=None
    ):
        super().__init__(
            n_splits=n_splits,
            random_state=random_state
        )

    def _iter_indices(self, X, y=None, groups=None):
        NORMAL = 0
        ABNORMAL = 1
        ind_abnormal = np.where(y == ABNORMAL)[0]
        ind_normal = np.where(y == NORMAL)[0]

        rng = check_random_state(self.random_state)
        for i in range(self.n_splits):
            # random partition
            permutation_normal = rng.permutation(ind_normal)
            ind_test_normal = permutation_normal[0:len(ind_abnormal)]
            ind_train_normal = permutation_normal[len(ind_abnormal):]

            ind_train = ind_train_normal
            
            ind_test = np.concatenate([ind_test_normal, ind_abnormal])
            yield ind_train, ind_test
            
    def _iter_indices_trainning_testing_val(self, X, y=None, groups=None, val_size=0.2):
        NORMAL = 1
        ABNORMAL = 0
        ind_abnormal = np.where(y == ABNORMAL)[0]
        ind_normal = np.where(y == NORMAL)[0]

        rng = check_random_state(self.random_state)
        for i in range(self.n_splits):
            # random partition
            permutation_normal = rng.permutation(ind_normal)
            ind_test_normal = permutation_normal[0:len(ind_abnormal)]
            ind_train_normal = permutation_normal[len(ind_abnormal):]

            ind_train = ind_train_normal
            
            ind_test = np.concatenate([ind_test_normal, ind_abnormal])
            yield ind_train, ind_test

import os
import glob
# from functools import reduce
def get_files_from_path(path):
    pattern = "*.wav"
    path_wav = lambda signal_class: glob.glob(os.path.join(path, signal_class, pattern))
    return path_wav(path)

def get_X_y_from_normal_abnormal(normal, abnormal):
    y0 = NORMAL * np.ones(len(normal))
    y1 = ABNORMAL * np.ones(len(abnormal))
    X = np.concatenate([normal, abnormal])
    y = np.concatenate([y0, y1])
    return X, y
    
def get_files_from_path_classes(path):
    X0 = get_files_from_path(os.path.join(path, "normal"))
    X1 = get_files_from_path(os.path.join(path, "abnormal"))
    X, y = get_X_y_from_normal_abnormal(X0, X1)
    return X, y


def files_train_test_split(path, random_state=None):
    X, y = get_files_from_path_classes(path)
    ind_train, ind_test = next(AbnormalSplit(random_state=random_state,n_splits=1).split(X, y))
    X_train, X_test, y_train, y_test = X[ind_train], X[ind_test], y[ind_train], y[ind_test]
    return X_train, X_test, y_train, y_test

def files_train_test_split_dcase2020_task2(path, pattern=''):
    """
    class_id_sample.wav
    
    :Examples:
    normal_id_00_00000110.wav
     
    """
    
    X_train_normal = glob.glob(os.path.join(path, 'train', f'normal_{pattern}*'))
    X_train_abnormal = glob.glob(os.path.join(path, 'train', f'anomaly_{pattern}*'))
    
    X_test_abnormal = glob.glob(os.path.join(path, 'test', f'anomaly_{pattern}*'))
    X_test_normal = glob.glob(os.path.join(path, 'test', f'normal_{pattern}*'))
    
    X_train, y_train = get_X_y_from_normal_abnormal(X_train_normal, X_train_abnormal)
    X_test, y_test = get_X_y_from_normal_abnormal(X_test_normal, X_test_abnormal)
    
    return X_train, X_test, y_train, y_test


def files_train_test_split_combined(paths):
    def reduce_data(d1, d2):
        data = zip(d1, d2)
        data = list(data)
        X_train, X_test, y_train, y_test = [np.concatenate([d1, d2]) for (d1,d2) in data]
        return X_train, X_test, y_train, y_test
    data = list(map(files_train_test_split, paths))
    X_train, X_test, y_train, y_test = reduce(reduce_data, data)
    return X_train, X_test, y_train, y_test

class mfcc2Tensor(BaseEstimator, TransformerMixin):
    
    def __init__(self) -> None:
        super().__init__()
        
    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self
    
    def transform(self, X, y=None, **fit_params):
        X = X.reshape(X.shape[0], X.shape[2], X.shape[1])
        return X


# def build_signal_processing_steps(use_array2mfcc=False, sampling_rate=16000, mono=False, use_featureUnion=False, use_mfcc2tensor = False,wav2array=False, standard_normalization=True):
#     steps = []
    
#     if wav2array:
#         wav2array = Wav2Array(sampling_rate=sampling_rate, mono=mono)
#         steps.append(("wav2array", wav2array))

#     if use_array2mfcc:
#         array2mfcc = Array2Mfcc(sampling_rate=sampling_rate)
#         steps.append(("array2mfcc", array2mfcc))

#     if use_featureUnion:
#         features = FeatureUnion(FEATURES)
#         steps.append(("features", features))
        
#     if use_mfcc2tensor:
#         steps.append(("mfcc2Tensor", mfcc2Tensor()))
        
#     return steps
