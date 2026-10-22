import numpy as np
import librosa as lib
import torch
from torch import nn

class Array2Mfcc(nn.Module):
    """
     Gets a numpy array containing audio signals and transforms it into a numpy array containing mfcc signals
     
    """
    def __init__(self, sampling_rate):
        super().__init__()
        self.sampling_rate = sampling_rate
        #self.n_mfcc = n_mfcc

    def forward(self, X, y=None, **fit_params):
        def extract_mfcc(y):
            if self.sampling_rate: 
                return lib.feature.mfcc(y=y, sr=self.sampling_rate)
            else: 
                return lib.feature.mfcc(y=y)
        Xt = np.array(list(map(extract_mfcc, X)))
        Xt = torch.from_numpy(Xt)
        return Xt
