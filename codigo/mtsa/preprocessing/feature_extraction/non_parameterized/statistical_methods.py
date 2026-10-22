import itertools as ite
from functools import reduce

import numpy as np
from scipy.fft import rfft
from scipy.stats import kurtosis, skew
from torch import nn

from mtsa.utils.correlation_networks import PearsonCorrelationNetwork

#region feature extraction strategies

class MagnitudeMeanFeatureMfcc(nn.Module):
    
    def __init__(self) -> None:
        super().__init__()
        
    def fit(self, X, y=None):
        return self
    
    def forward(self, X, y=None, **fit_params):
        Xt = X.mean(axis=2)
        return Xt
    
class MagnitudeStdFeatureMfcc(nn.Module):
    
    def __init__(self) -> None:
        super().__init__()
        
    def fit(self, X, y=None):
        return self
    
    def forward(self, X, y=None, **fit_params):
        Xt = X.std(axis=2)
        return Xt
    
class CorrelationFeatureMfcc(nn.Module):
    
    def __init__(self) -> None:
        super().__init__()
    
    def fit(self, X, y=None):
        return self
    
    def forward(self, X, y=None, **fit_params):
        def get_triu(G):
            return G[np.triu_indices(len(G), k = 1)]
        corr = PearsonCorrelationNetwork()
        Xt = map(corr.get_correlation, X)
        Xt = map(get_triu, Xt)
        Xt= np.array(list(Xt)) 
        return Xt
    
class RootMeanSquareFeature(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
    def fit(self, X, y=None):
        return self
    
    def forward(self, X, y=None, **fit_params):
        def forward_sample(x):
            xt = np.sqrt(np.mean(x**2))
            return xt
        
        Xt = []
        
        if len(X.shape) == 2:
            Xt = np.array([[forward_sample(x)] for x in X])
        else:      
            Xt = forward_sample(X)
        
        return Xt
    
class SquareRootOfAmplitude(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
    def fit(self, X, y=None):
        return self
    
    def forward(self, X, y=None, **fit_params):
        def forward_sample(x):
            xt = (np.mean(np.sqrt(np.absolute(x)))) ** 2
            return xt
        
        Xt = []
        
        if len(X.shape) == 2:
            Xt = np.array([[forward_sample(x)] for x in X])
        else:      
            Xt = forward_sample(X)
        
        return Xt
    
class Kurtosis(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
    def fit(self, X, y=None):
        return self
    
    def forward(self, X, y=None, **fit_params):
        def forward_sample(x):
            xt = kurtosis(x)
            return xt
        
        Xt = []
        
        if len(X.shape) == 2:
            Xt = np.array([[forward_sample(x)] for x in X])
        else:      
            Xt = forward_sample(X)
        
        return Xt
    
class Skewness(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
    def fit(self, X, y=None):
        return self
    
    def forward(self, X, y=None, **fit_params):
        def forward_sample(x):
            xt = skew(x)
            return xt
        
        Xt = []
        
        if len(X.shape) == 2:
            Xt = np.array([[forward_sample(x)] for x in X])
        else:      
            Xt = forward_sample(X)
        
        return Xt
    
class Peak2Peak(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
    def fit(self, X, y=None):
        return self
    
    def forward(self, X, y=None, **fit_params):
        def forward_sample(x):
            xt = np.max(x) - np.min(x)
            return xt
        
        Xt = []
        
        if len(X.shape) == 2:
            Xt = np.array([[forward_sample(x)] for x in X])
        else:      
            Xt = forward_sample(X)
        
        return Xt
    
class CrestFactor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.root_mean_square_feature = RootMeanSquareFeature()
        
    def fit(self, X, y=None):
        return self
    
    def forward(self, X, y=None, **fit_params):
        def forward_sample(x):
            xt = np.max(np.absolute(x)) / self.root_mean_square_feature.forward(x)
            return xt
        
        Xt = []
        
        if len(X.shape) == 2:
            Xt = np.array([[forward_sample(x)] for x in X])
        else:      
            Xt = forward_sample(X)
        
        return Xt
    
class ImpulseValue(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
    def fit(self, X, y=None):
        return self
    
    def forward(self, X, y=None, **fit_params):
        def forward_sample(x):
            xt = np.max(np.absolute(x)) / np.mean(np.absolute(x))
            return xt
        
        Xt = []
        
        if len(X.shape) == 2:
            Xt = np.array([[forward_sample(x)] for x in X])
        else:      
            Xt = forward_sample(X)
        
        return Xt
    
class MarginFactor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.square_root_of_amplitude = SquareRootOfAmplitude()
        
    def fit(self, X, y=None):
        return self
    
    def forward(self, X, y=None, **fit_params):
        def forward_sample(x):
            xt = np.max(np.absolute(x)) / self.square_root_of_amplitude.forward(x)
            return xt
        
        Xt = []
        
        if len(X.shape) == 2:
            Xt = np.array([[forward_sample(x)] for x in X])
        else:      
            Xt = forward_sample(X)
        
        return Xt
    
class ShapeFactor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.root_mean_square_feature = RootMeanSquareFeature()
        
    def fit(self, X, y=None):
        return self
    
    def forward(self, X, y=None, **fit_params):
        def forward_sample(x):
            xt = self.root_mean_square_feature.forward(x) / np.mean(np.absolute(x))
            return xt
        
        Xt = []
        
        if len(X.shape) == 2:
            Xt = np.array([[forward_sample(x)] for x in X])
        else:      
            Xt = forward_sample(X)
        
        return Xt

class KurtosisFactor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.kurtosis = Kurtosis()
        self.root_mean_square_feature = RootMeanSquareFeature()
        
    def fit(self, X, y=None):
        return self
    
    def forward(self, X, y=None, **fit_params):
        def forward_sample(x):
            xt = self.kurtosis.forward(x) / (self.root_mean_square_feature.forward(x) ** 4)
            return xt
        
        Xt = []
        
        if len(X.shape) == 2:
            Xt = np.array([[forward_sample(x)] for x in X])
        else:      
            Xt = forward_sample(X)
        
        return Xt
    
class FrequencyCenter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
    def fit(self, X, y=None):
        return self
    
    def forward(self, X, y=None, **fit_params):
        def forward_sample(x):
            xt = 2 * np.abs(rfft(x)) / x.size
            xt = np.mean(xt)
            return xt
        
        Xt = []
        
        if len(X.shape) == 2:
            Xt = np.array([[forward_sample(x)] for x in X])
        else:      
            Xt = forward_sample(X)
        
        return Xt
    
class RootMeanSquareFrequency(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
    def fit(self, X, y=None):
        return self
    
    def forward(self, X, y=None, **fit_params):
        def forward_sample(x):
            xt = 2 * np.abs(rfft(x)) / x.size
            xt = np.sqrt(np.mean(xt**2))
            return xt
        
        Xt = []
        
        if len(X.shape) == 2:
            Xt = np.array([[forward_sample(x)] for x in X])
        else:      
            Xt = forward_sample(X)
        
        return Xt
    
    
class RootFrequencyVariance(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.frequency_center = FrequencyCenter()
        
    def fit(self, X, y=None):
        return self
    
    def forward(self, X, y=None, **fit_params):        
        def forward_sample(x):
            xt = 2 * np.abs(rfft(x)) / x.size
            xt = np.sqrt(np.mean((xt - self.frequency_center.forward(xt)) ** 2))
            return xt
            
        Xt = []
        
        if len(X.shape) == 2:
            Xt = np.array([[forward_sample(x)] for x in X])
        else:      
            Xt = forward_sample(X)
            
        return Xt

#endregion

#region methods

def get_features(self):
    number_features = np.arange(1, len(FEATURES)+1)
    def combinations(r): return ite.combinations(FEATURES, r)
    all_combinations = map(combinations, number_features)
    features = reduce(lambda x, y: ite.chain(x, y), all_combinations)
    return features

#endregion

#region pre-arranged set of features

FEATURES = [
    ("M", MagnitudeMeanFeatureMfcc()), 
    ("S", MagnitudeStdFeatureMfcc()), 
    ("C", CorrelationFeatureMfcc())
]

StatisticalSignalDescriptors = [
    ("rootMeanSquareFeature", RootMeanSquareFeature()),
    ("squareRootOfAmplitude", SquareRootOfAmplitude()),
    ("kurtosis", Kurtosis()),
    ("skewness", Skewness()),
    ("peak2Peak", Peak2Peak()),
    ("crestFactor", CrestFactor()),
    ("impulseValue", ImpulseValue()),
    ("marginFactor", MarginFactor()),
    ("shapeFactor", ShapeFactor()),
    ("kurtosisFactor", KurtosisFactor()),
    ("frequencyCenter", FrequencyCenter()),
    ("rootMeanSquareFrequenc", RootMeanSquareFrequency()),
    ("rootFrequencyVariance", RootFrequencyVariance())
]

#endregion

