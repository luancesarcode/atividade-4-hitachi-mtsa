import numpy as np
from sklearn.preprocessing import StandardScaler
from torch import nn


class Standardization(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.is_fitted_ = False
        self.scaler = StandardScaler()
        
    @property
    def name(self):
        return "Standardization"
        
    def fit(self, X, y=None):
        if X.ndim > 3:
            raise ValueError('X should be 2D or 3D array')
        if X.ndim == 3:
            X_sample = X[0]
        else:
            X_sample = X
            
        self.scaler.fit(X=X_sample)
        self.is_fitted_ = True
        return X
    
    def forward(self, X, y=None, **kwargs):
        if self.is_fitted_:
            if X.ndim > 3:
                raise ValueError('X should be 2D or 3D array')
            if X.ndim == 3:
                X = np.array(list(map(self.scaler.transform, X)))
            else:
                X = self.scaler.transform(X)
        else: 
            X = self.fit(X=X)
            
        return X, y
        
            
