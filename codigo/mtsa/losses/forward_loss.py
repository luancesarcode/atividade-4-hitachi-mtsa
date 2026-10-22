import torch
import torch.nn as nn
from typing import Tuple, Any

class ForwardLoss:
    def __init__(self):
        pass
    
    def calculate(
        self, 
        X: torch.Tensor, 
        y: torch.Tensor, 
        model: nn.Module, 
        **kwargs: Any, 
    ) -> Tuple[Any, ...]:
        y_hat = model(X)
        loss = y_hat
        return (loss, ) 

    def __call__(self, X: torch.Tensor, y: torch.Tensor, model: nn.Module, **kwargs: Any) -> Tuple[Any, ...]:
        return self.calculate(X, y, model, **kwargs)

