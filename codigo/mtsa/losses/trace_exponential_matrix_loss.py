import torch
import torch.nn as nn
from typing import Tuple, Any

class TraceExponentialMatrixLoss:
    def __init__(self):
        pass
    
    def calculate(
        self, 
        X: torch.Tensor, 
        y: torch.Tensor, 
        model: nn.Module, 
        **kwargs: Any, 
    ) -> Tuple[Any, ...]:
        A = model.adjacency_matrix
        dimension = A.shape[0]
        acyclicity_coefficient = torch.trace(torch.matrix_exp(A * A)) - dimension
        loss = acyclicity_coefficient
        return (loss, ) 

    def __call__(self, X: torch.Tensor, y: torch.Tensor, model: nn.Module, **kwargs: Any) -> Tuple[Any, ...]:
        return self.calculate(X, y, model, **kwargs)

