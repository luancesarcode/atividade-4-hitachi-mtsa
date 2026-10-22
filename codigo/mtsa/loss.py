import torch
import torch.nn as nn
from typing import Protocol, Any, Tuple, runtime_checkable


@runtime_checkable
class Loss(Protocol):
    def calculate(
        self, 
        X: torch.Tensor, 
        y: torch.Tensor, 
        model: nn.Module, 
        *args: Any, 
        **kwargs: Any
    ) -> Tuple[Any, ...]:
        ...
        
    def __call__(
        self, 
        X: torch.Tensor, 
        y: torch.Tensor, 
        model: nn.Module, 
        *args: Any, 
        **kwargs: Any
    ) -> Tuple[Any, ...]:
        ...
