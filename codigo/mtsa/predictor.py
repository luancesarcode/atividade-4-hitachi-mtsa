import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Protocol, runtime_checkable, List, Any
from abc import ABC

@runtime_checkable
class Predictor(Protocol):    
    def predict(self, 
            model: nn.Module, 
            dataloader: DataLoader, 
        ) -> List[float]:
        ...

class BasePredictor(ABC):
    def __init__(self):
        self.time=None

    def __call__(self, model: nn.Module, dataloader: DataLoader) -> Any:
        return self.predict(model, dataloader)

    def predict(self, model: nn.Module, dataloader: DataLoader) -> Any:
        raise NotImplementedError