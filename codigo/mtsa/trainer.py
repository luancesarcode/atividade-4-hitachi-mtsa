import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from typing import Protocol, runtime_checkable, Optional, Any
from mtsa.loss import Loss
from abc import ABC

@runtime_checkable
class Trainer(Protocol):    
    def fit(self, 
            model: nn.Module, 
            optimizer: torch.optim.Optimizer, 
            loss: Loss, 
            train_loader: DataLoader, 
            val_loader: Optional[DataLoader], 
            **kwargs: Any
        ) -> nn.Module:
        ...

class BaseTrainer(ABC):
    def __init__(self):
        self.min_error=None
        self.time=None

    def __call__(self, 
                model: nn.Module, 
                optimizer: torch.optim.Optimizer, 
                loss: Loss, 
                train_loader: DataLoader, 
                val_loader: Optional[DataLoader] = None, 
                **kwargs: Any
            ) -> nn.Module:
        return self.fit(model, optimizer, loss, train_loader, val_loader, **kwargs)

    def fit(self, 
            model: nn.Module, 
            optimizer: torch.optim.Optimizer, 
            loss: Loss, 
            train_loader: DataLoader, 
            val_loader: Optional[DataLoader] = None, 
            **kwargs: Any
        ) -> nn.Module:
        raise NotImplementedError