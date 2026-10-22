import torch
import torch.nn as nn
from typing import Tuple, Any
from mtsa.loss import Loss

class LagrangeLoss:
    def __init__(self, primal_loss: Loss, dual_losses: list[Loss], lambd=0.01, phi = 0.5, rho=1.0, upsilon=0.2):
        self.lambd = lambd
        self.phi = phi
        self.rho = rho
        self.upsilon = upsilon
        self.primal_loss = primal_loss 
        self.dual_losses = dual_losses
        self.last_value = None
        
    def calculate(
        self, 
        X: torch.Tensor, 
        y: torch.Tensor, 
        model: nn.Module, 
        **kwargs: Any, 
    ) -> Tuple[Any, ...]:
        
        primal = self.primal_loss(X, y, model)[0]

        dual = torch.stack([
            loss_fn(X, y, model)[0]
            for loss_fn in self.dual_losses
        ])

        reduced_part = torch.sum(self.lambd * dual)

        augmented_part = self.phi * torch.sum(
            self.rho * dual.square()
        )

        lagrange_loss = primal + reduced_part + augmented_part
        
        self.last_value = lagrange_loss.item()
        self.last_dual_value = augmented_part.item()

        return (lagrange_loss,)

    def __call__(self, X: torch.Tensor, y: torch.Tensor, model: nn.Module, **kwargs: Any) -> Tuple[Any, ...]:
        return self.calculate(X, y, model, **kwargs)
    
    def update(self):
        self.lambd = self.lambd * self.last_dual_value
        self.rho *= self.upsilon
        
    def __str__(self):
        return str(self.last_value)
        