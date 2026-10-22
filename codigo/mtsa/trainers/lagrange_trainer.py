import torch
import torch.nn as nn
import time
from torch.utils.data import DataLoader
from typing import Optional, Any
from mtsa.loss import Loss
from tqdm import tqdm
from mtsa.losses.lagrange_loss import LagrangeLoss
from mtsa.predictor import BasePredictor
from mtsa.trainer import BaseTrainer


class LagrangeTrainer(BaseTrainer):
    def __init__(self):
        super().__init__()

    def fit(self,
            model: nn.Module,
            optimizer: torch.optim.Optimizer,
            loss: LagrangeLoss,
            train_loader: DataLoader,
            val_loader: Optional[DataLoader] = None,
            **kwargs: Any
            ) -> nn.Module:

        val_loss = None
        epoch_mean_errors = []
        epochs = kwargs.get('epochs', 30)
        interactions = kwargs.get('interactions', 2)
        patience = kwargs.get('patience', torch.inf)
        pbar = self.__build_pbar(epochs, interactions)

        for interaction in range(interactions):
            for epoch in range(epochs):
                model.train()
                total_loss = 0.0
                for x_batch, y_batch in train_loader:
                    batch_loss, = loss(x_batch, y_batch, model)
                    optimizer.zero_grad()
                    batch_loss.backward()
                    optimizer.step()
                    total_loss += batch_loss.item()
                avg_loss = total_loss / len(train_loader)
                pbar.set_postfix(loss=str(loss))
                pbar.update(1)
                epoch_mean_errors.append(avg_loss)

                if val_loader is not None:
                    current_val_loss = self.__get_val_loss(
                        model, loss, val_loader)

                    if val_loss is None:
                        val_loss = current_val_loss
                    elif current_val_loss > val_loss:
                        patience -= 1
                    if patience == 0:
                        break

            loss.update()

        self.time = pbar.n / pbar.format_dict["elapsed"]  # mean of times
        self.min_error = epoch_mean_errors
        return model

    def __get_val_loss(self, model, loss, val_loader):
        val_losses = []
        for x_val_batch, y_val_batch in val_loader:
            model.eval()
            val_batch_loss, = loss(x_val_batch, y_val_batch, model)
            val_losses.append(val_batch_loss.item())
        return val_losses

    def __build_pbar(self, epochs, interactions):
        return tqdm(
            total=interactions * epochs,
            desc="Training",
            unit="epoch",
            dynamic_ncols=True,
        )
