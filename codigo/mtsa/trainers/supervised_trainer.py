import torch
import torch.nn as nn
import time
from torch.utils.data import DataLoader
from typing import Optional, Any
from mtsa.loss import Loss
from tqdm import tqdm
from mtsa.trainer import BaseTrainer

class SupervisedTrainer(BaseTrainer):
    def __init__(self):
        super().__init__()
    
    def fit(self, 
            model: nn.Module, 
            optimizer: torch.optim.Optimizer, 
            loss: Loss, 
            train_loader: DataLoader, 
            val_loader: Optional[DataLoader] = None, 
            **kwargs: Any
        ) -> nn.Module:
        epochs = kwargs.get('epochs', 30)
        pbar = tqdm(range(epochs), desc=f"Training | {epochs} Epochs", unit="epoch", dynamic_ncols=True)
        start_time = time.time()
        
        for epoch in pbar:
            model.train()
            total_loss = 0.0

            for x_batch, y_batch in train_loader:
                x_batch = x_batch
                y_batch = y_batch 
                
                batch_loss, = loss(x_batch, y_batch, model)
  
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

                total_loss += batch_loss.item()
                
            avg_loss = total_loss / len(train_loader)

            pbar.set_postfix({"loss": f"{avg_loss:.4e}"})

        elapsed_time = time.time() - start_time
        self.time = elapsed_time
        self.min_error = avg_loss
        return model