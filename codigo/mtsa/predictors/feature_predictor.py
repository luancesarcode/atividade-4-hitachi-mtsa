import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from mtsa.predictor import BasePredictor

class FeaturePredictor(BasePredictor):
    def __init__(self):
        super().__init__()

    def predict(self, model: nn.Module, dataloader: DataLoader):
        model.eval()
        pred_feature = []

        with torch.no_grad():
            for x_batch, _ in dataloader:
                x_batch = x_batch
                x_feature = model(x_batch)
                pred_feature.append(x_feature) 

        pred_feature = torch.cat(pred_feature, dim=0)
        return pred_feature