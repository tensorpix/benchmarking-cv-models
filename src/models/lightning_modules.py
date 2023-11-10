import lightning as L
import torch
from torch import nn


class LitClassification(L.LightningModule):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.loss = nn.CrossEntropyLoss()

    def training_step(self, batch, batch_idx):
        y_hat = self.model(batch)
        y = torch.rand_like(y_hat)

        loss = self.loss(y_hat, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=2e-5)
