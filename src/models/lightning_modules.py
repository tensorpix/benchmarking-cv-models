import lightning as L
import torch
from torch import nn


class LitClassification(L.LightningModule):
    def __init__(self, model: nn.Module, optimizer: torch.optim = torch.optim.Adam):
        super().__init__()
        self.model = model
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = optimizer

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        y_hat = self.model(batch)
        y = torch.rand_like(y_hat)

        loss = self.loss(y_hat, y)
        return loss

    def configure_optimizers(self):
        return self.optimizer(self.parameters(), lr=2e-5)
