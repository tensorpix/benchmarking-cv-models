import torch
from torch.utils.data import Dataset


class InMemoryDataset(Dataset):
    def __init__(
        self,
        width: int = 224,
        height: int = 224,
        n_channels: int = 3,
        dataset_size: int = int(1e7),
    ):
        super().__init__()
        self.width = width
        self.height = height
        self.n_channels = n_channels
        self.dataset_size = dataset_size

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Must return a tensor of shape C x H x W with values in [0, 1] range.
        """
        return torch.rand(self.n_channels, self.height, self.width, dtype=torch.float32)
