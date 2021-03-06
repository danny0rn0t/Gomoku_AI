import torch
from torch.utils.data import Dataset, DataLoader

class PolicyGradientNetworkDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __getitem__(self, index):
        return self.data[index][0], self.data[index][1], self.data[index][2]
    def __len__(self):
        return len(self.data)