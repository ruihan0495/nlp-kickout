import torch
from torch.utils.data import Dataset

class MyDataset(Dataset):

    def __init__(self, tokens, window_size, batch_size) -> None:
        self.tokens = tokens
        self.indices = torch.randint(len(self.tokens['input_ids'][0])-window_size, (batch_size,))
        self.window_size = window_size

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        i = self.indices[index]
        input_ids = self.tokens['input_ids'][0][i:i+self.window_size]
        target_ids = self.tokens['input_ids'][0][i+1:i+self.window_size+1]
        return input_ids, target_ids
    
    