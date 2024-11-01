from torch.utils.data import Dataset
import torch
import numpy as np


class PhysicsDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data
        self.tx = np.concatenate([data["t"][:,None], data["x"][:,None]], axis = 1)
        self.x = data["x"]
        self.xdot = data["xdot"]
        self.xdotdot = data["xdotdot"]

    def __getitem__(self, index):
        tx = self.tx[index,:]
        t, x = tx[0], tx[1]

        xdot,xdotdot = self.xdot[index],self.xdotdot[index]
        return t,x,xdot,xdotdot
    
    def __len__(self):
        return len(self.tx)