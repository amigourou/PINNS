from torch.utils.data import Dataset
import numpy as np


class PhysicsDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data
        self.tx = np.concatenate([data["t"][None,:], data["x"], data["xdot"]], axis = 0).T
        self.x = data["x"]
        self.xdot = data["xdot"]

    def __getitem__(self, index):
        tx = self.tx[index,:]
        t, x, xdot = tx[0], tx[1:5], tx[5:]

        return t,x, xdot
    
    def __len__(self):
        return len(self.tx)