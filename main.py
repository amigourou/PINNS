import matplotlib.pyplot as plt

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from physics_utils import DampenSpringSystem
from model import MLP
from dataset import PhysicsDataset

@torch.no_grad()
def test_model(model,loader, t_max, M,alpha):
    total_loss = 0
    for t,x,xdot,xdotdot in loader:
        t = t.unsqueeze(1).float().to(device)
        x_pred = model(t)
        loss = physics_model.physics_loss(x_pred,x.to(device), t_max, M, alpha = alpha)
        total_loss+=loss

    return total_loss/len(loader)

if __name__ == "__main__":

    device = "cuda"

    mass = 1
    mu = 1
    k = [10]

    num_epoch = 8000
    t_max = 6
    M = 60
    alpha = 1

    for i,kk in enumerate(k): 
        system = DampenSpringSystem(1,0,mass, kk, mu)
        t = np.arange(0,10,0.2)
        x = system.get_solution(t)
        
        train_data = system.generate_noisy_datapoints(0,2.5,15,0.005,True)
        val_data = system.generate_noisy_datapoints(2.6,t_max,30,0.005,True)

        train_set =PhysicsDataset(train_data) 
        val_set = PhysicsDataset(val_data)

        train_loader = DataLoader(train_set, batch_size=3, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

        physics_model = MLP(system,1,1,[64,64]).to(device)
        optimizer = Adam(physics_model.parameters(), lr = 0.01)

        losses = []

        for epoch in tqdm(range(num_epoch)):
            total_loss = 0
            for t,x,xdot,xdotdot in train_loader:
                optimizer.zero_grad()

                t = t.unsqueeze(1).float().to(device)
                x_pred = physics_model(t).to(device)
                loss = physics_model.physics_loss(x_pred.float(),x.float().to(device), t_max,M, alpha = alpha)

                loss.backward()
                optimizer.step()
                
                total_loss+=loss
            if epoch%25 == 0:
                val_loss = test_model(physics_model, val_loader, t_max,M,alpha)

                print(f"Epoch {epoch} train loss: {total_loss/len(train_loader)} | validation loss: {val_loss}")

            losses.append(total_loss/len(train_loader))

with torch.no_grad():
    physics_model.eval().to("cpu")
    test_t = torch.linspace(0,t_max,50)
    test_t = test_t.unsqueeze(1)
    test_x_preds = physics_model(test_t)
    p_preds_dot = torch.gradient(test_x_preds.squeeze(1),spacing = (test_t.squeeze(1),))[0]
    p_preds_dotdot = torch.gradient(p_preds_dot,spacing = (test_t.squeeze(1),))[0]

    plt.scatter(test_t, test_x_preds)
    plt.scatter(train_data["t"], train_data["x"], label = "truth train")
    plt.scatter(val_data["t"], val_data["x"], label = "val")
    plt.show()