import os
from datetime import datetime
import time

import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from GravitationalFieldProblem.dataset import GravitationalDataset
from GravitationalFieldProblem.models import NewPlanetsModel

if __name__ == "__main__":
    device = "cuda"
    print("Training on GPU: ", torch.cuda.is_available())
    torch.manual_seed(42)
    np.random.seed(42)
    # Define system parameters
    num_samples = 20  # Number of training points
    d = 2              # Dimensionality of hyperspace
    R = 0.1             # Radius of the planet
    learning_rate = 1e-5
    num_epochs = 50000
    alpha = 0.001

    total_loss_list = []
    loss_physic_list = []
    loss_mse_list = []
    epoch_list = []
    lr_list = []

    val_total_loss_list = []
    planets = torch.tensor([[0.5,0.5]])
    mse_loss = nn.MSELoss()

    # Generate training data
    dataset = GravitationalDataset(dim=d, num_points=num_samples, planets=planets, random_points_factor=10, refinement_steps=0,
                                        noise_level=0.8)
    val_set = GravitationalDataset(d, 50, planets, random_points_factor=5, refinement_steps=0)

    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=1000, shuffle=False)

    # Initialize the neural network, loss, and optimizer
    # model = SIRENPlanetsModel(dim=d)
    model = NewPlanetsModel(d)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.00001)

    # Define the linear learning rate scheduler
    scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.00001, total_iters=num_epochs)

    model.train()
    model.to(device)

    # Define linear learning rate scheduler

    log_dir=f"runs/gravitational_field_experiments/{datetime.now().date()}/{datetime.now().hour}_{datetime.now().minute}_{datetime.now().second}"
    os.makedirs(log_dir,exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)


    # Training loop

    linspaces = [torch.linspace(0, 1, 150) for _ in range(2)]
    grids = torch.meshgrid(*linspaces, indexing='ij')  # 'ij' indexing for Cartesian coordinates
    hypergrid = torch.stack(grids, dim=-1).reshape(-1, 2)  # Flatten into (N, D)

    # hypergrid_polar = dataset._convert_to_polar(hypergrid, center = planets)
    for epoch in tqdm(range(num_epochs)):
        start_epoch = time.time()
        for x, phi_true, grad_true, lapl_true in dataloader:
            # Move data to device
            x = x.detach().to(device)
            # r = x[:,0].unsqueeze(1)
            phi_true = phi_true.detach().to(device)
            grad_true = grad_true.detach().to(device)
            lapl_true = lapl_true.detach().to(device)

            # Forward pass
            phi_pred = model(x).squeeze(-1)

            # Compute losses
            loss_mse = mse_loss(phi_pred, phi_true)
            residual, lapl, gradient = model.compute_residual(hypergrid, R, lapl_true.to(device), planets.to(device))
            loss_residual =  torch.tensor([0], device=device) + alpha* torch.mean(residual**2)
            total_loss = loss_mse + loss_residual

            # Backward pass
            optimizer.zero_grad()
            total_loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # Step the scheduler at the end of each epoch
        scheduler.step()

        # Print the learning rate for debugging purposes
        lr_list.append(scheduler.get_last_lr()[0])
        epoch_list.append(epoch)
        total_loss_list.append(total_loss.detach().cpu().item())
        loss_physic_list.append(loss_residual.detach().cpu().item())
        loss_mse_list.append(loss_mse.detach().cpu().item())

        # Logging
        if (epoch) % 250 == 0:
            temp_total_loss = []
            for x, phi_true, grad_true, lapl_true in val_loader:
                # Move data to device
                x = x.to(device)
                # r = x[:,0].unsqueeze(1)
                phi_true = phi_true.to(device)
                phi_pred = model(x).squeeze(-1)
                grad_true = grad_true.to(device)

                # Compute losses
                
                # residual,lapl, gradient = model.compute_residual(x, R, lapl_true.to(device), planets.to(device))
                # val_loss_residual = 0.001 * torch.mean(residual**2)
                val_loss_mse = mse_loss(phi_pred, phi_true).detach()
                temp_total_loss.append(val_loss_mse)
            
            val_total_loss = torch.mean(torch.Tensor(temp_total_loss))
            val_total_loss_list.append(val_total_loss.item())

            writer.add_scalar("Loss/Train", loss_mse.item(), epoch)
            writer.add_scalar("Loss/Physics", loss_residual.item(), epoch)
            writer.add_scalar("Loss/Validation", val_total_loss.item(), epoch)
            
                
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss.item():.6f}, "
                f"MSE: {loss_mse.item():.6f}, Residual: {loss_residual.item():.6f}, "
                f"Val Loss: {val_total_loss.item():.6f}, LR : {lr_list[-1]}")
            torch.save(model.state_dict(), os.path.join(log_dir, "checkpoints", f"model_{epoch}.pth"))
    # Save the trained model
    torch.save(model.state_dict(), os.path.join(log_dir, "checkpoints", f"model_final.pth"))
    writer.close()