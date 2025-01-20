import os
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from BodiesProblem.physics_utils import ThreeBodySystem, TwoBodySystem, OneBodySystem
from BodiesProblem.utils import save_gif, save_plot
from BodiesProblem.models import GravitationalBodiesMLP
from BodiesProblem.siren_utils import GravitationalBodiesSIREN
from BodiesProblem.dataset import PhysicsDataset

from torch.optim.lr_scheduler import LinearLR

def save_results_one_body(physics_model, ground_truth_x, ground_truth_v, ground_truth_a,  epoch, folder, t_max, M, device, train_data):
    with torch.no_grad():
        physics_model.eval()
        test_t = torch.linspace(0, t_max, M).unsqueeze(1).to(device)
        pred = physics_model(test_t)
        test_x_preds = torch.concat([pred[:, :2], ground_truth_x[:,2:4]], axis = 1)
        test_v_preds =  torch.concat([pred[:, 2:4], ground_truth_v[:,2:4]], axis = 1) #torch.gradient(test_x_preds, spacing=(test_t.squeeze(1),), dim=0)[0]
        test_a_preds = torch.gradient(test_v_preds, spacing=(test_t.squeeze(1),), dim=0)[0]
        # test_a_preds = torch.concat([test_a_preds, ground_truth_a[:,2:]], axis = 1)

        os.makedirs(os.path.join(folder,"positions"), exist_ok=True)
        os.makedirs(os.path.join(folder,"velocities"), exist_ok=True)
        os.makedirs(os.path.join(folder,"accelerations"), exist_ok=True)
        save_plot(test_x_preds.T.cpu(), ground_truth_x.T.cpu(), name=os.path.join(folder,"positions", f"position_{epoch}.png"), opt_scatter=train_data)
        save_plot(test_v_preds.T.cpu(), ground_truth_v.T.cpu(), name=os.path.join(folder,"velocities", f"velocity_{epoch}.png"))
        save_plot(test_a_preds.T.cpu(), ground_truth_a.T.cpu(), name=os.path.join(folder,"accelerations", f"acceleration2_{epoch}.png"))

        os.makedirs(os.path.join(folder,"checkpoints"), exist_ok=True)
        torch.save(physics_model, os.path.join(folder,"checkpoints", f"model_ckpt_{epoch}.pth"))

        # loss, mes_loss, p_loss = physics_model.physics_loss(
        #             x_pred.float(), truth.float().to(device), t_max, M, alpha=alpha, show=False
        #         )
        # fig,axs = plt.subplots(1,3)

if __name__ == "__main__":
    device = "cuda"
    print("Training on GPU: ", torch.cuda.is_available())
    torch.manual_seed(42)
    np.random.seed(42)
    # Define system parameters
    mass = [1,0.75,10] #[0.0001, 1]
    G = 1
    num_epoch = 100000
    t_max = 5
    M = 500
    alpha = 0.001

    # Initial positions and velocities
    # r1 = np.array([2, 0])
    # r2 = np.array([0, 0])
    # r3 = np.array([-2, 0])
    # v1 = np.array([0, 0.5])
    # v2 = np.array([0, -0.1])
    # v3 = np.array([0.5, 0])

    r1 = np.array([0.5, 0])
    r2 = np.array([-0.5, 0.5])
    r3 = np.array([0, 0])



    v1 = np.array([-0.2, 1])
    v2 = np.array([0, 0.5])
    v3 = np.array([0, 0])





    system = ThreeBodySystem(r1, r2, r3, v1, v2, v3, (0, t_max), mass, G)
    t = np.linspace(0, t_max, M)
    test_t = torch.linspace(0, t_max, M).unsqueeze(1).to(device)
    # ground_truth_x, ground_truth_v,times = system.get_solution()
    train_data = system.generate_noisy_datapoints(0, 1, 16, 0.0005)
    val_data = system.generate_noisy_datapoints(0, t_max, 128, 0.0005)
    # Plot initial data


    # ground_truth_x = torch.tensor(ground_truth_x.T, device = device)
    # ground_truth_v = torch.tensor(ground_truth_v.T, device = device)
    # ground_truth_a = torch.gradient(ground_truth_v, spacing=(test_t.squeeze(1),), dim=0)[0].cpu()
    # derivatives = system.body_equations(t, torch.concat([ground_truth_x.T, ground_truth_v.T]))

    # Prepare dataset and dataloader
    train_set = PhysicsDataset(train_data)
    val_set = PhysicsDataset(val_data)
    train_loader = DataLoader(train_set, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)



    # Initialize model, optimizer, and TensorBoard writer
    physics_model = GravitationalBodiesSIREN(system,
                                           num_predict_bodies=1,
                                           input_size=1,
                                           output_size=4,
                                           hidden_size=128,
                                           hidden_layers=4,
                                           omega_0=3,
                                           alpha_vel=alpha,
                                           alpha_eq=alpha,
                                           num_physics_points=M,
                                           device = device)
    # physics_model = GravitationalBodiesMLP(system,
    #                                        num_predict_bodies=1,
    #                                        input_size=1,
    #                                        output_size=4,
    #                                        backbone_sizes=[64,64,64],
    #                                        regressors_sizes=[32],
    #                                        alpha_vel=alpha,
    #                                        alpha_eq=alpha,
    #                                        num_physics_points=M,
    #                                        device = device)

    initial_lr = 0.0001  # Starting learning rate
    final_lr = initial_lr / 100  # Final learning rate 1000 times smaller

    optimizer = Adam(physics_model.parameters(), lr=initial_lr)

    # Define linear learning rate scheduler
    lr_scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=final_lr / initial_lr, total_iters=num_epoch)

    log_dir=f"runs/physics_model_experiment/{datetime.now().date()}/{datetime.now().hour}_{datetime.now().minute}_{datetime.now().second}"
    os.makedirs(log_dir,exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)


    for epoch in tqdm(range(num_epoch)):
        total_loss = 0
        for t, x, xdot in train_loader:
            optimizer.zero_grad()
            t = t.unsqueeze(1).float().to(device)
            truth = torch.concat([x, xdot], axis=1)
            loss, mse_loss, p_loss = physics_model.compute_loss(t, truth)
            
            loss.backward()
            
            # Gradient clipping (e.g., clip gradients to a maximum norm of 1.0)
            clip_grad_norm_(physics_model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += loss


        # Calculate average training loss
        avg_train_loss = total_loss / len(train_loader)

        # Validate and log losses to TensorBoard
        if epoch % 25 == 0:
            val_loss = physics_model.test_model(val_loader)
            print(f"Epoch {epoch} | Train Loss: {avg_train_loss/len(train_loader):.6f} | mse Loss: {mse_loss/len(train_loader):.6f} | Physics Loss: {p_loss/len(train_loader):.6f} | Validation Loss: {val_loss:.6f}")

            # Log losses to TensorBoard
            writer.add_scalar("Loss/Train", avg_train_loss, epoch)
            writer.add_scalar("Loss/Physics", p_loss/len(train_loader), epoch)
            writer.add_scalar("Loss/Validation", val_loss, epoch)
        
        # Step the learning rate scheduler
        lr_scheduler.step()

        if epoch % 500 == 0:
            physics_model.save_results(epoch, log_dir, train_data["x"])

    writer.close()

    # Generate predictions and save GIFs

            # save_plot(ground_truth_x, "groundtruth.png")
            # save_plot(ground_truth_v.T, "groundtruth_v.png")
            # save_plot(ground_truth_a.T, "groundtruth_a.png")
