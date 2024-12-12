import os

import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt

from utils import save_plot

class SineActivation(nn.Module):
    """Sinusoidal activation function for SIREN."""
    def forward(self, x):
        return torch.sin(x)


class SirenLayer(nn.Module):
    """A single SIREN layer with sine activation and specialized initialization."""
    def __init__(self, in_features, out_features, omega_0=30.0, is_first=False):
        super(SirenLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.omega_0 = omega_0
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features)

        # Initialization
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                # First layer uses a uniform initialization
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                # Other layers scale weights based on omega_0
                bound = np.sqrt(6 / self.in_features) / self.omega_0
                self.linear.weight.uniform_(-bound, bound)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))


class SIREN(nn.Module):
    """Full SIREN model with a backbone and output layers."""
    def __init__(self, in_features, out_features, hidden_features, hidden_layers, omega_0=30.0, omega_0_out=30.0):
        super(SIREN, self).__init__()
        self.net = nn.ModuleList()

        # Input layer
        self.net.append(SirenLayer(in_features, hidden_features, omega_0=omega_0, is_first=True))

        # Hidden layers
        for _ in range(hidden_layers):
            self.net.append(SirenLayer(hidden_features, hidden_features, omega_0=omega_0))

        # Output layer (linear for simplicity, optionally use Sine)
        self.net.append(nn.Linear(hidden_features, out_features))
        self.omega_0_out = omega_0_out

    def forward(self, x):
        for layer in self.net[:-1]:
            x = layer(x)
        x = self.net[-1](x)
        return x


class GravitationalBodiesSIREN(nn.Module):
    """SIREN-based gravitational body model."""
    def __init__(self, system,
                 num_predict_bodies=2,
                 input_size=1,
                 output_size=4, 
                 hidden_size=64,
                 hidden_layers=3,
                 omega_0=30.0,
                 alpha_vel=0.01,
                 alpha_eq=0.001,
                 num_physics_points=1000,
                 device="cuda"):
        
        super(GravitationalBodiesSIREN, self).__init__()
        self.device = device

        self.system = system
        self.input_size = input_size
        self.output_size = output_size
        self.num_predict_bodies = num_predict_bodies
        self.num_bodies = self.system.num_bodies

        self.alpha_vel = alpha_vel
        self.alpha_eq = alpha_eq
        self.num_physics_points = int(num_physics_points*3)
        self.physics_times_uniform = torch.linspace(0, self.system.t_span[1], self.num_physics_points).unsqueeze(1).to(self.device)
        uniform_times = torch.linspace(0, 1, self.num_physics_points).to(self.device)
        reversed_exponential_times = 1 - torch.exp(-5 * uniform_times)
        reversed_exponential_times = (reversed_exponential_times - reversed_exponential_times.min()) / (reversed_exponential_times.max() - reversed_exponential_times.min())
        self.physics_times = (reversed_exponential_times * self.system.t_span[1]).unsqueeze(1).to(self.device)

        plt.scatter(uniform_times.cpu(), self.physics_times.squeeze(1).cpu())
        plt.show()


        # SIREN backbone
        self.backbone = SIREN(in_features=input_size, 
                              out_features=hidden_size,
                              hidden_features=hidden_size, 
                              hidden_layers=hidden_layers,
                              omega_0=omega_0).to(device)

        # SIREN regressors for each body
        self.regressors = nn.ModuleList([
            SIREN(in_features=hidden_size, 
                  out_features=output_size,
                  hidden_features=hidden_size // 2, 
                  hidden_layers=1,
                  omega_0=omega_0).to(device)
            for _ in range(num_predict_bodies)
        ])
        
        self.criterion = nn.MSELoss()

        self.compute_solution()
        self.to(device)
    
    def forward(self, t):
        # Pass time input through the backbone
        features = self.backbone(t)
        # Predict output for each body
        predictions = [regressor(features) for regressor in self.regressors]
        return predictions

    def compute_solution(self):
        with torch.no_grad():
            self.true_position, self.true_velocity = self.system.get_solution(self.physics_times_uniform.squeeze(1).cpu())
            # self.physics_times = torch.tensor(self.physics_times, device= self.device).unsqueeze(1).to(torch.float32)
            self.true_position = torch.tensor(self.true_position, device=self.device).T.to(torch.float32)
            self.true_velocity = torch.tensor(self.true_velocity, device=self.device).T.to(torch.float32)
            self.true_acceleration = torch.gradient(self.true_velocity, spacing=(self.physics_times_uniform.squeeze(1),), dim=0)[0].to(self.device).to(torch.float32)

    def physics_loss(self):
        predictions = self.forward(self.physics_times)
        preds_r = [pred[:,:2] for pred in predictions]
        preds_v = [pred[:,2:] for pred in predictions]

        if self.num_bodies>self.num_predict_bodies:
            for k in range(self.num_predict_bodies, self.num_bodies):
                preds_r.append(self.true_position[:,k*2:(k+1)*2])
                preds_v.append(self.true_velocity[:,k*2:(k+1)*2])
            
        preds_r = torch.concat(preds_r, axis = 1)
        preds_v = torch.concat(preds_v, axis = 1)

        times = self.physics_times.squeeze(1)
        # print(times)
        preds_v_derived = torch.gradient(preds_r, spacing=(times,), dim=0)[0]
        preds_a = torch.gradient(preds_v, spacing=(times,), dim=0)[0]
        vel_loss = self.criterion(preds_v_derived, preds_v)

        equation_loss = self.system.equation_error(preds_r.T,
                                            preds_v.T,
                                            preds_a.T)

        return equation_loss.to(torch.float32) * self.alpha_eq + vel_loss.to(torch.float32) * self.alpha_vel

    def regression_loss(self,t,truth):
        predictions = self.forward(t)
        total_loss = torch.tensor(0.0, device=self.device)

        for i,pred in enumerate(predictions):
            truth_ri = truth[:,i*2:(i+1)*2].to(self.device).to(torch.float32)
            truth_vi = truth[:,(self.num_bodies + i)*2:(self.num_bodies + i + 1)*2].to(self.device).to(torch.float32)
            total_loss += self.criterion(pred,torch.concat([truth_ri, truth_vi], axis = 1))

        return total_loss.to(torch.float32)

    def compute_loss(self, t, truth):
        #TRAIN REGRESISON
        reg_loss = self.regression_loss(t,truth)
        #TRAIN PHYSICS
        p_loss = self.physics_loss()
        # Combine losses
        loss = p_loss + reg_loss
        return loss, reg_loss, p_loss
    
    @torch.no_grad()
    def test_model(self, loader):
        total_loss = 0
        for t, x, xdot in loader:
            t = t.unsqueeze(1).float().to(self.device)
            truth = torch.concat([x, xdot], axis=1)
            loss, _, _ = self.compute_loss(t, truth.to(self.device))
            total_loss += loss
        return total_loss / len(loader)
    
    @torch.no_grad()
    def save_results(self, epoch, folder, train_data):
        self.eval()
        predictions = self.forward(self.physics_times)
        preds_r = [pred[:,:2] for pred in predictions]
        preds_v = [pred[:,2:] for pred in predictions]

        if self.num_bodies>self.num_predict_bodies:
            for k in range(self.num_predict_bodies, self.num_bodies):
                preds_r.append(self.true_position[:,k*2:(k+1)*2])
                preds_v.append(self.true_velocity[:,k*2:(k+1)*2])
            
        preds_r = torch.concat(preds_r, axis = 1)
        preds_v = torch.concat(preds_v, axis = 1)

        preds_v_derived = torch.gradient(preds_r, spacing=(self.physics_times.squeeze(1),), dim=0)[0]
        preds_a = torch.gradient(preds_v, spacing=(self.physics_times.squeeze(1),), dim=0)[0]
        # test_a_preds = torch.concat([test_a_preds, ground_truth_a[:,2:]], axis = 1)

        os.makedirs(os.path.join(folder,"positions"), exist_ok=True)
        os.makedirs(os.path.join(folder,"velocities"), exist_ok=True)
        os.makedirs(os.path.join(folder,"accelerations"), exist_ok=True)
        save_plot(preds_r.T.cpu(), self.true_position.T.cpu(), name=os.path.join(folder,"positions", f"position_{epoch}.png"), opt_scatter=train_data)
        save_plot(preds_v.T.cpu(), self.true_velocity.T.cpu(), name=os.path.join(folder,"velocities", f"velocity_{epoch}.png"))
        save_plot(preds_a.T.cpu(), self.true_acceleration.T.cpu(), name=os.path.join(folder,"accelerations", f"acceleration2_{epoch}.png"))

        os.makedirs(os.path.join(folder,"checkpoints"), exist_ok=True)

        torch.save(self.state_dict(), os.path.join(folder,"checkpoints", f"model_ckpt_{epoch}.pth"))
