import os
from time import time
import torch
import torch.nn as nn

from BodiesProblem.utils import save_gif, save_plot

class LinearBlock(nn.Module):
    def __init__(self, sizes = [32,6]):
        super(LinearBlock, self).__init__()
        list_layers = [nn.Linear(sizes[0], sizes[1])]

        for i in range(1,len(sizes)-1):
            list_layers.append(nn.ReLU())
            list_layers.append(nn.Linear(sizes[i], sizes[i+1]))
            
        self.layers = nn.Sequential(*list_layers)

    def forward(self, x):
        return self.layers(x)


class GravitationalBodiesMLP(nn.Module):
    def __init__(self, system,
                 num_predict_bodies = 2,
                 input_size = 1,
                 output_size = 4, 
                 backbone_sizes = [64,64],
                 regressors_sizes = [32],
                 alpha_vel = 0.01,
                 alpha_eq = 0.001,
                 num_physics_points = 1000,
                 device = "cuda"):
        
        super(GravitationalBodiesMLP,self).__init__()
        self.device = device

        self.system = system
        self.input_size = input_size
        self.output_size = output_size
        self.num_predict_bodies = num_predict_bodies
        self.num_bodies = self.system.num_bodies

        self.alpha_vel = alpha_vel
        self.alpha_eq = alpha_eq
        self.num_physics_points = num_physics_points
        self.physics_times = torch.linspace(0, self.system.t_span[1], self.num_physics_points).unsqueeze(1).to(self.device)

        self.backbone = LinearBlock(sizes = [input_size] + backbone_sizes).to(device)

        self.regressors = [LinearBlock(sizes = [backbone_sizes[-1]] + regressors_sizes + [output_size]).to(device)
                              for _ in range(num_predict_bodies)]
        
        self.criterion = nn.MSELoss()

        

        self.compute_solution()

        self.to(device)

    
    def forward(self,t):
        features = self.backbone(t)
        predictions = [regressor(features) for regressor in self.regressors]
        return predictions
    
    def compute_solution(self):
        with torch.no_grad():
            self.true_position, self.true_velocity = self.system.get_solution(self.physics_times.squeeze(1).cpu())
            # self.physics_times = torch.tensor(self.physics_times, device= self.device).unsqueeze(1).to(torch.float32)
            self.true_position = torch.tensor(self.true_position, device=self.device).T.to(torch.float32)
            self.true_velocity = torch.tensor(self.true_velocity, device=self.device).T.to(torch.float32)
            self.true_acceleration = torch.gradient(self.true_velocity, spacing=(self.physics_times.squeeze(1),), dim=0)[0].to(self.device).to(torch.float32)

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