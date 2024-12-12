import torch
import torch.nn as nn

from SpringProblem.physics_utils import * 


class OneBodyMLP(nn.Module):
    def __init__(self, system, input_size = 1, output_size = 6,  sizes = [32,32]):
        super(OneBodyMLP,self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        list_layers = [nn.Linear(input_size, sizes[0]),
                       nn.ReLU()]
        self.system = system
        self.criterion = nn.MSELoss()

        self.true_position = None
        self.true_velocity = None

        self.device = None

        for i in range(1,len(sizes)):

            list_layers.append(nn.Linear(sizes[i-1], sizes[i]))
            list_layers.append(nn.ReLU())

        list_layers.append(nn.Linear(sizes[-1],output_size))
        self.layers = nn.Sequential(*list_layers)

    def compute_solution(self, time):
        self.true_position, self.true_velocity = self.system.get_solution(time)
        self.true_position = torch.tensor(self.true_position, device=self.device).T
        self.true_velocity = torch.tensor(self.true_velocity, device=self.device).T
        self.true_acceleration = torch.gradient(self.true_velocity, spacing=(time.to(self.device),), dim=0)[0].to(self.device)

    def forward(self,t):
        x = self.layers(t)
        return x
    
    def physics_loss(self, pred, truth, t_max, M, alpha=1, device="cuda", show=False):

        self.device = device

        #TRAIN REGRESISON
        r1_pred = pred[:,:2]
        r1_true = truth[:,:2]
        v1_pred = pred[:,2:]
        v1_true = truth[:,4:6]
        loss = self.criterion(torch.concat([r1_pred,v1_pred], axis=1), torch.concat([r1_true, v1_true], axis=1))        
        
        #TRAIN PHYSICS
        times = torch.linspace(0, t_max, M).unsqueeze(1).to(device)
        if self.true_position is None:
            self.compute_solution(times.squeeze(1).cpu())
        
        # Forward pass to get predictions at each time point
        preds = self.forward(times).squeeze(1)
        r1_pred = preds[:, :2]
        v1_pred = preds[:, 2:]
        v1_pred_derived = torch.gradient(r1_pred, spacing=(times.squeeze(1),), dim=0)[0]
        a1_pred = torch.gradient(v1_pred, spacing=(times.squeeze(1),), dim=0)[0]
        a1_pred_derived = torch.gradient(v1_pred_derived, spacing=(times.squeeze(1),), dim=0)[0]

        r1_true = self.true_position[:,:2]
        v1_true = self.true_velocity[:,:2]
        a1_true = self.true_acceleration[:,:2]

        r2_true = self.true_position[:,2:]
        v2_true = self.true_velocity[:,2:]
        a2_true = self.true_acceleration[:,2:]
        
        vel_loss = self.criterion(v1_pred_derived, v1_pred)

                
        # Calculate the physics-based loss using the subset of points
        equation_loss = self.system.equation_error(torch.concat([r1_pred.T,r2_true.T]),
                                            torch.concat([v1_pred.T,v2_true.T]),
                                            torch.concat([a1_pred.T, a2_true.T]))
        

        # Visualization (if requested)
        if show:
            fig, axs = plt.subplots(1, 3)
            true_points, true_speeds = self.system.get_solution(times.cpu().squeeze(1))
            subset_true_points = true_points.T[random_indices.cpu()]
            subset_true_speeds = true_speeds.T[random_indices.cpu()]
            # subset_true_acc = torch.gradient(torch.tensor(true_speeds).T.to(device), spacing=(times.squeeze(1),), dim=0)[0].cpu()[random_indices.cpu()]
            subset_true_acc = self.system.two_body_equations(times, np.concatenate([true_points, true_speeds])).T[:,4:]
            axs[0].plot(p_preds[:, 0].detach().cpu().numpy(), p_preds[:, 1].detach().cpu().numpy(), color = 'orange')
            axs[0].plot(p_preds[:, 2].detach().cpu().numpy(), p_preds[:, 3].detach().cpu().numpy(), color = 'blue')
            axs[0].scatter(subset_true_points[:, 0], subset_true_points[:, 1], s=1, alpha = 0.3, color = 'orange')
            axs[0].scatter(subset_true_points[:, 2], subset_true_points[:, 3], s=1, alpha = 0.3, color = 'blue')
            
            axs[1].plot(subset_p_preds_dot[:, 0].detach().cpu().numpy(), subset_p_preds_dot[:, 1].detach().cpu().numpy(), color = 'orange')
            axs[1].plot(subset_p_preds_dot[:, 2].detach().cpu().numpy(), subset_p_preds_dot[:, 3].detach().cpu().numpy(), color = 'blue')
            axs[1].scatter(subset_true_speeds[:, 0], subset_true_speeds[:, 1], s=1, alpha = 0.3, color = 'orange')
            axs[1].scatter(subset_true_speeds[:, 2], subset_true_speeds[:, 3], s=1, alpha = 0.3, color = 'blue')

            axs[2].plot(a1[:, 0].detach().cpu().numpy(), a1[:, 1].detach().cpu().numpy(), color = 'orange')
            axs[2].plot(a2[:, 0].detach().cpu().numpy(), a2[:, 1].detach().cpu().numpy(), color = 'blue')
            axs[2].scatter(subset_true_acc[:, 0], subset_true_acc[:, 1], s=1, alpha = 0.3, color = 'orange')
            axs[2].scatter(subset_true_acc[:, 2], subset_true_acc[:, 3], s=1, alpha = 0.3, color = 'blue')
            
            plt.show()
        
        # Momentum-based loss using full predictions (for stable gradient calculation)
        # predicted_momentum = self.system.momentum(p_preds_dot).float()
        # true_momentum = torch.tensor(self.system.initial_momentum, requires_grad=True, device=device).repeat(predicted_momentum.shape[0], 1).float()
        
        # Combine losses
        p_loss = equation_loss * alpha  + vel_loss * alpha #+ self.criterion(predicted_momentum, true_momentum) * 0
        return loss.float() + p_loss.float(), loss.float(), p_loss.float()

        

class TwoBodiesMLP(nn.Module):
    def __init__(self, system, input_size = 1, output_size = 6,  sizes = [32,32]):
        super(TwoBodiesMLP,self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        list_layers = [nn.Linear(input_size, sizes[0]),
                       nn.ReLU()]

        self.system = system
        self.criterion = nn.MSELoss()

        self.true_position = None
        self.true_velocity = None

        self.device = None

        for i in range(1,len(sizes)):

            list_layers.append(nn.Linear(sizes[i-1], sizes[i]))
            list_layers.append(nn.ReLU())

        list_layers.append(nn.Linear(sizes[-1],output_size))
        self.layers = nn.Sequential(*list_layers)

    def compute_solution(self, time):
        self.true_position, self.true_velocity = self.system.get_solution(time)
        self.true_position = torch.tensor(self.true_position, device=self.device).T
        self.true_velocity = torch.tensor(self.true_velocity, device=self.device).T
        self.true_acceleration = torch.gradient(self.true_velocity, spacing=(time.to(self.device),), dim=0)[0].to(self.device)

    def forward(self,t):
        x = self.layers(t)
        return x
    
    def physics_loss(self, pred, truth, t_max, M, alpha=1, device="cuda", show=False):

        self.device = device

        #TRAIN REGRESISON

        loss = self.criterion(pred,truth)        
        
        #TRAIN PHYSICS
        times = torch.linspace(0, t_max, M).unsqueeze(1).to(device)
        if self.true_position is None:
            self.compute_solution(times.squeeze(1).cpu())
        
        # Forward pass to get predictions at each time point
        preds = self.forward(times).squeeze(1)
        r_pred = preds[:, :4]
        v_pred = preds[:, 4:]
        v_pred_derived = torch.gradient(r_pred, spacing=(times.squeeze(1),), dim=0)[0]
        a_pred = torch.gradient(v_pred, spacing=(times.squeeze(1),), dim=0)[0]
        a_pred_derived = torch.gradient(v_pred_derived, spacing=(times.squeeze(1),), dim=0)[0]

        vel_loss = self.criterion(v_pred_derived, v_pred)

                
        # Calculate the physics-based loss using the subset of points
        equation_loss = self.system.equation_error(r_pred.T,
                                            v_pred.T,
                                            a_pred.T)

        # Momentum-based loss using full predictions (for stable gradient calculation)
        # predicted_momentum = self.system.momentum(p_preds_dot).float()
        # true_momentum = torch.tensor(self.system.initial_momentum, requires_grad=True, device=device).repeat(predicted_momentum.shape[0], 1).float()
        
        # Combine losses
        p_loss = equation_loss * alpha  + vel_loss * alpha #+ self.criterion(predicted_momentum, true_momentum) * 0
        return loss.float() + p_loss.float(), loss.float(), p_loss.float()