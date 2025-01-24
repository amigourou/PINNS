# /*-----------------------------------------------------------------------*
# Project: Deep Learning Recruitment

# Copyright (C) 2021 Nintendo, All rights reserved.

# These coded instructions, statements, and computer programs contain proprietary
# information of Nintendo and/or its licensed developers and are protected by
# national and international copyright laws. They may not be disclosed to third
# parties or copied or duplicated in any form, in whole or in part, without the
# prior written consent of Nintendo.

# The content herein is highly confidential and should be handled accordingly.
# *-----------------------------------------------------------------------*/

import torch
import torch.nn as nn

import torch.nn.init as init
import math
import torch
import torch.nn as nn
import torch.optim as optim

device = "cuda" if torch.cuda.is_available() else "cpu"

class SIRENLayer(nn.Module):
    """
    A single SIREN layer with sine activation.
    """
    def __init__(self, in_features, out_features, is_first=False, omega_0=30):
        """
        Initialize a SIREN layer.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            is_first (bool): Whether this is the first layer in the network.
            omega_0 (float): The frequency scaling factor for sine activation.
        """
        super(SIRENLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.omega_0 = omega_0
        self.is_first = is_first

        # Linear transformation
        self.linear = nn.Linear(in_features, out_features)

        # Initialization scheme
        self.init_weights()

    def init_weights(self):
        """
        Initialize the weights of the layer following SIREN initialization rules.
        """
        with torch.no_grad():
            if self.is_first:
                init.uniform_(self.linear.weight, -1 / self.in_features, 1 / self.in_features)
            else:
                init.uniform_(self.linear.weight, 
                              -math.sqrt(6 / self.in_features) / self.omega_0,
                              math.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, x):
        """
        Forward pass through the layer with sine activation.
        """
        return torch.sin(self.omega_0 * self.linear(x))


class SIRENPlanetsModel(nn.Module):
    """
    This class defines the neural network used to encode the location of the planets. \\
    It also includes the method to load the weights.\\
    IMPORTANT: For your submission, you need to make sure that:
    - we can properly load your model. Basically, we'll import this module,
      and call "model_candidate = PlanetsModel(nb_dimensions)" where nb_dimensions
      is in [2, 10, 20].
    - your forward pass returns the gravitational potential. If not, you need to
      justify it.
    - we can properly load your weights. For this, save the name of your weights, for
      each dimension, in answer.json. We'll import this module and call
      "model_candidate.load_weights(PATH_TO_YOUR_WEIGHTS, device)".

    You can modify the __init__, forward and load_weights function as long as it respects the 
    above constraints.

    Attributes:
      model: Neural network used to encode the location of the planets.

    Methods:
      __init__:
        Args:
          nb_dimensions: Number of dimensions in the hyperspace.
      forward:
        Args:
          inputs: Inputs to the neural network.
        Returns:
          potential: Gravitational potential output by the model.
      load_weights:
        Args:
          weight_file: Path to the weight file.
          device: Device on which we load the neural network (GPU, or CPU it not available).
    """

    """
    A SIREN network for representing neural fields.
    """
    def __init__(self, dim):
        """
        Initialize the SIREN network.

        Args:
            input_dim (int): Number of input features.
            hidden_dim (int): Number of hidden units in each layer.
            output_dim (int): Number of output features.
            planets (list): List of planetary positions and radii.
            omega_0 (float): Frequency scaling factor for sine activation.
        """
        super(SIRENPlanetsModel, self).__init__()
        hidden_dim = 256
        omega_0=30
        self.net = nn.Sequential(
            SIRENLayer(dim, hidden_dim, is_first=True, omega_0=omega_0),
            SIRENLayer(hidden_dim, hidden_dim, omega_0=omega_0),
            SIRENLayer(hidden_dim, hidden_dim, omega_0=omega_0),
            SIRENLayer(hidden_dim, hidden_dim, omega_0=omega_0),
            SIRENLayer(hidden_dim, hidden_dim//2, omega_0=omega_0),
            nn.Linear(hidden_dim//2, 1)  # Final layer has no sine activation
        )

    def forward(self, x):
        return self.net(x)

    def load_weights(self, weight_file, device):
        # DEFINE YOUR LOAD_WEIGHTS FUNCTION HERE
        self.load_state_dict(torch.load(weight_file, map_location=device))
        
    def compute_residual(self, x, R, true_lapl, planets):
        x = x.requires_grad_(True).to(device)

        phi = self.forward(x)
        
        dphi_dr = torch.autograd.grad(phi, x, grad_outputs=torch.ones_like(phi), create_graph=True)[0]
        d2phi_dr2 = torch.autograd.grad(dphi_dr, x, grad_outputs=torch.ones_like(dphi_dr), create_graph=True)[0]
        laplacian = torch.sum(d2phi_dr2, dim=-1)

        planets_tensor = torch.stack(planets)
        distances = torch.norm(planets_tensor.unsqueeze(1) - x.unsqueeze(0), dim=-1)
        
        inside_planet = distances < 0.1
        inside_planet = inside_planet.any(dim=0)

        residual = laplacian.unsqueeze(0)
        
        residual = torch.where(inside_planet, residual - 4 / (3 * R**2), residual)
        
        
        true_resid = true_lapl.unsqueeze(0)
        true_resid = torch.where(inside_planet, true_resid - 4 / (3 * R**2), true_resid)
        
        return residual, torch.norm(dphi_dr, dim=1).to(device)
    
  
class NewPlanetsModel(nn.Module):
    """
    This class defines the neural network used to encode the location of the planets. \\
    It also includes the method to load the weights.

    Attributes:
      hidden_size: Size of the intermediate layers.

    Methods:
      __init__:
        Args:
          nb_dimensions: Number of dimensions in the hyperspace.
      forward:
        Args:
          inputs: Inputs to the neural network.
        Returns:
          potential: Gravitational potential output by the model.
      load_weights:
        Args:
          weight_file: Path to the weight file.
          device: Device on which we load the neural network (GPU, or CPU it not available).

    """

    hidden_size = 64

    def __init__(self, nb_dimensions):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(nb_dimensions, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, 1))

    def forward(self, x):
        return self.fc(x)

    def load_weights(self, weight_file, device):
        self.load_state_dict(torch.load(weight_file, map_location=device))
        
    def compute_residual(self, x, R, true_lapl, planets):
      # Ensure input is on the correct device
      x = x.to(device)

      # Infer the grid size and reshape the input
      grid_shape = int(torch.sqrt(torch.tensor(x.shape[0])).item() ) # Assumes x is a flattened grid
      phi = self.forward(x).reshape(grid_shape, grid_shape)  # Reshape to (grid_size, grid_size)

      # Compute grid spacing based on the hypergrid dimensions
      grid_spacing = 1 / (grid_shape - 1)

      # Compute gradients using torch.gradient
      dphi_dx, dphi_dy = torch.gradient(phi, spacing=(grid_spacing, grid_spacing))

      # Compute second derivatives (Laplacian terms)
      d2phi_dx2, _ = torch.gradient(dphi_dx, spacing=(grid_spacing, grid_spacing))
      _, d2phi_dy2 = torch.gradient(dphi_dy, spacing=(grid_spacing, grid_spacing))

      # Laplacian as the sum of second derivatives
      laplacian = d2phi_dx2 + d2phi_dy2

      # Flatten grid for distance computations
      flat_phi = phi.flatten()
      laplacian_flat = laplacian.flatten()
      x_flat = x

      # Compute distances from planets
      distances = torch.norm(planets.unsqueeze(1) - x_flat.unsqueeze(0), dim=-1)

      # Identify points inside any planet
      inside_planet = distances < 0.1
      inside_planet = inside_planet.any(dim=0)

      # Compute the residual
      residual = laplacian_flat.unsqueeze(0)
      residual = torch.where(inside_planet, residual - 4 / (3 * R**2), residual)

      # Return residual, Laplacian, and gradient magnitude
      gradient_magnitude = torch.sqrt(dphi_dx**2 + dphi_dy**2).flatten()
      return residual, laplacian_flat, gradient_magnitude.to(device)

