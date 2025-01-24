import torch
from torch.utils.data import Dataset


class GravitationalDataset(Dataset):
    """
    PyTorch Dataset for a gravitational potential field in [0, 1]^D.
    """
    def __init__(self, dim, num_points, planets, random_points_factor=0, 
                 refinement_steps=2, refinement_threshold=0.1, noise_level=0.0):
        """
        Initialize the dataset.
        
        Args:
            dim (int): Dimension of the hypercube.
            num_points (int): Number of points per dimension initially.
            planets (list of tuples): List of planet positions and radii.
                                      Format: [(position, radius), ...].
            random_points_factor (int): Factor for additional global random points.
            refinement_steps (int): Number of refinement iterations for adaptive sampling.
            refinement_threshold (float): Threshold for high variation in gradient or Laplacian.
            noise_level (float): Standard deviation of Gaussian noise to add to the training data.
        """
        self.points = []
        self.dim = dim
        self.planets = planets
        self.radius = 0.1
        self.noise_level = noise_level

        # Initial sampling: points around planets and random global points
        for center in planets:
            center = torch.tensor(center, dtype=torch.float32)
            assert center.shape[0] == dim, "Planet position must match dimension of space."
            points = self._generate_points_in_sphere(center, 0.1, num_points)
            self.points.append(points)
        
        random_points = torch.rand(num_points * random_points_factor, dim)
        self.points.append(random_points)
        self.points = torch.clamp(torch.cat(self.points, dim=0).requires_grad_(True), 0, 1)

        # Compute potential and gradients
        self._compute_potentials_and_derivatives()
        
        # Refinement: Adaptive sampling
        for _ in range(refinement_steps):
            high_variation_points = self._identify_high_variation_points(refinement_threshold)
            self.points = torch.cat([self.points, high_variation_points], dim=0).requires_grad_(True)
            self._compute_potentials_and_derivatives()
        
        

    def _compute_potentials_and_derivatives(self):
        """Compute potentials, gradients, and Laplacian for the current points."""
        self.potentials = torch.zeros((self.points.shape[0],))
        for center in self.planets:
            self.potentials += self._planet_potential(self.points, torch.tensor(center), self.radius)
        self.potentials += torch.rand_like(self.potentials) * self.noise_level
        dphi_dr = torch.autograd.grad(self.potentials, self.points, grad_outputs=torch.ones_like(self.potentials), create_graph=True)[0]
        d2phi_dr2 = torch.autograd.grad(dphi_dr, self.points, grad_outputs=torch.ones_like(dphi_dr), create_graph=True)[0]
        self.gradients = torch.norm(dphi_dr, dim=1)
        self.laplacian = torch.sum(d2phi_dr2, dim=-1)

    def _identify_high_variation_points(self, threshold):
        """
        Identify regions of high variation in gradient or Laplacian and sample more points.
        
        Args:
            threshold (float): Threshold for high variation.
        
        Returns:
            torch.Tensor: Additional points sampled in high-variation regions.
        """
        high_variation_mask = (self.gradients > threshold) | (torch.abs(self.laplacian) > threshold)
        high_variation_points = self.points[high_variation_mask]

        # Add jitter to high-variation points for more dense sampling in the region
        jitter = torch.randn_like(high_variation_points) * 0.05
        new_points = high_variation_points + jitter

        # Ensure new points remain within [0, 1]^D
        new_points = torch.clamp(new_points, 0.0, 1.0)
        return new_points

    @staticmethod
    def _generate_points_in_sphere(center, radius, num_points):
        """Generate random points uniformly distributed inside a sphere."""
        dim = center.shape[0]
        points = torch.randn(num_points, dim)  # Sample from a normal distribution
        points = points / torch.norm(points, dim=1, keepdim=True)  # Normalize to lie on the sphere
        radii = torch.rand(num_points).pow(1 / dim) * radius  # Generate radii
        points = points * radii.unsqueeze(-1) + center  # Scale and shift to the sphere
        return points

    @staticmethod
    def _planet_potential(points, planet_position, radius):
        """Compute the gravitational potential induced by a single planet for all points."""
        square_distance = torch.sum((planet_position - points) ** 2, dim=-1)
        distance = torch.sqrt(square_distance + 1e-9)  # Add epsilon to avoid division by zero

        outside_potential = -1 / distance
        inside_potential = square_distance / (2 * radius ** 3) - 3 / (2 * radius)

        is_outside = distance > radius
        potential = torch.where(is_outside, outside_potential, inside_potential)

        # Normalize the potential
        normalized_potential = potential * 2 * radius / 3
        return normalized_potential

    def __len__(self):
        """Return the total number of samples."""
        return len(self.points)

    def __getitem__(self, idx):
        """
        Retrieve a single sample from the dataset.
        
        Args:
            idx (int): Index of the sample to retrieve.
            
        Returns:
            point (torch.Tensor): Coordinates of the point, shape (dim,).
            potential (torch.Tensor): Gravitational potential at the point, shape (1,).
        """
        x, phi_true, grad_true, lapl_true = self.points[idx], self.potentials[idx], self.gradients[idx], self.laplacian[idx]
        x = x.detach() if x.requires_grad else x
        phi_true = phi_true.detach() if phi_true.requires_grad else phi_true
        grad_true = grad_true.detach() if grad_true.requires_grad else grad_true
        lapl_true = lapl_true.detach() if lapl_true.requires_grad else lapl_true

        # Add noise to the features or labels
        # x += torch.randn_like(x) * self.noise_level
        # phi_true += torch.randn_like(phi_true) * self.noise_level
        # grad_true += torch.randn_like(grad_true) * self.noise_level
        # lapl_true += torch.randn_like(lapl_true) * self.noise_level

        return x, phi_true, grad_true, lapl_true
