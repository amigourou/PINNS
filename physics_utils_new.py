import numpy as np
import torch
from scipy.integrate import solve_ivp

class BodySystem:
    def __init__(self, positions, velocities, t_span, masses, G):
        self.positions = positions
        self.velocities = velocities
        self.t_span = t_span
        self.masses = masses
        self.G = G
        self.num_bodies = len(positions)
        
        # Flatten initial conditions for solve_ivp
        self.initial_conditions = np.concatenate(positions + velocities)

    
    def calculate_distances(self, r):
        """Calculate pairwise distances between bodies."""
        distances = {}
        for i in range(self.num_bodies):
            for j in range(i + 1, self.num_bodies):
                diff = r[i] - r[j]
                
                # Check if `r` is a numpy array or tensor and use the appropriate norm function
                if isinstance(diff, np.ndarray):
                    # Use axis=1 only if diff is 2D
                    distances[(i, j)] = np.linalg.norm(diff, axis=0) if diff.ndim > 1 else np.linalg.norm(diff)
                else:
                    # Use dim=1 only if diff is 2D for torch tensors
                    distances[(i, j)] = torch.norm(diff, dim=0) if diff.dim() > 1 else torch.norm(diff)
                
        return distances


    
    def gravitational_accelerations(self, r, distances):
        """Calculate gravitational accelerations for each body due to others."""
        if isinstance(r[0], np.ndarray):
            accelerations = [np.zeros_like(r[i]) for i in range(self.num_bodies)]
        else:
            accelerations = [torch.zeros_like(r[i]) for i in range(self.num_bodies)]
        
        for (i, j), dist in distances.items():

            # Compute the unit vector pointing from body i to body j
            direction = (r[j] - r[i]) / dist  # This is the normalized direction
            # Gravitational force magnitude
            force_magnitude = self.G * self.masses[i] * self.masses[j] / dist**2
            
            # Gravitational acceleration: F = m * a => a = F / m
            acc_i = force_magnitude * direction / self.masses[i]  # Acceleration on body i
            acc_j = -force_magnitude * direction / self.masses[j]  # Acceleration on body j (action = -reaction)
            
            # Update accelerations for both bodies
            accelerations[i] += acc_i
            accelerations[j] += acc_j
        return accelerations


    def equation_error(self, r, rdot, rdotdot):
        """Calculate weighted squared error based on gravitational forces."""
        r_list = [r[i*2:(i+1)*2] for i in range(self.num_bodies)]
        v_list = [rdot[i*2:(i+1)*2] for i in range(self.num_bodies)]
        a_list = [rdotdot[i*2:(i+1)*2] for i in range(self.num_bodies)]
        distances = self.calculate_distances(r_list)
        accelerations = self.gravitational_accelerations(r_list, distances)
        
        errors = torch.tensor(0.0, device=r.device)
        for i in range(self.num_bodies):
            mass = self.masses[i]
            
            errors += torch.mean((a_list[i] - accelerations[i])**2)
        return errors

    def body_equations(self, t, y):
        """Compute derivatives for solve_ivp based on the number of bodies."""
        # Determine if we're dealing with NumPy or PyTorch tensors
        is_numpy = isinstance(y, np.ndarray)
        array_type = np if is_numpy else torch

        # Reshape y into position (r) and velocity (v) arrays
        r = [y[i*2:(i+1)*2] for i in range(self.num_bodies)]
        v = [y[(self.num_bodies + i)*2:(self.num_bodies + i + 1)*2] for i in range(self.num_bodies)]
        # Calculate distances and accelerations with type-agnostic methods
        distances = self.calculate_distances(r)
        accelerations = self.gravitational_accelerations(r, distances)
        
        # Initialize list for derivatives
        derivatives = []
        
        # For each body, append the velocity (dr/dt) and acceleration (dv/dt)
        for i in range(self.num_bodies):
            # Ensure we keep the correct shape for each velocity and acceleration
            derivatives.append(array_type.atleast_1d(v[i]))  # dr/dt = v
            
        
        for i in range(self.num_bodies):
            derivatives.append(array_type.atleast_1d(accelerations[i]))  # dv/dt = acceleration

        # Convert the list of derivatives into a single array, preserving the correct dimensions
        return array_type.concatenate(derivatives) if is_numpy else array_type.cat(derivatives)


    def get_solution(self, times = None):
        if times is None:
            solution = solve_ivp(self.body_equations, self.t_span, self.initial_conditions, rtol=1e-12, atol=1e-12, method='DOP853')
            return solution.y[:self.num_bodies*2], solution.y[self.num_bodies*2:], solution.t
        else:
            solution = solve_ivp(self.body_equations, self.t_span, self.initial_conditions, t_eval=times, rtol=1e-9)
            return solution.y[:self.num_bodies*2], solution.y[self.num_bodies*2:]

    def generate_noisy_datapoints(self, t_min, t_max, num_points, std_x):
        t_sample = np.linspace(t_min, t_max, num_points)
        true_sol, true_vel = self.get_solution(t_sample)
        noisy_sol = true_sol + np.random.normal(0, std_x, true_sol.shape)
        return {
            "t": t_sample,
            "x": noisy_sol,
            "xdot": true_vel
        }


class OneBodySystem(BodySystem):
    def __init__(self, r1, v1, t_span, mass, G):
        super().__init__([r1, np.array([0, 0])], [v1, np.array([0, -0.50])], t_span, [mass, 1], G)

class TwoBodySystem(BodySystem):
    def __init__(self, r1, r2, v1, v2, t_span, masses, G):
        super().__init__([r1, r2], [v1, v2], t_span, masses, G)

class ThreeBodySystem(BodySystem):
    def __init__(self, r1, r2, r3, v1, v2, v3, t_span, masses, G):
        super().__init__([r1, r2, r3], [v1, v2, v3], t_span, masses, G)
