import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

import torch

class ThreeBodySystem():
    def __init__(self, r1,r2,r3,v1,v2,v3, t_span, mass, G):
        
        self.r1 = r1
        self.r2 = r2
        self.r3 = r3

        self.v1 = v1
        self.v2 = v2
        self.v3 = v3

        self.t_span = t_span

        self.mass = mass
        self.G = G

        self.initial_conditions = np.concatenate([r1, r2, r3, v1, v2, v3])


    def equation(self, r, rdot,rdotdot):
        r1, r2, r3 = r[:,0:2], r[:,2:4], r[:,4:6]
        rdotdot1,rdotdot2,rdotdot3 = rdotdot[:,0:2], rdotdot[:,2:4], rdotdot[:,4:6]
        # v1, v2, v3 = rdot[6:8], rdot[8:10], rdot[10:12]
        m1,m2,m3 = self.mass
        # Calculate distances
        def distance(rA, rB):
            try :
                return np.linalg.norm(rA - rB)
            
            except Exception:
                return torch.norm(rA - rB)

        r12, r13, r23 = distance(r1, r2), distance(r1, r3), distance(r2, r3)

        # Compute accelerations due to gravity
        a1 = self.G * m2 * (r2 - r1) / r12**3 + self.G * m3 * (r3 - r1) / r13**3
        a2 = self.G * m1 * (r1 - r2) / r12**3 + self.G * m3 * (r3 - r2) / r23**3
        a3 = self.G * m1 * (r1 - r3) / r13**3 + self.G * m2 * (r2 - r3) / r23**3
        return (m1*rdotdot1 - a1)**2 + (m2* rdotdot2 - a2)**2 + (m3 * rdotdot3 - a3)**2
    
    def three_body_equations(self, t, y):
        # Unpack positions and velocities
        r1, r2, r3 = y[0:2], y[2:4], y[4:6]
        v1, v2, v3 = y[6:8], y[8:10], y[10:12]
        m1,m2,m3 = self.mass
        # Calculate distances
        def distance(rA, rB):
            try:
                return np.linalg.norm(rA - rB)

            except Exception:
                return torch.norm(rA - rB)

        r12, r13, r23 = distance(r1, r2), distance(r1, r3), distance(r2, r3)

        # Compute accelerations due to gravity
        a1 = self.G * m2 * (r2 - r1) / r12**3 + self.G * m3 * (r3 - r1) / r13**3
        a2 = self.G * m1 * (r1 - r2) / r12**3 + self.G * m3 * (r3 - r2) / r23**3
        a3 = self.G * m1 * (r1 - r3) / r13**3 + self.G * m2 * (r2 - r3) / r23**3

        # Return derivative [dr1/dt, dr2/dt, dr3/dt, dv1/dt, dv2/dt, dv3/dt]
        return np.concatenate([v1, v2, v3, a1, a2, a3])
    
    def get_solution(self, t):
        
        solution = solve_ivp(self.three_body_equations, self.t_span, self.initial_conditions, t_eval=t, rtol=1e-9)
        return solution.y[:6], solution.y[6:12]

    def generate_noisy_datapoints(self, t_min, t_max, num_points, std_x):
        
        t_sample = np.linspace(t_min,t_max,num_points)

        true_sol, true_vel = self.get_solution(t_sample)
        noisy_sol = true_sol + np.random.normal(0,std_x,true_sol.shape)
        datadict = {
            "t":t_sample,
            "x":noisy_sol,
            "xdot": true_vel
        }

        return datadict


import numpy as np
from scipy.integrate import solve_ivp

class TwoBodySystem:
    def __init__(self, r1, r2, v1, v2, t_span, mass, G):
        self.r1 = r1
        self.r2 = r2
        self.v1 = v1
        self.v2 = v2
        self.t_span = t_span
        self.mass = mass
        self.G = G

        # Initial conditions for two bodies
        self.initial_conditions = np.concatenate([r1, r2, v1, v2])
        self.initial_momentum = mass[0] * v1 + mass[1] * v2

    def equation(self, r, rdot, rdotdot):
        r1, r2 = r[:, 0:2], r[:, 2:4]
        rdotdot1, rdotdot2 = rdotdot[:, 0:2], rdotdot[:, 2:4]
        m1, m2 = self.mass
        batch_size = r.shape[0]
        
        # Create time weights that increase with the sequence
        alpha = 1.0  # Adjust this value to change the weight growth rate
        time_weights = torch.pow(alpha, torch.arange(batch_size, device=r.device))
        
        time_weights = time_weights / time_weights.mean()  # Normalize weights
        
        # Calculate the distance between the two bodies
        def distance(rA, rB):
            return torch.norm(rA - rB, dim=1)
        
        r12 = distance(r1, r2).unsqueeze(1)
        
        # Compute accelerations due to gravity
        a1 = self.G * m2 * (r2 - r1) / r12**3
        a2 = self.G * m1 * (r1 - r2) / r12**3
        
        
        # Calculate squared errors and reduce along the coordinate dimension
        error1 = (rdotdot1 - a1)**2
        error2 = (rdotdot2 - a2)**2
        # Apply time weights to the reduced errors
        weighted_error1 = error1 * time_weights.unsqueeze(1)
        weighted_error2 = error2 * time_weights.unsqueeze(1)

        # Take mean of weighted errors
        return (torch.mean(weighted_error1) + torch.mean(weighted_error2).unsqueeze(0)).squeeze(0), a1, a2 #
    
    def momentum(self, rdot):
        rdot1, rdot2 = rdot[:, 0:2], rdot[:, 2:4]
        m1, m2 = self.mass

        return m1 * rdot1 + m2 * rdot2

    def two_body_equations(self, t, y):
        # Unpack positions and velocities
        r1, r2 = y[0:2], y[2:4]
        v1, v2 = y[4:6], y[6:8]

        m1, m2 = self.mass

        # Calculate the distance between the two bodies
        def distance(rA, rB):
            try:
                return np.linalg.norm(rA - rB, axis = 0)
            except Exception:
                return torch.norm(rA - rB, dim=0)
        r12 = distance(r1, r2)
        # Compute accelerations due to gravity
        a1 = self.G * m2 * (r2 - r1) / r12**3
        a2 = self.G * m1 * (r1 - r2) / r12**3

        # Return derivative [dr1/dt, dr2/dt, dv1/dt, dv2/dt]
        return np.concatenate([v1, v2, a1, a2])

    def get_solution(self, t):
        solution = solve_ivp(self.two_body_equations, self.t_span, self.initial_conditions, t_eval=t, rtol=1e-9)
        return solution.y[:4], solution.y[4:8]

    def generate_noisy_datapoints(self, t_min, t_max, num_points, std_x):
        t_sample = np.linspace(t_min, t_max, num_points)
        true_sol, true_vel = self.get_solution(t_sample)
        noisy_sol = true_sol + np.random.normal(0, std_x, true_sol.shape)
        datadict = {
            "t": t_sample,
            "x": noisy_sol,
            "xdot": true_vel
        }
        return datadict
