import numpy as np
import matplotlib.pyplot as plt

class DampenSpringSystem():
    def __init__(self, x0, v0, mass, k, mu):
        
        self.x0 = x0
        self.v0 = v0
        self.mass = mass
        self.k = k
        self.mu = mu

        self.omega = np.sqrt(k/mass)
        self.ksi = mu/(2*np.sqrt(mass*k))


    def equation(self, x, xdot,xdotdot):
        return self.omega**2 * x + 2 * self.ksi * self.omega * xdot + xdotdot
    
    def get_solution(self, t):

        if self.ksi<1:
            omega_d = self.omega*np.sqrt(1-self.ksi**2)
            c1 = self.x0
            c2 = (self.v0 + self.ksi * self.omega * self.x0)/omega_d
            x = np.exp(-self.ksi * self.omega * t) * (c1 * np.cos(omega_d * t) + c2*np.sin(omega_d*t))

        elif np.isclose(self.ksi, 1):
            x = (self.x0 + (self.v0 + self.omega + self.x0)*t)*np.exp(-self.omega * t)
        
        else:
            lambda_1 = -self.ksi*self.omega + self.omega * np.sqrt(self.ksi**2 - 1)
            lambda_2 = -self.ksi*self.omega - self.omega * np.sqrt(self.ksi**2 - 1)

            c2 = (self.v0 - lambda_1 * self.x0)/(lambda_2 - lambda_1)
            c1 = self.x0 - c2

            x = c1*np.exp(lambda_1 * t) + c2 * np.exp(lambda_2 * t)
        
        return x
    
    def get_first_derivatives(self,t):
        if self.ksi<1:
            omega_d = self.omega*np.sqrt(1-self.ksi**2)

            c1 = self.x0
            c2 = (self.v0 + self.ksi * self.omega * self.x0)/omega_d

            xdot = np.exp(-self.ksi * self.omega * t) * ((c1 * self.omega * self.ksi+c2*omega_d) * np.cos(omega_d * t) - (c2*self.ksi*self.omega + c1*omega_d)*np.sin(omega_d*t))

        elif np.isclose(self.ksi, 1):
            xdot = (self.v0*(1- self.omega * t) - t*self.x0*self.omega**2)*np.exp(-self.omega * t)
        
        else:
            lambda_1 = -self.ksi*self.omega + self.omega * np.sqrt(self.ksi**2 - 1)
            lambda_2 = -self.ksi*self.omega - self.omega * np.sqrt(self.ksi**2 - 1)

            c2 = (self.v0 - lambda_1 * self.x0)/(lambda_2 - lambda_1)
            c1 = self.x0 - c2

            xdot = c1*lambda_1 * np.exp(lambda_1 * t) + c2 * lambda_2 * np.exp(lambda_2 * t)
        
        return xdot
    
    def get_second_derivatives(self,t):
        return -(1/self.mass) * (self.k*self.get_solution(t) + self.mu*self.get_first_derivatives(t))


    def generate_noisy_datapoints(self, t_min, t_max, num_points, std_x, true_derivatives = True):
        
        t_sample = np.linspace(t_min,t_max,num_points)

        true_sol = self.get_solution(t_sample)
        noisy_sol = true_sol + np.random.normal(0,std_x,true_sol.shape)
        if not true_derivatives:
            print(noisy_sol.shape)
            first_der = np.gradient(noisy_sol, t_sample, edge_order=2)            
            second_der = np.gradient(first_der, t_sample, edge_order=2)
        
        else:
            first_der = self.get_first_derivatives(t_sample)
            second_der = self.get_second_derivatives(t_sample)
        
        datadict = {
            "t":t_sample,
            "x":noisy_sol,
            "xdot":first_der,
            "xdotdot":second_der
        }

        return datadict



if __name__ == "__main__" :

    mass = 1
    mu = 1
    k = [10]

    fig, axs = plt.subplots(2,1)
    for i,kk in enumerate(k): 
        system = DampenSpringSystem(1,0,mass, kk, mu)
        t = np.arange(0,10,0.0000001)
        x = system.get_solution(t)

        data = system.generate_noisy_datapoints(0,10,100,0.05,True)

        # axs[i].scatter(t,x)
        # axs[i].plot(t,x)

        axs[i].scatter(data["t"][1:-1],data["x"][1:-1], label = "Noisy", color = "blue")
        axs[i].scatter(data["t"][1:-1],data["xdot"][1:-1], label = "Noisy", color = "red")
        axs[i].plot(data["t"][1:-1],data["xdotdot"][1:-1], label = "Noisy", color = "orange")
        # axs[i].plot(t,x)
        axs[i].set_title(f"ksi: {system.ksi}")


    plt.legend()
    plt.show()


