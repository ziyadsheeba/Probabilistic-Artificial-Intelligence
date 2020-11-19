import numpy as np
from scipy.optimize import fmin_l_bfgs_b

domain = np.array([[0, 5]])

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
import math
import random
from statistics import NormalDist

""" Solution """


class BO_algo():
    def __init__(self):
        """Initializes the algorithm with a parameter configuration. """

        self.speed_min = 1.2
        self.logspeed_min = math.log(self.speed_min)

        #define noise parameters
        self.var_f = 0.15**2
        self.var_v = 0.0001**2

        #initialize GPs
        kernel_var_f = 0.5
        kernel_var_v = math.sqrt(2)

        self.kernel_f = Matern(length_scale=0.5,length_scale_bounds="fixed", nu=2.5) #+ WhiteKernel(noise_level=self.var_f, noise_level_bounds="fixed") 
        self.kernel_v = Matern(length_scale=0.5,length_scale_bounds="fixed", nu=2.5) #+ WhiteKernel(noise_level=self.std_v) # + ConstantKernel(constant_value=1.5, constant_value_bounds="fixed")
        
        self.prior_mean_v  = 1.5 

        self.gp_f = GaussianProcessRegressor(kernel=self.kernel_f, alpha=kernel_var_f, n_restarts_optimizer=1) 
        self.gp_v = GaussianProcessRegressor(kernel=self.kernel_v, alpha=kernel_var_v, n_restarts_optimizer=1) #, normalize_y=True)

        # initialize  UCB Aquisition function parameters
        #self.sqrt_beta_f = math.sqrt(2*math.log(5*(self.t**2)*math.pi**2/(6))/5)
        #self.sqrt_beta_v = math.sqrt(2*math.log(5*(self.t**2)*math.pi**2/(6))/5)

        #define iteration parameter
        self.t = 1

        #define data array, initialize with 0 
        self.theta =  np.array([])
        self.y_f = np.array([])
        self.y_v = np.array([]) 

        #store best f, v values (needed for expected improvement) 
        self.best_f = -10000
        self.best_v = 1.5         
        

    def next_recommendation(self):
        """
        Recommend the next input to sample.

        Returns
        -------
        recommendation: np.ndarray
            1 x domain.shape[0] array containing the next point to evaluate
        """
        
        # In implementing this function, you may use optimize_acquisition_function() defined below.
        # At beginning draw random theta
        if(self.theta.shape[0]==0):
            #print("random draw")
            x_next = np.asarray([[random.uniform(0,5)]])
            self.t = 1
            #print(x_next)
        else :
            self.t = self.t + 1
            x_next = self.optimize_acquisition_function()

        return x_next


    def optimize_acquisition_function(self):
        """
        Optimizes the acquisition function.

        Returns
        -------
        x_opt: np.ndarray
            1 x domain.shape[0] array containing the point that maximize the acquisition function.
        """

        def objective(x):
            return -self.acquisition_function(x)

        f_values = []
        x_values = []

        # Restarts the optimization 20 times and pick best solution
        for _ in range(20):
            x0 = domain[:, 0] + (domain[:, 1] - domain[:, 0]) * \
                 np.random.rand(domain.shape[0])
            result = fmin_l_bfgs_b(objective, x0=x0, bounds=domain,
                                   approx_grad=True)
            x_values.append(np.clip(result[0], *domain[0]))
            f_values.append(-result[1])

        ind = np.argmax(f_values)
        return np.atleast_2d(x_values[ind])

    def acquisition_function(self, x):
        """
        Compute the acquisition function.

        Parameters
        ----------
        x: np.ndarray
            x in domain of f

        Returns
        ------
        af_value: float
            Value of the acquisition function at x
        """
        # Compute marginal likelihood of  GP
        x_np = np.reshape(x, (-1,1))
        mu_f, sigma_f = self.gp_f.predict(x_np, return_std=True)
        mu_v, sigma_v = self.gp_v.predict(x_np, return_std=True)            

        #Compute probabilistic constraint 
        constraint = 1 - NormalDist(mu=mu_v, sigma=sigma_v).cdf(self.speed_min - self.prior_mean_v)#*(1+self.var_v)

        # UCB Aquisition function - TODO try again with the constraint violation limit
        #self.sqrt_beta_f = math.sqrt(2*math.log(5*(self.t**2)*(math.pi**2)/6)/5)
        #self.sqrt_beta_v = math.sqrt(2*math.log(5*(self.t**2)*(math.pi**2)/6)/5)     
        #self.sqrt_beta_f = 1.0
        #self.sqrt_beta_v = 1.0  
        #af_value = (mu_f + self.sqrt_beta_f*sigma_f)*constraint   

        #Expected Improvement Aquisition function
        xi = 0.01 # hyperparameter for exploration (the larger the more explore, less exploit)
        Z_f = (mu_f - self.best_f - xi)/(sigma_f)
        Z_v = (mu_v - self.best_v - xi)/(sigma_v)        

        # According to Gelbart et al, 2014 if constraint is violated,  we only need to optimize constraint distribution
        # Violation of the constraint occurs, when the probability that v is below v_min is smaller than
        # a predefined threshold hyperparameter
        if(constraint > 0.6 and sigma_f!=0):
            ei_f = (mu_f - self.best_f - xi)*NormalDist().cdf(Z_f) + sigma_f*NormalDist().pdf(Z_f)
            ei_v = (mu_v - self.best_v - xi)*NormalDist().cdf(Z_v) + sigma_v*NormalDist().pdf(Z_v)
            af_value = constraint*ei_f
        else :
            af_value = np.asarray([constraint])
 
        return af_value       

     


    def add_data_point(self, x, f, v):
        """
        Add data points to the model.

        Parameters
        ----------
        x: np.ndarray
            Hyperparameters
        f: np.ndarray
            Model accuracy
        v: np.ndarray
            Model training speed
        """

        self.theta = np.append(self.theta, x[0])
        self.theta = np.reshape(self.theta, (-1,1))
        self.y_f = np.append(self.y_f,f)
        self.y_v = np.append(self.y_v, v)

        #if(self.theta.shape[0]>1):
        self.gp_v.fit(self.theta, self.y_v - self.prior_mean_v)
        self.gp_f.fit(self.theta, self.y_f)              

        # Compute best value so far seen
        self.best_f = self.y_f.max()
        self.best_v = self.y_v.max()

        #print(self.theta)
        #print(self.y_f)
        #print(self.y_v)
  

    def get_solution(self):
        """
        Return x_opt that is believed to be the maximizer of f.

        Returns
        -------
        solution: np.ndarray
            1 x domain.shape[0] array containing the optimal solution of the problem
        """
         # Select x s.t. speed constraint not violated
        lim_sel = self.y_v > self.speed_min
       
        if(np.count_nonzero(lim_sel)>0):
            idx_sel = self.y_f[lim_sel].argmax()
            theta_temp = self.theta[lim_sel]
            x_opt = theta_temp[idx_sel]
        else :
            self.gp_f.fit(self.theta, self.y_f)
            self.gp_v.fit(self.theta, self.y_v - self.prior_mean_v)
            x_opt = self.optimize_acquisition_function()      
        return x_opt



""" Toy problem to check code works as expected """

def check_in_domain(x):
    """Validate input"""
    x = np.atleast_2d(x)
    return np.all(x >= domain[None, :, 0]) and np.all(x <= domain[None, :, 1])


def f(x):
    """Dummy objective"""
    mid_point = domain[:, 0] + 0.5 * (domain[:, 1] - domain[:, 0])
    return - np.linalg.norm(x - mid_point, 2)  # -(x - 2.5)^2


def v(x):
    """Dummy speed"""
    return 2.0


def main():
    # Init problem
    agent = BO_algo()

    # Loop until budget is exhausted
    for j in range(20):
        # Get next recommendation
        x = agent.next_recommendation()

        # Check for valid shape
        assert x.shape == (1, domain.shape[0]), \
            f"The function next recommendation must return a numpy array of " \
            f"shape (1, {domain.shape[0]})"

        # Obtain objective and constraint observation
        obj_val = f(x)
        cost_val = v(x)
        agent.add_data_point(x, obj_val, cost_val)

    # Validate solution
    solution = np.atleast_2d(agent.get_solution())
    assert solution.shape == (1, domain.shape[0]), \
        f"The function get solution must return a numpy array of shape (" \
        f"1, {domain.shape[0]})"
    assert check_in_domain(solution), \
        f'The function get solution must return a point within the ' \
        f'domain, {solution} returned instead'

    # Compute regret
    if v(solution) < 1.2:
        regret = 1
    else:
        regret = (0 - f(solution))

    print(f'Optimal value: 0\nProposed solution {solution}\nSolution value '
          f'{f(solution)}\nRegret{regret}')


if __name__ == "__main__":
    main()