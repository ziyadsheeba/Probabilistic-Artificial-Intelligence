import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from scipy.stats import norm
import matplotlib.pyplot as plt

# GP-related imports
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RBF, ExpSineSquared, RationalQuadratic, Matern
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.base import RegressorMixin
from sklearn.base import BaseEstimator

domain = np.array([[0, 5]])


""" Solution """


class BO_algo(BaseEstimator,RegressorMixin):
    def __init__(self):
        """Initializes the algorithm with a parameter configuration. """

        # TODO: enter your code here
        #####################
        # Objective GP
        kernel_f = Matern(length_scale=0.5, length_scale_bounds="fixed", nu=2.5) + \
                WhiteKernel(noise_level=np.power(0.15,2), noise_level_bounds="fixed")

        GP_f = GaussianProcessRegressor(kernel=kernel_f,
                                        n_restarts_optimizer = 1, 
                                        normalize_y=True,
                                        random_state=1)

        ####################
        # Speed GP
        # TODO: prescribe the 1.5 mean somehow.
        kernel_v = Matern(length_scale=0.5, length_scale_bounds="fixed", nu=2.5) + \
                WhiteKernel(noise_level=np.power(0.15,2), noise_level_bounds="fixed")

        GP_v = GaussianProcessRegressor(kernel=kernel_v,
                                        n_restarts_optimizer = 1, 
                                        normalize_y=False,
                                        random_state=1)
        self.model = [GP_f, GP_v]

        ####################
        # Training data placeholder
        self.train_data = np.empty((0,domain.shape[0]+2) dtype="object")

        # Constraint lower bound
        self.v_min = 1.2
        


    def next_recommendation(self):
        """
        Recommend the next input to sample.

        Returns
        -------
        recommendation: np.ndarray
            1 x domain.shape[0] array containing the next point to evaluate
        """

        # TODO: enter your code here
        # In implementing this function, you may use optimize_acquisition_function() defined below.
        return self.optimize_acquisition_function()[None, 0,:]


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

        # TODO: enter your code here

        # Find minimum objective over current sample points
        # TODO: should also take into account feasibility
        if (self.train_data.size == 0):
            target = -1000
        else:
            target = self.train_data[:,1].max()

        # Predict objective
        x = np.atleast_2d(x)
        y, y_std = self.model[0].predict(x, return_std=True)
        z = (y - target)/y_std

        # Predict constraint
        v, v_std = self.model[1].predict(x, return_std=True)
        z_v = (self.v_min - v)/v_std
        constraint_prob = 1 - norm.cdf(z_v, loc=0, scale=1)

        # Compute Expected Improvement
        EI = y_std * ( z * norm.cdf(z, loc=0, scale=1) + norm.pdf(z, loc=0, scale=1)) * constraint_prob
        return EI

        


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

        # TODO: enter your code here
        
        # Add new observation to the training data.
        self.train_data = np.append( self.train_data, np.array([[x[0,0], f, v]]), axis=0)

        # Refit the models
        # TODO: Maybe use some tricks here: 
        # e.g. the log regarding speed as in the paper
        # adding some constant to the training data of speed to ensure mean = 1.5
        if (self.train_data.shape[0] > 1): # for 1 training sample the fit crashes
            self.model[0].fit(self.train_data[:, 0].reshape(-1,1), self.train_data[:,1])
            self.model[1].fit(self.train_data[:, 0].reshape(-1,1), self.train_data[:,2])

    def get_solution(self):
        """
        Return x_opt that is believed to be the maximizer of f.

        Returns
        -------
        solution: np.ndarray
            1 x domain.shape[0] array containing the optimal solution of the problem
        """

        # TODO: enter your code here
        feasible = self.train_data[:,2] > self.v_min
        subset_idx = self.train_data[feasible,1].argmax()
        parent_idx = np.arange(self.train_data.shape[0])[feasible][subset_idx]
        return self.train_data[ parent_idx, 0:domain.shape[0] ]

def plot_model(gp, train_x, train_y, test_x, plot_points=True):
    y, y_std = gp.predict(test_x, return_std=True)
    lower, upper = y-y_std, y+y_std
    
    if plot_points:
        plt.plot(train_x, train_y, 'k*', label='Train Data')
        
    #test_y = regression_function(test_x, noise=0).detach()
    #plt.plot(test_x, test_y,'k-', label='Noise-free Function')
    
    plt.plot(test_x, y, 'b-', label='Mean Prediction')
    plt.fill_between(test_x.squeeze(), lower.squeeze(), upper.squeeze(), 
                     color='b', alpha=0.2, label='Predictive Distribution')
    
    plt.ylim([-5, 4])
    plt.legend(loc='upper left')

""" Toy problem to check code works as expected """

def check_in_domain(x):
    """Validate input"""
    x = np.atleast_2d(x)
    return np.all(x >= domain[None, :, 0]) and np.all(x <= domain[None, :, 1])


def f(x):
    """Dummy objective"""
    #mid_point = domain[:, 0] + 0.5 * (domain[:, 1] - domain[:, 0])
    #return - np.linalg.norm(x - mid_point, 2) + np.random.normal(loc=0.0, scale=0.15) # -(x - 2.5)^2
    a_4, a_3, a_2, a_1, a_0, of = -0.3, 0.7, 1.2, -1.4, 0.3, -3.7
    # -3.7, 0.3, -1.4, -1.2, 0.7, -0.3
    coef = np.array([-0.3, 0.7, 1.2, -1.4, 0.3])
    return float(np.polyval(coef, (-1.2*x+3.7)))

def v(x):
    """Dummy speed"""
    #return 2.0
    #return float( np.max( np.array([0, (x-2)*1.2]) ) ) + np.random.normal(loc=0.0, scale=0.07)
    return float( 1.2+0.5*np.sin(5*x) + np.random.normal(loc=0.0, scale=0.07) )


def main():
    # Init problem
    agent = BO_algo()

    # Loop until budget is exhausted
    for j in range(100):
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

    test_x = np.linspace(0, 5, 100)
    plot_model(agent.model[0], agent.train_data[:,0], agent.train_data[:,1], test_x.reshape(-1,1))
    plot_model(agent.model[1], agent.train_data[:,0], agent.train_data[:,2], test_x.reshape(-1,1))
    print(f'Optimal value: 0\nProposed solution {solution}\nSolution value '
          f'{f(solution)}\nRegret{regret}')


if __name__ == "__main__":
    main()