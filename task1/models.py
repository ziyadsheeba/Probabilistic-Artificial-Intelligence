import numpy as np
import math
import time

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, Matern, RationalQuadratic, ExpSineSquared
from sklearn.kernel_approximation import Nystroem, RBFSampler

import torch
import gpytorch



## Constants for Cost function
THRESHOLD = 0.5
W1 = 1
W2 = 20
W3 = 100
W4 = 0.04

"""
Fill in the methods of the Model. Please do not change the given methods for the checker script to work.
You can add new methods, and make changes. The checker script performs:


    M = Model()
    M.fit_model(train_x,train_y)
    prediction = M.predict(test_x)

It uses predictions to compare to the ground truth using the cost_function above.
"""


class GP_SK_1():

    def __init__(self):
        #TODO dont fixate kernel here
        self.kernel_ = RBF(10)
        self.training_score_ = math.inf # coefficient of determination R^2, best value 0 
        self.estimator = GaussianProcessRegressor(kernel=self.kernel_,random_state=0, n_restarts_optimizer=10)

    def predict(self, test_x):
        y, y_cov = self.estimator.predict(test_x, return_cov=True)
        #print("Covariance of prediction : \n {}".format(y_cov))
        return y

    def fit_model(self, train_x, train_y):
        print("Starting model fitting...")
        print("Dimensions of training data X : {},  y : {}".format(train_x.shape, train_y.shape))
        self.estimator.fit(train_x, train_y)
        start_time = time.time()
        self.training_score_ = self.estimator.score(train_x, train_y)
        print("Time for fitting the model: {:3f}".format(time.time() - start_time))
        print("Learned kernel: \n {}".format(self.estimator.kernel_))
        print("Score for training data : {}".format(self.training_score_))
        # TODO use crossvalidation on predictive performance max marginal likelihood of the data (sci kit log_marginal_likelihood(theta) )see


class GP_Torch_1(gpytorch.models.ExactGP):

    def __init__(self, train_x, train_y, likelihood):
        super(GP_Torch_1, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean() # TODO change if necessary
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
     
    def forward(self, x): 
        """ Returns MultivariateNormal with prior mean and covariance conditioned at x """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)




class Wrapper_Model_Torch():
    """ Used in order to preserve model class and method structure, s.t. checker still works """
    def __init__(self):
        self.likelihood_ = None
        self.model_ = None

    def predict(self, test_x):

        # Get into evaluation (predictive posterior) mode
        self.model_.eval()
        self.likelihood_.eval()

        # Test points are regularly spaced along [0,1]
        # Make predictions by feeding model through likelihood
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            y = self.likelihood_(self.model_(test_x))
        return y

    def fit_model(self, train_x, train_y):
        train_x_tensor = torch.tensor(train_x, dtype=torch.double)
        train_y_tensor = torch.tensor(train_y, dtype=torch.double)

        training_iter = 2

  


        self.likelihood_ = gpytorch.likelihoods.GaussianLikelihood()
        self.model_ = GP_Torch_1(train_x_tensor, train_y_tensor , self.likelihood_)

  

        # --------- taken form gpytorch website ----
        # Find optimal model hyperparameters
        self.model_.train()
        self.likelihood_.train()

        # Use the adam optimizer
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood_, self.model_)

        for i in range(training_iter):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = self.model_(train_x_tensor)
            # Calc loss and backprop gradients
            loss = -mll(output, train_y_tensor)
            loss.backward()
            print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
                i + 1, training_iter, loss.item(),
                self.model_.covar_module.base_kernel.lengthscale.item(),
                self.model_.likelihood.noise.item()
            ))
            optimizer.step()





class DefaultModel():

    def __init__(self):
        """
            TODO: enter your code here
        """
        pass

    def predict(self, test_x):
        """
            TODO: enter your code here
        """
        ## dummy code below
        y = np.ones(test_x.shape[0]) * THRESHOLD - 0.00001
        return y

    def fit_model(self, train_x, train_y):
        """
             TODO: enter your code here
        """
        pass




