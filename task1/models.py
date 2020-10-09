import numpy as np
import math
import time

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, Matern, RationalQuadratic, ExpSineSquared
from sklearn.kernel_approximation import Nystroem, RBFSampler



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




