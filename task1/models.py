import numpy as np
import math
import time
import statistics

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn import linear_model
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, Matern, RationalQuadratic, ExpSineSquared, WhiteKernel, Sum
from sklearn.kernel_approximation import Nystroem, RBFSampler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import log_loss


import torch
import gpytorch






#from skopt import gp_minimize


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

def cost_function(true, predicted):
    """
        true: true values in 1D numpy array
        predicted: predicted values in 1D numpy array

        return: float
    """
    cost = (true - predicted)**2

    # true above threshold (case 1)
    mask = true > THRESHOLD
    mask_w1 = np.logical_and(predicted>=true,mask)
    mask_w2 = np.logical_and(np.logical_and(predicted<true,predicted >=THRESHOLD),mask)
    mask_w3 = np.logical_and(predicted<THRESHOLD,mask)

    cost[mask_w1] = cost[mask_w1]*W1
    cost[mask_w2] = cost[mask_w2]*W2
    cost[mask_w3] = cost[mask_w3]*W3

    # true value below threshold (case 2)
    mask = true <= THRESHOLD
    mask_w1 = np.logical_and(predicted>true,mask)
    mask_w2 = np.logical_and(predicted<=true,mask)

    cost[mask_w1] = cost[mask_w1]*W1
    cost[mask_w2] = cost[mask_w2]*W2

    reward = W4*np.logical_and(predicted < THRESHOLD,true<THRESHOLD)
    if reward is None:
        reward = 0
    return np.mean(cost) - np.mean(reward)



class GP_SK_1():

    def __init__(self):
        self.estimator = None
        ## DENSE DATA REGION ESTIMATOR
        # Hyperparameter for grid search 
        self.hyper_param_grid = [
            {'nystroem__kernel': [o*Sum(Matern(length_scale=lm,nu=n), RBF())
                             for lm in np.logspace(-1, 0.1, 50)
                             for n in np.logspace(-2, 0.4, 50)
                             for o in np.logspace(-2,1,20)] ,}
        ]    

        # Kernel approximation for dense data region        
        self.feature_map = Nystroem(n_components=1000, random_state=2)        
        #Pipeline
        self.pipeline = Pipeline([('nystroem', self.feature_map),
                                 ('brr', linear_model.BayesianRidge(compute_score=True))])
        
        self.pipeline.set_params(nystroem__gamma = 100)


        #dense_estimator = RandomizedSearchCV(self.pipeline, self.hyper_param_grid, random_state = 3, cv=2, return_train_score=True, verbose=2, refit=True, n_jobs=2, n_iter=2)
        #dense_estimator = GridSearchCV(self.pipeline, self.hyper_param_grid, cv=5, return_train_score=True, verbose=1, refit=True, n_jobs=2)
        dense_estimator = self.pipeline

        ## SPARSE DATA REGION ESTIMATOR
        self.kernel =  Matern() + WhiteKernel() + RBF()
        sparse_estimator = GaussianProcessRegressor(kernel=self.kernel, normalize_y=True, n_restarts_optimizer = 4)
        

        self.estimator = [dense_estimator, sparse_estimator]



    def predict(self, test_x, return_std=False):
        n = test_x.shape[0]
        y_mean = np.zeros((n,))
        y_std = np.zeros((n,))

                

        # Dense region
        y_1, y_std_1 = self.estimator[0].predict(test_x, return_std = True)
        print(np.max(y_std_1))
        y_1 = y_1 + 1.3*np.exp(-(y_1 - THRESHOLD)**2)*y_std_1
        # Sparse region
        y_0, y_std_0 = self.estimator[1].predict(test_x, return_std = True)
        y_0 = y_0 +  1.3*np.exp(-(y_0 - THRESHOLD)**2)*y_std_0

        sparse_domain = test_x[:,0] > -0.5
        y = sparse_domain*y_0 + np.invert(sparse_domain)*y_1
        y_std =  sparse_domain*y_std_0 + np.invert(sparse_domain)*y_std_1    
                
        # Bayesian Decision theory with asymmetric cost, see IML slides
        """
        c1 = 100 #TODO change depending on cost function        
        c2 = 1
        c  = c1/(c1+c2)
        n = test_x.shape[0]
        for i in range(n) :
            y[i] = y[i] + y_std[i]*statistics.NormalDist(y[i], y_std[i]).inv_cdf(c)        

        """
        if(return_std) :
            return y, y_std
        else :
            return y    


    def fit_model(self, train_x, train_y):

        print("Starting model fitting...")
        print("Dimensions of training data X : {},  y : {}".format(train_x.shape, train_y.shape))  

        print("Fitting sparse region estimator")
        self.estimator[1].fit( train_x[train_x[:,0] > -0.5], train_y[train_x[:,0] > -0.5])
        print("Fitting dense region estimator")
        self.estimator[0].fit( train_x[train_x[:,0] <= -0.5], train_y[train_x[:,0] <= -0.5])

        """       
        # Subsample
        samples_idx = np.random.choice(dense_region_idx, size = (100, ) , replace = True)
        self.estimator[1].fit(np.concatenate((train_x[sparse_region_idx,:], train_x[samples_idx,:])),
                                  np.concatenate((train_y[sparse_region_idx],train_y[samples_idx])))

        #print(self.estimator[0].best_estimator_.get_params(['kernel']))
        #self.estimator[1].set_params(['kernel' : self.estimator[0].best_estimator_.get_params(['nystroem__kernel']))
        #self.estimator[1].fit(train_x[sparse_region_idx], train_y[sparse_region_idx])
        """

       


class GP_Torch_1(gpytorch.models.ExactGP):

    def __init__(self, train_x, train_y, likelihood):
        super(GP_Torch_1, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean() # TODO change if necessary
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

        test_x_tensor = torch.from_numpy(test_x).float()

        
        # Make predictions by feeding model through likelihood
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            y = self.likelihood_(self.model_(test_x_tensor))
        return y.mean #? TODO what to return

    def fit_model(self, train_x, train_y):
        train_x = train_x.astype('d')
        train_y = train_y.astype('d')

        train_x_tensor = torch.from_numpy(train_x).float()
        train_y_tensor = torch.from_numpy(train_y).float()

  
        #train_x_tensor = torch.tensor(train_x, dtype=torch.double)
        #train_y_tensor = torch.tensor(train_y, dtype=torch.double)

        training_iter = 50  
        self.likelihood_ = gpytorch.likelihoods.GaussianLikelihood()
        self.model_ = GP_Torch_1(train_x_tensor, train_y_tensor , self.likelihood_)

  

        # --------- taken from gpytorch website ----
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
            #print(train_y_tensor.dtype)
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
        y = np.sin(test_x[:,0]+3*test_x[:,1])
        #y = np.ones(test_x.shape[0]) * THRESHOLD - 0.00001
        return y

    def fit_model(self, train_x, train_y):
        """
             TODO: enter your code here
        """
        pass




def analyze_prediction(M, x_test, y_test) :
    y_predicted = M.predict(x_test)
    above_threshold_index = np.where(np.any(y_test>THRESHOLD))
    y_above_threshold = y_test[above_threshold_index]
    y_predicted_sameidx = y_predicted[above_threshold_index]
    num_wrongly_predicted = np.count_nonzero(y_predicted_sameidx < y_above_threshold)
    print('Number of wrongly underpredicted over THRESHOLD labels  : {} '.format(num_wrongly_predicted))

