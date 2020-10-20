import numpy as np

# Scikit Learn
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.kernel_approximation import Nystroem
from sklearn.gaussian_process.kernels import RBF, Matern,RationalQuadratic, WhiteKernel
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

# Scipy 
from scipy.stats import norm

# Debugging
#import ipdb

# Plotting 
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D

## Constant for Cost function
THRESHOLD = 0.5
W1 = 1
W2 = 20
W3 = 100
W4 = 0.04




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

"""
Fill in the methods of the Model. Please do not change the given methods for the checker script to work.
You can add new methods, and make changes. The checker script performs:


    M = Model()
    M.fit_model(train_x,train_y)
    prediction = M.predict(test_x)


It uses predictions to compare to the ground truth using the cost_function above.
"""


class Model():

    '''
        The approach used to solve this problem will be the following:
            
            1) Subsample the data N_iteration times and get an estimate of the hyperparameters used 
               using the marginal likelihood method. The average will be weighted by the likelihood
               function.

            2) Choose the kernel that returns the higher log marginal likelihood on average.
            
            3) Using the estimated hyperparameter for the optimal kernel and compute a Nystroem approximation
               and return a new data matrix

            4) Use the new data matrix to fit a Bayesian Ridge Regression model

    '''

    def __init__(self): 

        feature_approximator = Nystroem(n_components = 200)
        dense_estimator = BayesianRidge()

        # Setting up the pipelines
        dense_pipeline      = Pipeline([ ('nystroem', feature_approximator),
                                         ('BRR', BayesianRidge(compute_score = True)) ])

        self.dense_hyper_params_ = [{'nystroem__kernel': [RBF(length_scale=ls)
                                            for ls in np.logspace(-2, 2, num = 100)]}]  
        
        cv = [(slice(None), slice(None))]
        self.dense_tuner         = GridSearchCV(dense_pipeline,
                                            self.dense_hyper_params_,
                                            return_train_score = True,
                                            verbose = 1,
                                            cv = cv,
                                            n_jobs = -1)
        self.dense_estimator  = None
        self.sparse_estimator = GaussianProcessRegressor(kernel = RBF() +  WhiteKernel())
        self.estimators = None

        # Storing a scaler object for the data matrix
        self.x_scaler = StandardScaler()

        # Storing the mean and the variance of the targets for scaling
        self.y_mean = 0
        self.y_var  = 0
    pass

    def predict(self, test_x):
        """
            TODO: enter your code here
        """        
        # Scale the test data
        test_x = self.x_scaler.transform(test_x)
        
        n = test_x.shape[0]

        y_pred_mean  = np.zeros([n,2])
        y_pred_std   = np.zeros([n,2])
        y_pred       = np.zeros([n, 2])
        y_prediction = np.zeros(n)  
        for i ,estimator in enumerate(self.estimators):
            
            # predict the mean and the variance
            mean, std        = estimator.predict(test_x, return_std = True)
            y_pred_mean[:,i] = mean*(self.y_var**0.5) + self.y_mean
            y_pred_std[:,i]  = std*(self.y_var**0.5)
            
            # adjust the cost based on decision theory
            cost_factor = 1 - norm.cdf(np.divide(0.5-y_pred_mean[:,i], y_pred_std[:,i]))
            c1 = np.divide(W2 + W3*cost_factor, cost_factor+1)
            c2 = W1*np.ones([n])
            y_pred[:,i] = y_pred_mean[:,i] + np.multiply(y_pred_std[:,i], norm.ppf(np.divide(c1, c1+c2)))
            
            y_prediction += np.divide(y_pred[:,i], y_pred_std[:,i])
    
        print("################################ Predictions  ################################")
        print("BRR Prediction: ", y_pred[:,0])
        print("BRR std: ", y_pred_std[:,0])

        print("GP Prediction: ",  y_pred[:,1])
        print("GP std: ", y_pred_std[:,1])
        y_prediction = np.divide(y_prediction,  y_pred_std[:,0]**-1 + y_pred_std[:,1]**-1)  

        return y_prediction

    def fit_model(self, train_x, train_y):
        """
             TODO: enter your code here
        """
        print()
        print("################################ Grid Search ################################")
        
        # Scale the targets to match the GP prior 
        self.y_mean = np.mean(train_y)
        self.y_var  = np.var(train_y)
        
        train_y -= self.y_mean
        train_y  = train_y/(self.y_var**0.5)
    
        # Scale the data matrix for stable training
        train_x = self.x_scaler.fit_transform(train_x)

        # split the data into sparse and dense regions
        sparse_idx = np.where(train_x[:,0]>-0.5)[0]
        dense_idx  = np.where(train_x[:,0]<=-0.5)[0]
        
        train_x_sparse = train_x[sparse_idx,:]
        train_y_sparse = train_y[sparse_idx]
         
        # Tune the hyperparameters using grid search and extract the best estimator (dense domain)
        self.dense_tuner.fit(train_x, train_y)
        self.dense_estimator = self.dense_tuner.best_estimator_
        
        # subsample the dense domain for the GP model
        samples_idx = np.random.choice(dense_idx, size = (1000, ) , replace = True)
        self.sparse_estimator.fit(np.concatenate((train_x_sparse, train_x[samples_idx,:])),
                                  np.concatenate((train_y_sparse,train_y[samples_idx])))


        
        self.estimators = [self.dense_estimator, self.sparse_estimator]

        print("Grid search best score: ")
        print(self.dense_tuner.best_score_)
        print("Optimal dense pipeline: ")
        print(self.dense_tuner.best_estimator_)
    pass


        
        


def main():
    
    # Flags 
    plot_data = False
    print_data_stats = True
    
    train_x_name = "train_x.csv"
    train_y_name = "train_y.csv"

    train_x = np.loadtxt(train_x_name, delimiter=',')
    train_y = np.loadtxt(train_y_name, delimiter=',')

    # Analyze input data 
    if(plot_data):

        fig = plt.figure()
        ax = fig.add_subplot(projection = '3d')
        ax.scatter(train_x[:,0], train_x[:,1], train_y)
        ax.set_xlabel('X1 Label')
        ax.set_ylabel('X2 Label')
        ax.set_zlabel('Y Label')
        plt.show()

    if(print_data_stats):
        print("################################ Data Stats ################################")
    
        # Identify the size of the sparse/dense regions
        sparse_size = len(np.where(train_x[:,0] > -0.5)[0])
        dense_size  = train_x.shape[0] - sparse_size
        print(" Sparse data region size: ", sparse_size)
        print(" Dense data region size: ", dense_size)
        
        # Identify the mean and variance of the targets
        train_y_mean = np.mean(train_y)
        train_y_var  = np.var(train_y)
        print(" Target mean: ", train_y_mean)
        print(" Target variance: ", train_y_var)
    
    # load the test dateset
    test_x_name = "test_x.csv"
    test_x = np.loadtxt(test_x_name, delimiter=',')

    M = Model()
    M.fit_model(train_x, train_y)
    prediction = M.predict(test_x)

    print("Final Prediction: ", prediction)


if __name__ == "__main__":
    main()
