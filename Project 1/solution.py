import numpy as np
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RBF, ExpSineSquared, RationalQuadratic, Matern
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.base import RegressorMixin
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVR
from sklearn.kernel_approximation import Nystroem
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import scipy.integrate as integrate
import scipy.special as special
import scipy.integrate as integrate
from scipy.optimize import minimize
from scipy.stats import norm

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

def optimal_action(y, std):
    expected = lambda a : expected_cost(a, y, std) 
    #Optimize
    x0 = np.array([y])
    res = minimize(expected, x0, tol = 1e-4)
    return res.x

def expected_cost(a, y, std):
    integ = lambda yi: norm.pdf(yi, loc = y, scale = std) * cost_function(yi, a)
    res, err = integrate.quad(integ, 0 , 1, epsabs=1.49e-04, epsrel = 1.49e-04)
    return res

"""
Fill in the methods of the Model. Please do not change the given methods for the checker script to work.
You can add new methods, and make changes. The checker script performs:


    M = Model()
    M.fit_model(train_x,train_y)
    prediction = M.predict(test_x)

It uses predictions to compare to the ground truth using the cost_function above.
"""


class Model(BaseEstimator,RegressorMixin):

    def __init__(self):
        #######################################
        ## TODO: Implement and test dense model
        #######################################
        
        # Model to predict on the dense domain ( x >= -0.5)
        estimator = [('trans', Nystroem(random_state = 2)), ('clf', BayesianRidge(verbose = 2))]
        pipe = Pipeline(estimator)
        # parameters = {'trans__gamma':np.logspace(0.1,5, num = 3, base = 10),
                      # 'trans__n_components':np.logspace(0,3, num = 3, base = 10).astype(int)}
        # self.model = GridSearchCV(pipe,parameters, verbose = 2, n_jobs = -1, scoring = cost_function)
        pipe.set_params( trans__gamma = 100, trans__n_components = 500)
        # self.model = pipe
        
        # Model to predict on the sparse domain (x < -0.5)
        self.model = [GaussianProcessRegressor(kernel = Matern() + WhiteKernel() + RBF(),
                                              n_restarts_optimizer = 4,
                                              normalize_y = True),
                      pipe]

    def predict(self, test_x):

        ##################################
        ### TODO: Predict in the 2 domains
        ##################################
        # Predict with full GP
        y_0, y_std_0 = self.model[0].predict(test_x, return_std = True)
        # y_0 = y_0 + 1.3*y_std_0
        for elem in range(len(y_0)):
            y_0[elem] = optimal_action(y_0[elem], y_std_0[elem])
            print(elem/len(y_0))
        # Predict with approximate GP
        y_1, y_std_1 = self.model[1].predict(test_x, return_std = True)
        # y_1 = y_1 + 1.3*y_std_1
        for elem in range(len(y_1)):
            y_1[elem] = optimal_action(y_1[elem], y_std_1[elem])
            print(elem/len(y_1))
        #Choose predictions based on domain
        sparse_domain = test_x[:,0] > -0.5
        y = sparse_domain*y_0 + np.invert(sparse_domain)*y_1
        
        return y

    def fit_model(self, train_x, train_y):
        ######################################
        ########### TODO: Fit the dense model
        ######################################

        self.model[0].fit( train_x[train_x[:,0] > -0.5], train_y[train_x[:,0] > -0.5])
        self.model[1].fit( train_x[train_x[:,0] <= -0.5], train_y[train_x[:,0] <= -0.5])
        
    def plot_Gauss(self, X, Y, test_x):
        """
        Plot the gaussian process surface and the datapoints.
        The test points are also plotted, in a different scheme.
        """
        # Generate grid
        x = X[:,0]
        y = X[:,1]
        z = Y
        xp, yp = np.mgrid[np.min(x):np.max(x):100j, np.min(y):np.max(y):100j ]
        xy = np.column_stack([xp.flat, yp.flat])
        
        # Predict on grid points
        zp = self.predict(xy)
        zp = zp.reshape(xp.shape)
        
        # Plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        # Data Points
        ax.scatter(x, y, z, c='r', marker = "+", label='data', alpha = 0.5)
        ax.scatter(test_x[:,0], test_x[:,0], 0.5*np.ones(test_x[:,0].shape), 
                   c='b', marker = "o", label='data', alpha = 0.5)
        # Gaussian Process
        ax.plot_surface(xp, yp, zp, rstride=1, cstride=1, alpha=0.3, cmap=cm.coolwarm)
        
        # Labels
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        ax.set_zlim(-1, 1)
        plt.show()


def main():
            #%%
    train_x_name = "train_x.csv"
    train_y_name = "train_y.csv"
    
    train_x = np.loadtxt(train_x_name, delimiter=',')
    train_y = np.loadtxt(train_y_name, delimiter=',')
    
    # load the test dateset
    test_x_name = "test_x.csv"
    test_x = np.loadtxt(test_x_name, delimiter=',')
    
    X_train, X_test, y_train, y_test = train_test_split( train_x, train_y, test_size=0.006, random_state=1)
    
    M = Model()
    M.fit_model(X_train, y_train)
    
    #M.plot_Gauss(train_x, train_y, test_x)  
    #M.predict(test_x)
    print(cost_function(y_test, M.predict(X_test)))

if __name__ == "__main__":
    main()
