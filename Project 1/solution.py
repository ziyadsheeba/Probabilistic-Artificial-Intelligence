import numpy as np
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RBF
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.base import RegressorMixin
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt
# from matplotlib import cm

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


class Model(BaseEstimator,RegressorMixin):

    def __init__(self):
        """
            TODO: enter your code here
        """
        kernel = DotProduct() + WhiteKernel() + RBF()
        self.model = GaussianProcessRegressor(kernel=kernel, random_state=0)

    def predict(self, test_x):
        """
            TODO: enter your code here
        """
        ## dummy code below
        y = self.model.predict(test_x)
        return y

    def fit_model(self, train_x, train_y):
        """
             TODO: enter your code here
        """
        train_x, X_test, train_y, y_test = train_test_split(train_x, train_y, 
                                            shuffle = True, train_size=0.1, random_state=1)
        self.model.fit( train_x, train_y)
        print("Finished fitting.")
        
    # def plot_Gauss(self, X, Y):
    #     x = X[:,0]
    #     y = X[:,1]
    #     z = Y
    #     xp, yp = np.mgrid[np.min(x):np.max(x):100j, np.min(y):np.max(y):100j ]
    #     xy = np.column_stack([xp.flat, yp.flat])
    #     zp = self.model.predict(xy, return_std=False)
        
    #     zp = zp.reshape(xp.shape)
        
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111, projection='3d')
        
    #     # Plot gaussian
    #     ax.scatter(x, y, z, c='r', marker = "+", label='data', alpha = 0.2)
    #     ax.plot_surface(xp, yp, zp, rstride=1, cstride=1, alpha=0.7, cmap=cm.coolwarm)
    #     plt.show()


def main():
    train_x_name = "train_x.csv"
    train_y_name = "train_y.csv"

    train_x = np.loadtxt(train_x_name, delimiter=',')
    train_y = np.loadtxt(train_y_name, delimiter=',')

    # load the test dateset
    test_x_name = "test_x.csv"
    test_x = np.loadtxt(test_x_name, delimiter=',')

    M = Model()
    M.fit_model(train_x, train_y)
    prediction = M.predict(train_x)

    print(prediction)
    # M.plot_Gauss(train_x, train_y)    
    print(np.sqrt(np.mean((prediction-train_y)**2)))
    # print(cost_function(train_y, prediction))

if __name__ == "__main__":
    main()
