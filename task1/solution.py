import numpy as np
from models import *
if __name__=='__main__' :
    #plotting 
    from graphics import *


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
    def __init__(self):
        self.estimator = None
        ## DENSE DATA REGION ESTIMATOR
        # Hyperparameter for grid search 
        self.hyper_param_grid = [
            {'nystroem__kernel': [o*Sum(Matern(length_scale=lm,nu=n), RBF())
                             for lm in np.logspace(-1, 0.1, 5)
                             for n in np.logspace(-2, 0.4, 5)
                             for o in np.logspace(-2,1,5)] ,}
        ]    

        # Kernel approximation for dense data region        
        self.feature_map = Nystroem(, n_components=100, random_state=2)        
        #Pipeline
        self.pipeline = Pipeline([('nystroem', self.feature_map),
                                 ('brr', linear_model.BayesianRidge(compute_score=True))])
        
        self.pipeline.set_params(nystroem__gamma = 100)


        #dense_estimator = RandomizedSearchCV(self.pipeline, self.hyper_param_grid, random_state = 3, cv=2, return_train_score=True, verbose=2, refit=True, n_jobs=2, n_iter=2)
        #dense_estimator = GridSearchCV(self.pipeline, self.hyper_param_grid, cv=2, return_train_score=True, verbose=1, refit=True, n_jobs=2)
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
    prediction = M.predict(test_x)
    #print(prediction)

        
    #plot_3d(test_x, prediction)
    #analyze_prediction(M, train_x, train_y)
    plot_sample_points(M,train_x)
    plot_x1_slice(M, train_x, 0.4)




if __name__ == "__main__":
    main()
