#%%
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

THRESHOLD = 0.5
W1 = 1
W2 = 20
W3 = 100
W4 = 0.04

def load_train_data_from_file():
    train_x_name = "train_x.csv"
    train_y_name = "train_y.csv"

    train_x = np.loadtxt(train_x_name, delimiter=',')
    train_y = np.loadtxt(train_y_name, delimiter=',')

    print("Dimensions of training data X : {},  y : {}".format(train_x.shape, train_y.shape))

    return (train_x, train_y)

"""
    Plots data points in 3D 
    x : 2D data points
    y : 1D label

"""
def plot_3d(x,y):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x1 = x[:,0]
    x2 = x[:,1]
    colormap = 'RdYlBu_r'
    ax.scatter(x1, x2, y,c=y-THRESHOLD, cmap=colormap)
    ax.set_xlabel('x_1')
    ax.set_ylabel('x_2')

    #plot colorbar
    scalarmappaple = cm.ScalarMappable(cmap=colormap)
    scalarmappaple.set_array(y-THRESHOLD)
    cbar = plt.colorbar(scalarmappaple)      
    cbar.set_label("y - 0.5")


    plt.show()
    return fig

def plot_2d(x,y):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x1 = x[:,0]
    x2 = x[:,1]
    colormap = 'RdYlBu_r'
    ax.scatter(x1, x2, 20, c=y-THRESHOLD, cmap=colormap)
    ax.set_xlabel('x_1')
    ax.set_ylabel('x_2')

    #plot colorbar
    scalarmappaple = cm.ScalarMappable(cmap=colormap)
    scalarmappaple.set_array(y-THRESHOLD)
    cbar = plt.colorbar(scalarmappaple)      
    cbar.set_label("y - 0.5")


    plt.show()
    return fig

def plot_x1_slice(x, y, x1_pos):
    pass





if __name__ == "__main__":
    datapts = load_train_data_from_file()
    print(datapts[1])
    plot_3d(datapts[0], datapts[1])
    plot_2d(datapts[0], datapts[1])

# %%
