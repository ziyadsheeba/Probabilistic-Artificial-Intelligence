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

def plot_x1_slice(model, x, x2_value):
    x1 = x[:,0]
    x2 = x[:,1]
    x1max = np.max(x1)
    x1min = np.min(x2)
    x2max = np.max(x2)
    x2min = np.min(x2)
    numPoints = 100

    x_1_axis = np.linspace(x1max, x1min,num=numPoints)
    x_2_axis = np.repeat(x2_value,numPoints).T
    
    x_sel = np.column_stack([x_1_axis, x_2_axis])
    y, y_std  = model.predict(x_sel, return_std=True)

    ci =  1.96 * y_std
    fig, ax = plt.subplots()
    ax.plot(x_1_axis,y)
    ax.fill_between(x_1_axis, (y-ci), (y+ci), color='b', alpha=.1)
    ax.set_xlabel('x_1')
    ax.set_ylabel('y')
  
    
    plt.show()




def plot_sample_points(model, x):
    x1 = x[:,0]
    x2 = x[:,1]
    x1max = np.max(x1)
    x1min = np.min(x2)
    x2max = np.max(x2)
    x2min = np.min(x2)
    numPoints = 100j

    x1, x2 = np.mgrid[x1min:x1max:numPoints, x2min:x2max:numPoints]
    x = np.column_stack([x1.flat, x2.flat])


    y = model.predict(x)    
    y = y.reshape(x1.shape)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    colormap = 'RdYlBu_r'
    #ax.scatter(x1, x2, 20, c=y-THRESHOLD, cmap=colormap)
    ax.set_xlabel('x_1')
    ax.set_ylabel('x_2')
    

    ax.plot_surface(x1,x2,y, cmap=colormap)

    plt.show()







if __name__ == "__main__":
    datapts = load_train_data_from_file()
    print(datapts[1])
    plot_3d(datapts[0], datapts[1])
    plot_2d(datapts[0], datapts[1])

# %%
