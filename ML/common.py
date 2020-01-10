import numpy as np
import matplotlib.pyplot as plt

# extract X and y data from a csv file
def extractData(filename):
    f = open(filename)

    data = np.loadtxt(fname=f, delimiter=',')

    f.close()

    X = data[:, 0:-1]
    y = data[:, -1]

    return X, y


def hypothesis(X, theta):
    return np.dot(X, theta)


# Compute the cost value from the given data and the theta values
def computeCost(X, y, theta):
    m = len(X)
    theta = theta.reshape(len(theta), 1)
    y = y.reshape(m, 1)
    predictions = hypothesis(X, theta)
    errorSqr = (predictions - y)**2
    return (1/(2*m))*errorSqr.sum()


# Perform gradient descent to find the optimal theta values
def gradientDescent(X, y, theta, alpha, itt):
    m = len(X)
    J_history = np.zeros((itt, 1))
    theta = theta.reshape(len(theta), 1)
    y = y.reshape(m, 1)
    for i in range(itt):
        gamma = (1/m)*np.dot(X.transpose(), (hypothesis(X, theta)-y))
        theta = theta - alpha*gamma
        J_history[i, 0] = computeCost(X, y, theta)

    return theta, J_history
