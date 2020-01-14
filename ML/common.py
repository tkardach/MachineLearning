import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression

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


def map_feature(x, degree=2):
    poly = PolynomialFeatures(degree)
    if len(x.shape) == 1:
        x = x.reshape(len(x), 1)
    return poly.fit_transform(x)


# Find the optimal parameters for the given estimator on the data
def optimize_parameters(est, X, y, parameters, cv=None):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

    grid = GridSearchCV(est, parameters, n_jobs=-1)

    grid.fit(X_train, y_train)

    return grid.best_params_, grid.best_score_, grid.score(X_test, y_test)


# Find the best parameters to use with logistic regression
def analyze_log_reg(X, y, poly=None):
    best_score = 0.0
    best_poly = 1
    best_params = None

    parameters = {
        'C':[0.1,1,3,10,30,100], 
        'solver':['newton-cg','lbfgs','liblinear','sag','saga'], 
        'fit_intercept':[True, False], 
        'penalty':['l1', 'l2','elasticnet','none']
        }

    for i in range(1,poly+1):
        if i == 1:
            X_poly = X
        else:
            X_poly = map_feature(X, i)
        b_params, grid_score, test_score = optimize_parameters(LogisticRegression(max_iter=1000), X_poly, y, parameters)    

        if grid_score > best_score:
            best_score = grid_score
            best_poly = i
            best_params = b_params

    return best_params, best_poly, best_score