import sys
sys.path.append('..\MLNumpy')

from ML.common import extractData
from ML.plotting import generate_validation_curve
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split


def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy


def map_feature(x, degree=2):
    poly = PolynomialFeatures(degree)
    if len(x.shape) == 1:
        x = x.reshape(len(x), 1)
    return poly.fit_transform(x)


def generate_decision_contour(u, v, X, theta, name=None, degree=6):
    z = np.zeros((len(u), len(v)))

    # Create a grid of prediction values using the learned coefficients
    for i in range(len(u)):
        for j in range(len(v)):
            z[i,j] = np.dot(map_feature(np.array([[u[i], v[j]]]), degree), theta.transpose())
    z = z.transpose()

    if name is not None:
        plt.figure(num=name)
    else:
        plt.figure()

    plt.xlim(u.min(), u.max())
    plt.ylim(v.min(), v.max())
    plt.contourf(u, v, z, 0, cmap=cm.RdBu, alpha=.8)
    plt.draw()

    return z
    

def run_example_1():
    # Extract data an load into X,y variables
    X, y = extractData("ex2/ex2data1.csv")

    # Create positive and negative value index arrays
    pos = np.argwhere(y == 1)
    neg = np.argwhere(y == 0)

    # Regularize the data by adding polynomial features
    X = map_feature(X, 2)

    # Create the training set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=4)

    # Fit the training data to a logistic regression model
    clf = LogisticRegression()
    clf.fit(X_train, y_train)

    #Plot the logistic classification decision boundary
    x_min, x_max = X[:,1].min(), X[:,1].max()
    y_min, y_max = X[:,2].min(), X[:,2].max()

    generate_decision_contour(
        np.linspace(x_min - .5, x_max + .5, 50), 
        np.linspace(y_min - .5, y_max + .5, 50), 
        X, 
        clf.coef_, 
        name="Example 1 : Logistic Regression over Polynomial Features",
        degree=2)

    # Add data points to contour map
    plt.scatter(X[pos,1], X[pos,2], marker='+', c="green")
    plt.scatter(X[neg,1], X[neg,2], marker='.', c="red")
    plt.text(95, 35, ("%.2f" % clf.score(X_test, y_test)).lstrip('0'), size=15, horizontalalignment='right')

    plt.draw()

    # Plot the validation curve
    t_sizes, t_scores, cv_scores = learning_curve(
        LogisticRegression(), 
        X, 
        y, 
        train_sizes= np.arange(1, len(X_train), 1),
        cv = 5,
        shuffle=True,
        scoring="neg_mean_squared_error"
    )

    generate_validation_curve(t_scores, cv_scores, t_sizes, name="Validation Curve 1")


def run_example_2():
    # Extract data an load into X,y variables
    X, y = extractData("ex2/ex2data2.csv")

    # Create positive and negative value index arrays
    pos = np.argwhere(y == 1)
    neg = np.argwhere(y == 0)

    # Regularize the data by adding polynomial features
    X = map_feature(X, 6)

    # Create the training set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)

    # Fit the training data to a logistic regression model
    clf = LogisticRegression(C=10, solver='liblinear', random_state=4, fit_intercept=False)
    clf.fit(X_train, y_train)

    # Plot the logistic classification decision boundary
    x_min, x_max = X[:,1].min(), X[:,1].max()
    y_min, y_max = X[:,2].min(), X[:,2].max()
    generate_decision_contour(
        np.linspace(x_min - .5, x_max + .5, 50), 
        np.linspace(y_min - .5, y_max + .5, 50), 
        X, 
        clf.coef_, 
        name="Example 2 : Logistic Regression over Polynomial Features",
        degree=6)

    # Add data points to contour map
    plt.scatter(X[pos,1], X[pos,2], marker='+', c="green")
    plt.scatter(X[neg,1], X[neg,2], marker='.', c="red")

    plt.text(1.0, -.75, ("%.2f" % clf.score(X_test, y_test)).lstrip('0'), size=15, horizontalalignment='right')

    # Plot the validation curve
    t_sizes, t_scores, cv_scores = learning_curve(
        LogisticRegression(C=10, solver='liblinear', random_state=4, fit_intercept=False), 
        X, 
        y, 
        train_sizes= np.arange(1, len(X_train), 1),
        cv = 5,
        shuffle=True,
        scoring="neg_mean_squared_error"
    )

    generate_validation_curve(t_scores, cv_scores, t_sizes, name="Validation Curve 2")


run_example_1()
run_example_2()
plt.show()
