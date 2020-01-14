import sys
sys.path.append('..\MLNumpy')

from ML.common import extractData, optimize_parameters, analyze_log_reg, map_feature
from ML.plotting import generate_validation_curve
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib.offsetbox import AnchoredText


def generate_roc_curve(clf, X, y, graph=None, name=None):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

    clf.fit(X_train, y_train)

    ns_probs = [0 for _ in range(len(y_test))]

    lr_probs = clf.predict_proba(X_test)
    lr_probs = lr_probs[:,1]

    ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
    lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)

    ns_score = 'No Skill: ROC AUC=%.3f' % (roc_auc_score(y_test, ns_probs))
    lr_score = 'Logistic: ROC AUC=%.3f' % (roc_auc_score(y_test, lr_probs))


    if graph is None:
        if name is not None:
            plt.figure(num=name)
        else:
            plt.figure()
        graph = plt
        graph.xlabel("False Positive Rate")
        graph.ylabel("True Positive Rate")
    else:    
        graph.set_xlabel("False Positive Rate")
        graph.set_ylabel("True Positive Rate")
        graph.set_title(name)
    
    graph.annotate(lr_score, xy=(1,0), horizontalalignment='right', verticalalignment='bottom')

    graph.plot(ns_fpr, ns_tpr, linestyle='--', label="No Skill")
    graph.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')

    graph.legend(loc='upper right')

def generate_decision_contour_clf(u, v, X, clf, graph=None, name=None, degree=6):
    z = np.zeros((len(u), len(v)))

    # Create a grid of prediction values using the learned coefficients
    for i in range(len(u)):
        for j in range(len(v)):
            z[i,j] = clf.predict(map_feature(np.array([[u[i], v[j]]]), degree))
    z = z.transpose()

    if graph is None:
        if name is not None:
            plt.figure(num=name)
        else:
            plt.figure()
        graph = plt
        graph.xlim(u.min(), u.max())
        graph.ylim(v.min(), v.max())
    else:
        graph.set_title(name)
        graph.set_xlim(u.min(), u.max())
        graph.set_ylim(v.min(), v.max())

    graph.contourf(u, v, z, 1, cmap=cm.RdBu, alpha=.8)

    return z
    

def run_example_1(params):
    # Extract data an load into X,y variables
    X, y = extractData("ex2/ex2data1.csv")

    # Create positive and negative value index arrays
    pos = np.argwhere(y == 1)
    neg = np.argwhere(y == 0)

    # Regularize the data by adding polynomial features
    X = map_feature(X, 2)

    # Create the training set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

    # Fit the training data to a logistic regression model
    clf = LogisticRegression(**params)
    clf.fit(X_train, y_train)

    #Plot the logistic classification decision boundary
    x_min, x_max = X[:,1].min(), X[:,1].max()
    y_min, y_max = X[:,2].min(), X[:,2].max()

    fig, (ax1, ax2, ax3) = plt.subplots(1,3)

    generate_decision_contour_clf(
        np.linspace(x_min - .5, x_max + .5, 100), 
        np.linspace(y_min - .5, y_max + .5, 100), 
        X, 
        clf, 
        graph=ax1,
        name="Logistic Regression Model",
        degree=2)

    # Add data points to contour map
    ax1.scatter(X[pos,1], X[pos,2], marker='+', c="green")
    ax1.scatter(X[neg,1], X[neg,2], marker='.', c="red")
    ax1.text(95, 35, ("%.2f" % clf.score(X_test, y_test)).lstrip('0'), size=15, horizontalalignment='right')

    # Plot the validation curve
    t_sizes, t_scores, cv_scores = learning_curve(
        LogisticRegression(**params), 
        X, 
        y, 
        train_sizes= np.linspace(0.01,1.0,15),
        cv = 3,
        shuffle=True,
        scoring="neg_mean_squared_error"
    )

    generate_validation_curve(t_scores, cv_scores, t_sizes, graph=ax2, name="Learning Curve")

    generate_roc_curve(LogisticRegression(**params), X, y, graph=ax3, name="ROC Curve")


def run_example_2(params):
    # Extract data an load into X,y variables
    X, y = extractData("ex2/ex2data2.csv")

    # Create positive and negative value index arrays
    pos = np.argwhere(y == 1)
    neg = np.argwhere(y == 0)

    # Regularize the data by adding polynomial features
    X = map_feature(X, 3)

    # Create the training set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Fit the training data to a logistic regression model
    clf = LogisticRegression(**params)
    clf.fit(X_train, y_train)

    # Plot the logistic classification decision boundary
    x_min, x_max = X[:,1].min(), X[:,1].max()
    y_min, y_max = X[:,2].min(), X[:,2].max()

    fig, (ax1, ax2, ax3) = plt.subplots(1,3)

    generate_decision_contour_clf(
        np.linspace(x_min - .5, x_max + .5, 100), 
        np.linspace(y_min - .5, y_max + .5, 100), 
        X, 
        clf, 
        graph=ax1,
        name="Logistic Regression Model",
        degree=3)

    # Add data points to contour map
    ax1.scatter(X[pos,1], X[pos,2], marker='+', c="green")
    ax1.scatter(X[neg,1], X[neg,2], marker='.', c="red")

    ax1.text(1.0, -.75, ("%.2f" % clf.score(X_test, y_test)).lstrip('0'), size=15, horizontalalignment='right')

    # Plot the validation curve
    t_sizes, t_scores, cv_scores = learning_curve(
        LogisticRegression(**params), 
        X, 
        y, 
        train_sizes= np.linspace(0.01,1.0,15),
        cv = 3,
        shuffle=True,
        scoring="neg_mean_squared_error"
    )

    generate_validation_curve(t_scores, cv_scores, t_sizes, graph=ax2, name="Learning Curve")

    generate_roc_curve(LogisticRegression(**params), X, y, graph=ax3, name="ROC Curve")


# Find a well fitted model for the first data set
def analyze_dataset_1():
    X, y = extractData("ex2/ex2data1.csv")

    best_params, best_poly, best_score = analyze_log_reg(X, y, 10)

    print("Best parameters : ", best_params)
    print("Best Polynomial Degree: ", best_poly)
    print("Best Score: ", best_score)


# Find a well fitted model for the second data set
def analyze_dataset_2():
    X, y = extractData("ex2/ex2data2.csv")

    best_params, best_poly, best_score = analyze_log_reg(X, y, 10)

    print("Best parameters : ", best_params)
    print("Best Polynomial Degree: ", best_poly)
    print("Best Score: ", best_score)


# Analyze data sets for the best parameters
#analyze_dataset_1()
#analyze_dataset_2()

# Using the best fitting parameters, train the data and graph the results
run_example_1({'C': 3, 'solver': 'liblinear', 'fit_intercept': True, 'penalty': 'l2', 'random_state': 4, 'max_iter': 1000})
run_example_2({'C': 3, 'solver': 'newton-cg', 'fit_intercept': True, 'penalty': 'l2', 'random_state': 4, 'max_iter': 1000})
plt.show()