import sys
sys.path.append('..\MLNumpy')

import numpy as np
from ML.common import *
from ML.plotting import generate_validation_curve
from matplotlib import ticker, cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
    

# Return a scatter plot using the provided X and Y points
def plotGraph(X, y, graphName="Scatter plot", xlabel="X Axis", ylabel="Y Axis", figureName="Scatter Plot Figure"):
    plt.figure(num=figureName)
    plt.scatter(X, y, s=5)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(graphName)
    return plt

# Return a surface using the x/y ranges along with the respective Z values
def plotSurface(x, y, Z, graphName="Surface graph", figureName="Surface Graph Figure"):
    fig = plt.figure(num=figureName)
    ax = Axes3D(fig)
    ax.set_title(graphName)
    X, Y = np.meshgrid(x, y)
    ax.plot_surface(X, Y, Z)
    return plt

# Return a contour plot using the x/y ranges along with the respective Z values
def plotContour(x, y, Z, logRange, graphName="Contour plot", figureName="Contour Plot Figure"):
    X, Y = np.meshgrid(x, y)
    fig, ax = plt.subplots(1, 1, num=figureName)
    plt.title(graphName)

    cp = ax.contourf(X, Y, Z, logRange, cmap=cm.PuBu_r)
    fig.colorbar(cp)
    return plt

def do_coursework(X, y):
    m = len(X)

    # Add ones vector to X array
    X = np.insert(X, 0, 1.0, axis=1)
    theta = np.zeros(2, dtype=int)

    itterations = 1500
    alpha = 0.01

    print("\nTesting the cost function...\n")
    J = computeCost(X, y, np.array([-1, 2]))

    print("With theta = [-1; 2]\n Cost Computed = %f\n", J)
    print("Expected cost value (approx) 54.24\n")

    print("\nRunning Gradient Descent...\n")
    theta, J_history = gradientDescent(X, y, theta, alpha, itterations)

    print("Theta found by gradient descent:\n")
    print(theta)
    print("\nExpected values (approx) -3.6303, 1.1664\n")

    print("\nPlotting linear fit\n")
    linReg = plotGraph(X[:, [1]], y, graphName="Linear Fit", xlabel="Population of City in 10,000s",
                    ylabel="Profit in $10,000s", figureName="Linear Fit")
    linReg.plot(X[:, [1]], hypothesis(
        np.insert(X[:, [1]], 0, 1.0, axis=1), theta), c="red")
    linReg.draw()

    print("\nVisualizing the Cost Function over Itterations\n")
    costProg = plotGraph(np.arange(itterations), J_history,
                        graphName="Gradient Descent", xlabel="Itterations", ylabel="Cost Function Value", figureName="Cost Function Over Itterations")
    costProg.draw()

    predict1 = np.dot(np.matrix([1, 3.5]), theta.reshape(len(theta), 1))
    print("Predition for population = 35,000    :    ", predict1.item() * 10000)

    predict2 = np.dot(np.matrix([1, 7]), theta.reshape(len(theta), 1))
    print("Predition for population = 70,000    :    ", predict2.item() * 10000)

    print("\nVisualize the Cost Function Over 3D Space\n")
    theta0 = np.linspace(-10, 10, num=100)
    theta1 = np.linspace(-1, 4, num=100)

    Z = np.zeros((len(theta0), len(theta1)))

    for i in range(len(theta1)):
        for j in range(len(theta0)):
            t = np.array([theta0[j], theta1[i]])
            Z[i, j] = computeCost(X, y, t)


    surf = plotSurface(theta0, theta1, Z, figureName="3D Plot of Cost Function")
    surf.draw()


    print("\nVisualize the Cost Function on a Contour Graph")
    plot = plotContour(theta0, theta1, Z, np.logspace(-2, 3, 20), figureName="Contour Plot of Cost Function")
    plot.scatter(theta[0, 0], theta[1, 0], c="red")
    plot.show()

def model_dataset_1():
    scaler = StandardScaler()

    # Find a well fitted model for the first data set
    X, y = extract_data("ex1/ex1data1.csv")

    X_scaled = scaler.fit_transform(X)

    fig, (ax1, ax2) = plt.subplots(1,2)

    # Here we divide the data into training and testing sets
    #   test_size : percentage we want allocated towards the test set (20%)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=5)

    clf = LinearRegression(copy_X=True, fit_intercept=True, normalize=False)

    clf.fit(X_train, y_train)

    print("Accuracy for example 1: ", clf.score(X_test, y_test))

    # Plot learning curve of the model
    t_sizes, t_scores, cv_scores = learning_curve(
        LinearRegression(copy_X=True, fit_intercept=True, normalize=False),
        X_scaled,
        y,
        cv=5,
        train_sizes=np.linspace(0.1,1.0,15),
        shuffle=True,
        scoring="neg_mean_squared_error"
    )

    generate_validation_curve(t_scores, cv_scores, t_sizes, graph=ax1)

    # Plot data and model prediction
    ax2.scatter(X,y)
    ax2.plot(X, clf.predict(X_scaled))

    plt.draw()

def model_dataset_2():
    scaler = StandardScaler()

    # Find a well fitted model for the first data set
    X, y = extract_data("ex1/ex1data2.csv")

    X_scaled = scaler.fit_transform(X)

    # Here we divide the data into training and testing sets
    #   test_size : percentage we want allocated towards the test set (20%)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=2)

    # Here we are creating our LinearRegression object which we will use to fit our linear model
    #   The LinearRegression object will fit itself to our training data
    clf = LinearRegression(copy_X=True, fit_intercept=True, normalize=True)

    # Train our linear model using LinearRegression
    clf.fit(X_train, y_train)

    print("Accuracy for example 2: ", clf.score(X_test, y_test))

    # Generate information needed to graph a learning curve
    t_sizes, t_scores, cv_scores = learning_curve(
        LinearRegression(copy_X=True, fit_intercept=True, normalize=True),
        X_scaled,
        y,
        train_sizes=np.linspace(0.1,1.0,15),
        cv=5,
        shuffle=True,
        scoring="neg_mean_squared_error"
    )

    print(clf.coef_)
    print(clf.predict(scaler.transform([[1650, 3]])))

    generate_validation_curve(t_scores, cv_scores, t_sizes.reshape(len(t_sizes),1))


def analyze_dataset(X, y):
    best_params, best_poly, best_score = analyze_lin_reg(X, y)

    print("Best parameters : ", best_params)
    print("Best Polynomial Degree: ", best_poly)
    print("Best Score: ", best_score)


#X, y = extract_data("ex1/ex1data1.csv")
#analyze_dataset(X, y)

#X, y = extract_data("ex1/ex1data2.csv")
#analyze_dataset(X,y)

model_dataset_1()
model_dataset_2()
plt.show()

