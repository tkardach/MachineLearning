import sys
sys.path.append('..\MLNumpy')

import numpy as np
from ML.common import *
from matplotlib import ticker, cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

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

X, y = extractData("ex1/ex1data1.csv")
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
