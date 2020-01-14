# Assignment 1

# Example 1
We are given a dataset which represents  profit in $10,000s (y) for population size in 10,000s (X). We are not given any specific information what the significance of the data is: it could be the expected profit when selling a product in certain cities, or it could be the profit of a city based on its population. Using Gradient Descent, we are able to fit a linear model to the data with aroun 67% accuracy. 

![Linear Model Fit to Training Data](https://raw.githubusercontent.com/tkardach/MachineLearning/master/ex1/images/LinearModel.PNG)

It appears that the model is fit as best as it can, and the remaining error cannot be helped. To be sure, plotting the learning curve will show us whether we can achieve a more accurate result by reducing bias or variance.

![Learning Curve for Dataset 1](https://raw.githubusercontent.com/tkardach/MachineLearning/master/ex1/images/LearningCurve1.PNG)

The learning curve seems relatively ideal. It does not look like the error will decrease if we continue adding examples (which would indicate high variance), and the error does not seem high enough to indicate a high bias either. This model is a good fit.


# Example 2
We are given a dataset which represents the square footage and number of bedrooms (X) of a house and its worth (y). We need to scale our data so that the 
