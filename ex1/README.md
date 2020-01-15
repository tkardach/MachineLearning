# Assignment 1

Assignment 1 uses linear regression to model continuous linear data. To find the best parameters to use with linear regression, I use a GridSearchCV with a polynomial feature generator to find the best results; The best fits come from linear data, with no additional polynomial features.

# Example 1
We are given a dataset which represents  profit in $10,000s (y) for population size in 10,000s (X). We are not given any specific information what the significance of the data is: it could be the expected profit when selling a product in certain cities, or it could be the profit of a city based on its population. Using Gradient Descent, we are able to fit a linear model to the data with around 75% accuracy. 

It appears that the model is fit as best as it can, and the remaining error cannot be helped. To be sure, plotting the learning curve will show us whether we can achieve a more accurate result by reducing bias or variance. The learning curve seems relatively ideal. It does not look like the error will decrease if we continue adding examples (which would indicate high variance), and the error does not seem high enough to indicate a high bias either. This model is a good fit.

![Learning Curve for Dataset 1](https://raw.githubusercontent.com/tkardach/MachineLearning/master/ex1/images/LearningCurve1.PNG)



# Example 2
We are given a dataset which represents the square footage and number of bedrooms of a house (X) and its worth (y). We scale our data so that the estimator will treat each feature equally and optimize our estimator with the best parameters. We are able to fit a linear model to the data with around 82% accuracy.

The learning curve looks pretty ideal; it has low error for both the training and validation curves, and the gap between the training and validation curves is very small. According to the learning curve the model is a good fit.

![Learning Curve for Dataset 2](https://raw.githubusercontent.com/tkardach/MachineLearning/master/ex1/images/LearningCurve2.PNG)