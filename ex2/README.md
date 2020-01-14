# Assignment 2

Assignment 2 uses logistic regression to classify polynomial data. To find the best parameters to use with logistic regression, I use a GridSearchCV. Additionally, since our data is initially linear, we need to create polynomial features so that we can fit the data properly. Combining these 2 techniques I am able to create a model which fits the data almost perfectly.

# Example 1

From the contour map, we can see that data points of each class are divided into their own corner of the plot. There is not a single point in one class that should have been in the other. The data fits perfectly.

Viewing the learning curve for example 1 shows us that the model is a great fit, there is 0% error for the training set and the cross-validation error approaches 0 as we add more examples. Since the cross-validation curve plateaus, we can assume that adding more data will not give us more accurate results. It makes sense that this graph fits 100%, as the data is supposed to represent university admission rates based on the scores of 2 exams; we would hope that candidates are chosen purely based on the exam scores, and this curve is representative of their decision making process.

Looking at the ROC curve, we see that the area under the ROC curve is 1.0; the model predicted the test set perfectly.

![Example 1](https://raw.githubusercontent.com/tkardach/MachineLearning/master/ex2/images/GraphExample1.PNG)

# Example 2

From the contour map, we can see that the model fits the data very well. There are some data points which cross over the edges of the border, but for the most part the data is separated very well.

Viewing the learning curve for example 2 shows us that the model fits pretty well, the error is very low and the gap between the cross-validation error and the training error is small. However, it looks like we may be able to benefit from a larger dataset, as the cross-validation curve does not seem to have plateaued yet.

Looking at the ROC curve, we see that the area under the ROC curve is .85; the model predicted the test set 85% correctly. This is a good fit, but it contains 15% error.

![Example 2](https://raw.githubusercontent.com/tkardach/MachineLearning/master/ex2/images/GraphExample2.PNG)

