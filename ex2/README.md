# Assignment 2

Assignment 2 uses logistic regression to classify polynomial data. To find the best parameters to use with logistic regression, I use a GridSearchCV. Additionally, since our data is initially linear, we need to create polynomial features so that we can fit the data properly. Combining these 2 techniques I am able to create a model which fits the data almost perfectly.

![Example 1 Contour](https://raw.githubusercontent.com/tkardach/MachineLearning/master/ex2/images/Ex1Contour.PNG)
![Example 2 Contour](https://raw.githubusercontent.com/tkardach/MachineLearning/master/ex2/images/Ex2Contour.PNG)

Viewing the learning curve for example 1 shows us that the model is a great fit, there is 0% error for the training set and the cross-validation error approaches 0 as we add more examples. Since the cross-validation curve plateaus, we can assume that adding more data will not give us more accurate results. It makes sense that this graph fits 100%, as the data is supposed to represent university admission rates based on the scores of 2 exams; we would hope that candidates are chosen purely based on the exam scores, and this curve is representative of their decision making process.

![Example 1 Learning Curve](https://raw.githubusercontent.com/tkardach/MachineLearning/master/ex2/images/Ex1LearningCurve.PNG)

Viewing the learning curve for example 2 shows us that the model fits pretty well, the error is very low and the gap between the cross-validation error and the training error is small. However, it looks like we may be able to benefit from a larger dataset, as the cross-validation curve does not seem to have plateaued yet.

![Example 2 Learning Curve](https://raw.githubusercontent.com/tkardach/MachineLearning/master/ex2/images/Ex2LearningCurve.PNG)

