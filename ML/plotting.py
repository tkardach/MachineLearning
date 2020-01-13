import numpy as np
import matplotlib.pyplot as plt

def generate_validation_curve(train_scores, validation_scores, train_sizes, name=None):
    train_scores_mean = -np.mean(train_scores, axis=1)
    validation_scores_mean = -np.mean(validation_scores, axis=1)

    if name is not None:
        plt.figure(num=name)
    else:
        plt.figure()

    plt.grid()
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
                    label="Training score", markersize=3)
    plt.plot(train_sizes, validation_scores_mean, 'o-', color="g",
                    label="Cross-validation score", markersize=3)
    plt.legend(loc="best")
    
    plt.ylim(train_scores_mean.min() - 1.0, validation_scores_mean.max() + 1.0)
    plt.xlim(train_sizes.min(), train_sizes.max())
    plt.draw()