import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


def generate_validation_curve(train_scores, validation_scores, train_sizes, graph=None, name=None):
    train_scores_mean = -np.mean(train_scores, axis=1)
    validation_scores_mean = -np.mean(validation_scores, axis=1)

    print(train_scores_mean)
    print(validation_scores_mean)

    if graph is None:
        if name is not None:
            plt.figure(num=name)
        else:
            plt.figure()
        graph = plt
        graph.ylim(np.nanmin(train_scores_mean) - 1.0, np.nanmax(validation_scores_mean) + 1.0)
        graph.xlim(np.nanmin(train_sizes), np.nanmax(train_sizes))
    else:
        graph.set_title(name)
        graph.set_ylim(np.nanmin(train_scores_mean) - 1.0, np.nanmax(validation_scores_mean) + 1.0)
        graph.set_xlim(np.nanmin(train_sizes), np.nanmax(train_sizes))

    graph.grid()
    graph.plot(train_sizes, train_scores_mean, 'o-', color="r",
                    label="Training score", markersize=3)
    graph.plot(train_sizes, validation_scores_mean, 'o-', color="g",
                    label="Cross-validation score", markersize=3)
    graph.legend(loc="best")
    

