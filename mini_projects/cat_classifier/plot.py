import numpy as np
import matplotlib.pyplot as plt

def plot_decision_boundary(model, X, y, title="Decision Boundary"):
    """
    model: function that takes X.T and returns predictions (0 or 1)
    X: input data of shape (m, 2)  -> only works for 2D features (for visualization)
    y: labels of shape (m,)
    """
    # Create a meshgrid over the feature space
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    h = 0.01  # step size

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Predict for every point in the meshgrid
    grid_points = np.c_[xx.ravel(), yy.ravel()].T  # shape (2, N)
    Z = model(grid_points)  # should return probabilities or 0/1
    Z = Z.reshape(xx.shape)

    # Plot decision boundary (contour) + scatter points
    plt.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.coolwarm)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, edgecolor='k', cmap=plt.cm.coolwarm)
    plt.title(title)
    plt.show()

def plot_cost_curve(training_cost, validation_cost=None) :
    """
    Plot cost change curve with respect to learning iteration.
    """

    plt.figure("Cost Curve")
    plt.plot(training_cost)
    if validation_cost :
        plt.plot(validation_cost)
    plt.xlabel("Iterations")
    plt.ylabel("Cost (Loss)")
    plt.title("Training Cost Curve")
    plt.legend()
    plt.grid(True)
    plt.show(block=False)
    plt.pause(2)
    # plt.close()