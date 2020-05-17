import numpy as np
import util
import matplotlib.pyplot as plt

def main(lr, train_path, eval_path, save_path):
    """Problem: Poisson regression with gradient ascent.

    Args:
        lr: Learning rate for gradient ascent.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        save_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    Poi = PoissonRegression(step_size=lr)
    Poi.fit(x_train, y_train)
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)
    y_predicted = Poi.predict(x_eval)
    np.savetxt(save_path, y_predicted)

    plt.scatter(y_eval, y_predicted, c='r')
    plt.title('Daily Traffic Poisson Prediction')
    plt.xlabel('True Count')
    plt.ylabel('Expected Predicted Count')
    plt.show()


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

class PoissonRegression:

    def __init__(self, step_size=1e-5, max_iter=10000000, eps=1e-5,
                 theta_0=None, verbose=True):
        """
        Args:
            step_size: Step size for iterative solvers only.
            max_iter: Maximum number of iterations for the solver.
            eps: Threshold for determining convergence.
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
        """
        self.theta = theta_0
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose

    def fit(self, x, y):
        """Run gradient ascent to maximize likelihood for Poisson regression.
        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """

        i = 0
        if self.theta is None:
            self.theta = np.zeros(x.shape[1])
        converge = False

        while not converge and i < self.max_iter:
            y_hat = np.exp(np.dot(x, self.theta))
            old_theta = self.theta
            change = self.step_size * np.dot(y - y_hat, x)

            self.theta = self.theta + change
            new_theta = self.theta
            if np.linalg.norm(new_theta - old_theta, ord=1) < self.eps:
                converge = True
            i += 1

    def predict(self, x):
        """Make a prediction given inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Floating-point prediction for each input, shape (n_examples,).
        """
        preds = np.dot(x, self.theta)
        return preds

if __name__ == '__main__':
    main(lr=1e-5,
        train_path='train.csv',
        eval_path='valid.csv',
        save_path='poisson_pred.txt')
