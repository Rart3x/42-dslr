import argparse
from typing import Tuple, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from colorama import Fore, Style

nb_epochs = 50000
learning_rate = 0.01

features_to_keep = [
    "Arithmancy",
    "Astronomy",
    "Herbology",
    "Divination",
    "Muggle Studies",
    "Ancient Runes",
    "History of Magic",
    "Transfiguration",
    "Charms",
    "Flying"
]

houses = [
    "Ravenclaw",
    "Hufflepuff",
    "Slytherin",
    "Gryffindor"
]


def process_data(df: pd.DataFrame) -> np.ndarray:
    """
    Takes a raw dataFrame as input and returns
    the cleaned and standardized version
    of the dataFrame, using the mean and the
    standard deviation :

    x_standardize = (x - mu) / sigma

    mu = mean of the dataFrame
    sigma = standard deviation of the dataFrame

    :param df: the raw dataFrame
    :return: the cleaned and standardized numpy array
    """

    # Keep only the wanted subjects
    x_features = df[features_to_keep]

    # Fill the missing values using the mean
    x_features = x_features.fillna(x_features.mean())

    # Standardize data using the mean and the standard deviation
    mu = x_features.mean()
    sigma = x_features.std()

    x_standardize = (x_features - mu) / sigma

    # Convert the pandas dataFram into a numpy array (matrix)
    X = x_standardize.to_numpy()

    # Insert ones at the start of each line as the bias
    return np.insert(X, 0, 1, axis=1)


def sigmoid(z: np.ndarray) -> np.ndarray:
    """
    the sigmoid function takes a number or list of numbers as
    a parameter and returns a value between 0 and 1 (could be 0.24 for example)
    to get a value that represents a probability. It is defined by :

    f(x) = 1 / 1 + exp(-x)

    :param z: the list we want to convert to a probability
    :return: the value between 0 and 1, representing a probability
    """
    return 1 / (1 + np.exp(-z))


def compute_loss(y: np.ndarray, h: np.ndarray) -> float:
    """
    Calculates the loss using the log-loss (binary cross entropy) function.

    y : the vector containing the answers to the prediction
    h : the vector containing the predictions done by our model

    :return: the loss
    """

    # Iterate through the predictions
    # replaces the value of h by eps if it smaller than eps
    # replaces the value of h by 1 - eps if it is higher than eps
    # This prevents abnormal calculations if the prediction is 1 or 0
    eps = 1e-15
    h = np.clip(h, eps, 1 - eps)

    loss = (- (1 / y.shape[0]) *
            np.sum((y * np.log(h) + (1 - y) * np.log(1 - h))))

    return loss


def train(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, List[float]]:
    """
    The function that trains our model through the gradient descent.

    X = our matrix of data
    W = our vector containing weights for each subject
    y = the vector containing the answers to the prediction
    learning rate = defines the step with which we perform the gradient descent

    :param X: matrix of data.
    :param y: vector containing the answers
    :return: the weights after the training, the loss history
    """
    W = np.zeros(X.shape[1])
    loss_history = []

    # Tolerance to determine if our model is still learning
    tolerance = 1e-7

    for epoch in range(nb_epochs):
        Z = np.dot(X, W)
        predictions = sigmoid(Z)
        loss_history.append(compute_loss(y, predictions))

        # Early stopping, if our learning is lower than the tolerance, we stop
        if epoch > 1:
            learning = abs(loss_history[epoch] - loss_history[epoch - 1])
            if learning < tolerance:
                break

        error = predictions - y
        gradient = 1 / X.shape[0] * np.dot(np.transpose(X), error)
        W = W - (learning_rate * gradient)

    return W, loss_history


def main(path: str) -> None:
    try:
        df = pd.read_csv(path)
        X = process_data(df)

        weights_for_house: dict[str, np.ndarray] = {}
        training_loss_history: dict[str, List[float]] = {}

        # Train our four models, one per house
        for house in houses:
            y = (df["Hogwarts House"] == house).astype(int).to_numpy()
            (weights_for_house[house],
             training_loss_history[house]) = train(X, y)

        fig, axes = plt.subplots(nrows=2, ncols=2)
        axes = axes.flatten()

        # Plot the learning curve of each model, one per house
        for i, house in enumerate(houses):
            axes[i].plot(training_loss_history[house])
            axes[i].set_title(house)
            axes[i].set_xlabel("Epochs")
            axes[i].set_ylabel("Loss")

        plt.tight_layout()
        plt.show()

        # Save the weights in a file
        np.savez("weights.npz", **weights_for_house)

        print(f"{Fore.GREEN}"
              f"Training file saved as weights.npz"
              f"{Style.RESET_ALL}")

    except Exception as e:
        print(f"{Fore.RED}An error occurred: {e}{Style.RESET_ALL}")


if __name__ == "__main__":
    # Create the argument parser
    parser = argparse.ArgumentParser(
        description=""
    )

    # Mandatory CSV file argument
    parser.add_argument('path',
                        type=str,
                        help='path to file (the dataset)')

    # Parse the arguments
    args = parser.parse_args()

    # Call the main function
    main(args.path)
