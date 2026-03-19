import argparse

import numpy as np
import pandas as pd
from colorama import Fore, Style

nb_epochs = 10000
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


def train(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    The function that trains our model through the gradient descent.

    X = our matrix of data
    W = our vector containing weights for each subject
    y = the vector containing the answers to the prediction
    learning rate = defines the step with which we perform the gradient descent

    :param X: matrix of data.
    :param y: vector containing the answers
    :return: the weights after the training
    """
    W = np.zeros(X.shape[1])

    for epoch in range(nb_epochs):
        Z = np.dot(X, W)
        predictions = sigmoid(Z)
        error = predictions - y
        gradient = 1 / X.shape[0] * np.dot(np.transpose(X), error)
        W = W - (learning_rate * gradient)

    return W


def main(path: str) -> None:
    try:
        df = pd.read_csv(path)

        X = process_data(df)
        weights_for_house: dict[str, np.ndarray] = {}

        for house in houses:
            y = (df["Hogwarts House"] == house).astype(int).to_numpy()
            weights_for_house[house] = train(X, y)

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
