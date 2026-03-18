import argparse

import numpy as np
import pandas as pd
from colorama import Fore, Style
from logreg_train import features_to_keep


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))


def main(dataset_test: str, file_weights: str) -> None:
    try:
        # Load dataset
        df = pd.read_csv(dataset_test)

        # Load weights + normalization values
        data = np.load(file_weights)

        mu = data["mu"]
        sigma = data["sigma"]

        # Extract weights for each house
        weights_for_house = {
            key: data[key]
            for key in data.files
            if key not in ["mu", "sigma"]
        }

        # Prepare features
        x_features = df[features_to_keep]
        x_features = x_features.fillna(x_features.mean())

        # Standardize using TRAIN values
        X = (x_features - mu) / sigma
        X = X.to_numpy()

        # Add bias term
        X = np.insert(X, 0, 1, axis=1)

        # Predictions
        predictions = []

        for x in X:
            scores = {}

            for house, W in weights_for_house.items():
                prob = sigmoid(np.dot(x, W))
                scores[house] = prob

            predicted_house = max(scores, key=scores.get)
            predictions.append(predicted_house)

        # Save results
        (pd.DataFrame({
            "Hogwarts House": predictions
        }).reset_index().rename(columns={"index": "Index"})
            .to_csv("houses.csv", index=False))

        print(f"{Fore.GREEN}"
              f"Prediction file saved as houses.csv"
              f"{Style.RESET_ALL}")

    except Exception as e:
        print(f"{Fore.RED}"
              f"An error occurred: "
              f"{e}"
              f"{Style.RESET_ALL}")


if __name__ == "__main__":
    parser = (argparse.ArgumentParser
              (description="Logistic Regression Prediction"))

    parser.add_argument('dataset_test',
                        type=str,
                        help='Path to the test dataset')

    parser.add_argument('weights',
                        type=str,
                        help='Path to the weights file')

    args = parser.parse_args()

    main(args.dataset_test, args.weights)
