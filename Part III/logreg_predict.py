import argparse

import numpy as np
import pandas as pd
from colorama import Fore, Style

from logreg_train import sigmoid, process_data


def main(dataset_test: str, file_weights: str) -> None:
    try:
        # Load dataset
        df = pd.read_csv(dataset_test)

        # Load weights
        data = np.load(file_weights)

        # Extract weights for each house
        weights_for_house = {
            key: data[key]
            for key in data.files
        }

        # Get a standardized version of the dataFrame ready for predictions
        X = process_data(df)

        # Predictions
        predictions = []

        for x in X:
            scores = {}
            # Iterating through every house
            # to see which one is the most likely for the student
            for house, W in weights_for_house.items():
                prob = sigmoid(np.dot(x, W))
                scores[house] = prob

            # Taking the highest probability over the four houses
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
