import argparse

import numpy as np
import pandas as pd
from colorama import Fore, Style
from logreg_train import features_to_keep


def main(dataset_test: str, file_weights: str) -> None:
    try:
        df = pd.read_csv(dataset_test)
        np.load(file_weights)

        x_features = df[features_to_keep]
        x_features = x_features.fillna(x_features.mean())

    except Exception as e:
        print(f"{Fore.RED}An error occurred: {e}{Style.RESET_ALL}")


if __name__ == "__main__":
    # Create the argument parser
    parser = argparse.ArgumentParser(
        description=""
    )

    # Mandatory CSV file argument
    parser.add_argument('dataset_test',
                        type=str,
                        help='Path to the test dataset')

    # Mandatory file containing weights
    parser.add_argument('weights',
                        type=str,
                        help='Path to the weights file')

    # Parse the arguments
    args = parser.parse_args()

    # Call the main function
    main(args.dataset_test, args.weights)