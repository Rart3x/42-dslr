import argparse

import matplotlib.pyplot as plt
import pandas as pd
from colorama import Fore, Style

from constants import houses

first_subject = "Astronomy"
second_subject = "Defense Against the Dark Arts"


def main(path: str) -> None:
    """
    Creates a scatter plot of the first subject and the second subject.

    :param path: path to CSV file.
    """

    try:
        df = pd.read_csv(filepath_or_buffer=path)

        # Loop on all houses to create a scatter plot for each of them
        for house in houses:
            df_house = df[df["Hogwarts House"] == house].copy()
            df_house.dropna(subset=[first_subject, second_subject],
                            inplace=True)
            plt.scatter(x=df_house[first_subject],
                        y=df_house[second_subject],
                        alpha=0.5,
                        label=house)
            plt.xlabel("Astronomy")
            plt.ylabel("Defense Against the Dark Arts")

        # Show the histogram
        plt.legend()
        plt.show()
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
