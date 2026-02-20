import argparse

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from constants import houses, subjects, columns_to_delete


def main(path: str) -> None:
    df = pd.read_csv(path)

    for column in columns_to_delete:
        df.pop(column)

    sns.pairplot(df, hue="Hogwarts House")
    plt.show()

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
