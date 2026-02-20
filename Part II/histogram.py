import argparse
import pandas as pd
import matplotlib.pyplot as plt
from colorama import Fore, Style
from constants import houses

subjects = [
    "Arithmancy",
    "Astronomy",
    "Herbology",
    "Defense Against the Dark Arts",
    "Divination",
    "Muggle Studies",
    "Ancient Runes",
    "History of Magic",
    "Transfiguration",
    "Potions",
    "Care of Magical Creatures",
    "Charms",
    "Flying"
]


def main(path: str) -> None:
    """
    Displays a histogram showing the score of each houses in each subject.

    :param path: the path to the CSV file.
    """

    try:
        df = pd.read_csv(filepath_or_buffer=path)
        fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(10, 10))
        axes = axes.flatten()

        # Loop on each subject to create the histogram
        for i, subject in enumerate(subjects):
            ax = axes[i]
            for house in houses:
                subject_score = df[df["Hogwarts House"] == house][subject]
                subject_score.dropna(inplace=True)
                ax.hist(subject_score, label=house, alpha=0.5)
                ax.legend()
                ax.set_xlabel("Score")
                ax.set_ylabel("Frequency")
                ax.set_title(subject)

        # Hide unused axes
        for j in range(len(subjects), len(axes)):
            axes[j].set_visible(False)

        # Show the histogram
        plt.tight_layout()
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
