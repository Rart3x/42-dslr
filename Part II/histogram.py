import argparse
import pandas as pd
import matplotlib.pyplot as plt
from colorama import Fore, Style
from constants import houses, subjects


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
                clean_subject_score = subject_score.dropna()
                ax.hist(clean_subject_score, label=house, alpha=0.5)
                ax.legend()
                ax.set_xlabel("Score")
                ax.set_ylabel("Frequency")
                ax.set_title(subject)

        # Hide unused axes
        for j in range(len(subjects), len(axes)):
            axes[j].set_visible(False)

        # Show the histogram
        plt.tight_layout()

        def on_click(event):
            if event.inaxes is None or event.dblclick is False:
                return

            clicked_ax = event.inaxes

            try:
                idx = list(axes).index(clicked_ax)
            except ValueError:
                return

            if idx >= len(subjects):
                return

            subject = subjects[idx]

            fig_new, ax_new = plt.subplots(figsize=(6, 6))

            for house in houses:
                subject_score = df[df["Hogwarts House"] == house][subject]
                clean_subject_score = subject_score.dropna()
                ax_new.hist(clean_subject_score, label=house, alpha=0.5)

            ax_new.set_title(subject)
            ax_new.set_xlabel("Score")
            ax_new.set_ylabel("Frequency")
            ax_new.legend()

            plt.show()

        fig.canvas.mpl_connect('button_press_event', on_click)

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
