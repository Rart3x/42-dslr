import argparse

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from colorama import Style, Fore

from constants import houses, subjects, columns_to_delete

matplotlib.use('TkAgg')


def main(path: str) -> None:
    df = pd.read_csv(path)
    nb_rows_lines = len(subjects)
    i = 0

    for column in columns_to_delete:
        df.pop(column)

    fig, axes = plt.subplots(nrows=nb_rows_lines, ncols=nb_rows_lines, figsize=(5, 5))
    axes = axes.flatten()

    try:
        for row_idx, subject_line in enumerate(subjects):
            for col_idx, subject_column in enumerate(subjects):
                ax = axes[i]
                if subject_column == subject_line:
                    for house in houses:
                        subject_score = df[df["Hogwarts House"] == house][subject_line]
                        clean_subject_score = subject_score.dropna()
                        ax.hist(clean_subject_score, label=house, alpha=0.5)
                else:
                    for house in houses:
                        df_house = df[df["Hogwarts House"] == house].copy()
                        clean_df_house = df_house.dropna(subset=[subject_line, subject_column])
                        ax.scatter(x=clean_df_house[subject_line],
                                   y=clean_df_house[subject_column],
                                   alpha=0.5,
                                   label=house,
                                   marker=".",
                                   s=1)

                if row_idx == len(subjects) - 1:
                    ax.set_y

                i += 1

        manager = plt.get_current_fig_manager()
        manager.window.attributes("-zoomed", True)
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
