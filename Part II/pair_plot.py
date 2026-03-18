import argparse

import matplotlib.pyplot as plt
import pandas as pd
from colorama import Style, Fore

from constants import houses, subjects, columns_to_delete


def main(path: str) -> None:
    """
    Displays a pair plot.

    This function will loop over all different subjects and make
    a scatter plot of each subject against another subject.

    On the main diagonal, instead of plotting a scatter plot,
    we make a histogram since it's 2 times the same subject.

    :param path: the path to the CSV file.
    """

    try:
        df = pd.read_csv(path)
        nb_rows_lines = len(subjects)
        i = 0

        for column in columns_to_delete:
            df.pop(column)

        fig, axes = plt.subplots(nrows=nb_rows_lines,
                                 ncols=nb_rows_lines,
                                 figsize=(20, 20))
        axes = axes.flatten()

        for row_idx, subject_line in enumerate(subjects):
            for col_idx, subject_column in enumerate(subjects):
                ax = axes[i]
                if subject_column == subject_line:
                    for house in houses:
                        house_idx = df["Hogwarts House"] == house
                        subject_score = df[house_idx][subject_line]
                        clean_subject_score = subject_score.dropna()
                        ax.hist(clean_subject_score, label=house, alpha=0.5)
                else:
                    for house in houses:
                        df_house = df[df["Hogwarts House"] == house].copy()
                        clean_df_house = df_house.dropna()
                        ax.scatter(x=clean_df_house[subject_line],
                                   y=clean_df_house[subject_column],
                                   alpha=0.5,
                                   label=house,
                                   marker=".",
                                   s=1)

                if col_idx == 0:
                    ax.set_ylabel(subject_line,
                                  fontsize=7,
                                  rotation=0,
                                  labelpad=15,
                                  ha='right',
                                  va='center')
                    ax.tick_params(axis='y', labelsize=6)
                else:
                    ax.set_yticks([])

                if row_idx == len(subjects) - 1:
                    ax.set_xlabel(subject_column, fontsize=7)
                    ax.tick_params(axis='x', labelsize=6)
                else:
                    ax.set_xticks([])

                i += 1

        plt.subplots_adjust(wspace=0.1, hspace=0.1, bottom=0.15, left=0.15)

        def on_click(event):
            if event.inaxes is None or event.dblclick is False:
                return

            clicked_ax = event.inaxes

            try:
                idx = list(axes).index(clicked_ax)
            except ValueError:
                return

            row_idx = idx // nb_rows_lines
            col_idx = idx % nb_rows_lines

            subject_line = subjects[row_idx]
            subject_column = subjects[col_idx]

            fig_new, ax_new = plt.subplots(figsize=(6, 6))

            if subject_column == subject_line:
                for house in houses:
                    house_idx = df["Hogwarts House"] == house
                    subject_score = df[house_idx][subject_line]
                    clean_subject_score = subject_score.dropna()
                    ax_new.hist(clean_subject_score, label=house, alpha=0.5)
            else:
                for house in houses:
                    df_house = df[df["Hogwarts House"] == house].copy()
                    clean_df_house = df_house.dropna()
                    ax_new.scatter(
                        x=clean_df_house[subject_line],
                        y=clean_df_house[subject_column],
                        alpha=0.5,
                        label=house,
                        marker=".",
                        s=15
                    )

            ax_new.set_xlabel(subject_column)
            ax_new.set_ylabel(subject_line)
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
