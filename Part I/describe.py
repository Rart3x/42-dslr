import mathematical as mth
import pandas as pd
import argparse

from colorama import Fore, Style


def describe(path: str, show_all_rows: bool = False):
    """
    Reproduce the behavior of Linux 'describe' command (like pandas describe).

    Reads a CSV file and prints a summary of all numeric columns,
    including count, mean, std, min, quartiles, and max.

    :param path: The path of the CSV file to be described.
    :param show_all_rows: Whether to display all rows (stats) in the output.
    """
    try:
        # Conditionally show all rows
        if show_all_rows:
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', 200)
            pd.set_option('display.max_rows', None)

        # Load CSV into a pandas DataFrame
        df = pd.read_csv(path)

        # Select only numeric columns and drop columns that are empty
        numeric_df = (df.select_dtypes(include=["number"])
                      .dropna(axis=1, how="all"))

        # Compute statistics for each numeric column
        stats = {
            "count":    mth.count(numeric_df),
            "mean":     mth.mean(numeric_df),
            "std":      mth.std(numeric_df),
            "min":      mth.min(numeric_df),
            "25%":      mth.percentile_25(numeric_df),
            "50%":      mth.percentile_50(numeric_df),
            "75%":      mth.percentile_75(numeric_df),
            "max":      mth.max(numeric_df),
        }

        # Convert the stats dictionary to a DataFrame for better formatting
        # .T to transpose so that stats are rows and columns are column names
        describe_df = pd.DataFrame(stats).T

        # Optional: round the numbers for nicer output
        if not show_all_rows:
            describe_df = describe_df.round(6)

        # Print the summary table
        print(describe_df)

    except Exception as e:
        print(f"{Fore.RED}An error occurred: {e}{Style.RESET_ALL}")


def main():
    # Create the argument parser
    parser = argparse.ArgumentParser(
        description="Reproduce the behavior of Linux 'describe'"
                    "command for CSV files."
    )

    # Mandatory CSV file argument
    parser.add_argument(
        "file",
        type=str,
        help="Path to the CSV file to describe"
    )

    # Optional flag to show all rows
    parser.add_argument(
        "--all-rows",
        action="store_true",
        help="Display all rows of the summary table"
    )

    # Parse the arguments
    args = parser.parse_args()

    # Call describe with parsed arguments
    describe(args.file, show_all_rows=args.all_rows)


if __name__ == "__main__":
    main()
