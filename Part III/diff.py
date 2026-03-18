import argparse
import os
import pandas as pd
from colorama import init, Fore

# Initialize colorama
init(autoreset=True)


def validate_file(path: str) -> None:
    """Check if the file exists, is a file, and is readable."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    if not os.path.isfile(path):
        raise ValueError(f"Not a file: {path}")
    if not os.access(path, os.R_OK):
        raise PermissionError(f"File is not readable or is locked: {path}")


def compare_csv(file1: str, file2: str) -> None:
    """Compare two CSV files, print differences, and success percentage."""
    validate_file(file1)
    validate_file(file2)

    try:
        df1 = pd.read_csv(file1)
    except Exception as e:
        print(Fore.RED + f"Error reading {file1}: {e}")
        return

    try:
        df2 = pd.read_csv(file2)
    except Exception as e:
        print(Fore.RED + f"Error reading {file2}: {e}")
        return

    if df1.shape != df2.shape:
        print(Fore.YELLOW + "Warning: the files have different sizes!")
        print(Fore.YELLOW + f"{file1}: {df1.shape}, {file2}: {df2.shape}")

    differences = []
    total_cells = 0
    matching_cells = 0

    # Compare cell by cell
    for row_idx in range(min(len(df1), len(df2))):
        for col in df1.columns:
            if col in df2.columns:
                total_cells += 1
                val1 = df1.at[row_idx, col]
                val2 = df2.at[row_idx, col]
                if pd.isna(val1) and pd.isna(val2):
                    matching_cells += 1
                    continue
                if val1 != val2:
                    differences.append((row_idx, col, val1, val2))
                else:
                    matching_cells += 1
            else:
                total_cells += 1
                differences.append(
                    (row_idx, col, df1.at[row_idx, col],
                     "Column missing in 2nd CSV")
                )

    # Print differences
    if differences:
        print("Differences found:\n")
        for diff in differences:
            row_idx, col, val1, val2 = diff
            print(
                Fore.RED
                + f"Row {row_idx}, Column '{col}': "
                  f"{file1}= {Fore.YELLOW}'{val1}' {Fore.RED}vs {file2}= '{Fore.YELLOW}{val2}'"
            )
    else:
        print(Fore.GREEN + "The two CSV files are identical!")

    # Print success percentage
    if total_cells > 0:
        success_percentage = (matching_cells / total_cells) * 100
        print(Fore.CYAN + f"\nMatching cells: {matching_cells}/{total_cells} "
              f"({success_percentage:.2f}%)")
    else:
        print(Fore.YELLOW + "No cells to compare.")


def main(file1: str, file2: str) -> None:
    """Main function to call compare_csv."""
    compare_csv(file1, file2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare two CSV files,"
                    "show differences, and success percentage."
    )
    parser.add_argument("file1", type=str, help="Path to the first CSV file")
    parser.add_argument("file2", type=str, help="Path to the second CSV file")
    args = parser.parse_args()
    main(args.file1, args.file2)
