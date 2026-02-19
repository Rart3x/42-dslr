import sys

from colorama import Fore, Style


def main():
    """"""
    if len(sys.argv) != 2:
        print(
            f"{Fore.RED}Usage: {sys.argv[0]} <file_path>{Style.RESET_ALL}"
        )
        return

    args = sys.argv[1]

    try:
        with open(args, "r") as f:
            file_path = f.read()
    except Exception as e:
        print(f"{Fore.RED}Error reading file: {e}{Style.RESET_ALL}")
        return


if __name__ == "__main__":
    main()