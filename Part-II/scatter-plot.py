from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


def main() -> None:
    print("Hello World")

if __name__ == "__main__":
    parser = ArgumentParser(
        description="Scatter plot",
    )