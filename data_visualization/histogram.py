import argparse
import matplotlib.pyplot as plt

def main(path):
    print(path)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('path', type=str, help='path to file (the dataset)')
    args = parser.parse_args()

    main(args.path)