import argparse
import pandas as pd
import matplotlib.pyplot as plt

houses = [
    "Ravenclaw",
    "Hufflepuff",
    "Slytherin",
    "Gryffindor",
]

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

def main(path: str):
    df = pd.read_csv(filepath_or_buffer=path)
    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(10, 10))
    axes = axes.flatten()

    for i, subject in enumerate(subjects):
        ax = axes[i]
        for house in houses:
            subject_score = df[ df["Hogwarts House"] == house][subject]
            subject_score.dropna(inplace=True)
            ax.hist(subject_score, label=house, alpha=0.5)
            ax.legend()
            ax.set_xlabel("Score")
            ax.set_ylabel("Frequency")
            ax.set_title(subject)

    for j in range(len(subjects), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('path', type=str, help='path to file (the dataset)')
    args = parser.parse_args()

    main(args.path)