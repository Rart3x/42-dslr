import argparse

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from colorama import Fore, Style
from logreg_train import sigmoid, process_data


def plot_confusion_matrix(cm, class_names, output_path):
    """
    Plot and save confusion matrix
    """
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                cbar_kws={'label': 'Number of samples'})
    plt.title('Confusion Matrix', fontsize=16, pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"{Fore.GREEN}Confusion matrix saved to: "
          f"{output_path}{Style.RESET_ALL}")


def main(dataset_test: str, dataset_truth: str, file_weights: str) -> None:
    try:
        df = pd.read_csv(dataset_test)
        data = np.load(file_weights)

        weights_for_house = {key: data[key] for key in data.files}
        X = process_data(df)

        predictions = []

        for x in X:
            scores = {}
            for house, W in weights_for_house.items():
                prob = sigmoid(np.dot(x, W))
                scores[house] = prob
            predicted_house = max(scores, key=scores.get)
            predictions.append(predicted_house)

        (pd.DataFrame({"Hogwarts House": predictions})
         .reset_index()
         .rename(columns={"index": "Index"})
         .to_csv("houses.csv", index=False))

        print(f"{Fore.GREEN}Prediction file saved as houses.csv"
              f"{Style.RESET_ALL}")

        if dataset_truth:
            df_truth = pd.read_csv(dataset_truth)

            if "Hogwarts House" not in df_truth.columns:
                print(f"{Fore.YELLOW}"
                      f"Pas de colonne 'Hogwarts House'"
                      f"dans le dataset truth →"
                      f"matrice de confusion ignorée"
                      f"{Style.RESET_ALL}")
                return

            true_labels = df_truth["Hogwarts House"].values
            class_names = sorted(weights_for_house.keys())

            n = len(class_names)
            label_to_idx = {label: i for i, label in enumerate(class_names)}
            cm = np.zeros((n, n), dtype=int)

            for true, pred in zip(true_labels, predictions):
                if pd.notna(true):
                    i = label_to_idx[true]
                    j = label_to_idx[pred]
                    cm[i][j] += 1

            plot_confusion_matrix(cm, class_names, "confusion_matrix.png")

    except Exception as e:
        print(f"{Fore.RED}An error occurred: {e}{Style.RESET_ALL}")


if __name__ == "__main__":
    parser = (argparse.ArgumentParser
              (description="Logistic Regression Prediction"))

    parser.add_argument('dataset_test',
                        type=str,
                        help='Path to the test dataset')

    parser.add_argument('--dataset_truth',
                        type=str,

                        help='Path to the truth dataset (with true labels)')

    parser.add_argument('weights',
                        type=str,
                        help='Path to the weights file')

    args = parser.parse_args()

    main(args.dataset_test, args.dataset_truth, args.weights)
