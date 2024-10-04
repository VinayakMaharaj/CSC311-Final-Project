import numpy as np
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt

from utils import (
    load_valid_csv,
    load_public_test_csv,
    load_train_sparse,
    sparse_matrix_evaluate,
)


def bootstrapped_samples(matrix):
    """Return m bags of n samples per bag."""
    np.random.seed(99)
    bags = []
    for _ in range(3):
        mask = np.full(matrix.shape, False)
        samples = list(
            set(np.random.choice(matrix.shape[0], size=matrix.shape[0], replace=True))
        )
        mask[samples, :] = True
        bag = np.where(mask, matrix, np.nan)
        bags.append(bag)
    return bags


def aggregate(matrices):
    """Calculate the average predictions using the prediction matrices given."""
    stacked_matrices = np.stack(matrices, axis=0)
    average_predictions = np.mean(stacked_matrices, axis=0)
    aggregated_predictions = np.round(average_predictions).astype(int)
    return aggregated_predictions


def main():
    sparse_matrix = load_train_sparse("./data").toarray()
    valid_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data")

    ks = [1, 6, 11, 16, 21, 26]

    predictions = np.empty((3,), dtype=object)
    best_accuracies = [-1, -1, -1]

    bags = bootstrapped_samples(sparse_matrix)

    for i in range(3):
        for k in ks:
            nbrs = KNNImputer(n_neighbors=k, keep_empty_features=True)
            matrix = nbrs.fit_transform(bags[i])
            accuracy = sparse_matrix_evaluate(valid_data, matrix)
            if accuracy > best_accuracies[i]:
                predictions[i], best_accuracies[i] = matrix, accuracy

    average_prediction = aggregate(predictions)
    print(
        f"Validation accuracy: {sparse_matrix_evaluate(valid_data, average_prediction)}"
    )
    print(f"Test accuracy: {sparse_matrix_evaluate(test_data, average_prediction)}")


if __name__ == "__main__":
    main()
