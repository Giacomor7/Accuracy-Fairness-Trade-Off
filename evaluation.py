import numpy as np


def calculate_accuracy(predictions: np.ndarray, labels: np.ndarray) -> float:
    """
    Calculate the accuracy given the predictions and labels.

    Parameters:
        predictions (np.ndarray): Array of predicted values.
        labels (np.ndarray): Array of true labels.

    Returns:
        float: Accuracy as a value between 0 and 1.
    """
    if predictions.shape != labels.shape:
        raise ValueError("Shape of predictions and labels must be the same.")

    correct = np.sum(predictions == labels)
    total = len(labels)
    accuracy = correct / total
    return accuracy