import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier


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


def demographic_parity_difference(y_pred, sa):
    """
    Calculates the Demographic Parity Difference (DPD).
    """
    sa_0 = y_pred[sa == 1]
    sa_1 = y_pred[sa == 2]
    rate_0 = np.mean(sa_0)
    rate_1 = np.mean(sa_1)
    return abs(rate_0 - rate_1)


def calculate_demographic_parity_difference(predictions,
                                                sensitive_attribute,
                                                positive_class=1):
    """
    Measures the demographic parity of a model's predictions.

    Parameters:
        predictions (array-like): Model predictions (binary or probabilities).
        sensitive_attribute (array-like): Sensitive attribute values (e.g.,
            gender, race).
        positive_class (int or float, optional): The value of the positive
            class in predictions. Default is 1.

    Returns:
        dict: A dictionary with the positive prediction rates for each group
            and the demographic parity difference.
    """
    # Convert inputs to pandas Series for easier manipulation
    predictions = pd.Series(predictions)
    sensitive_attribute = pd.Series(sensitive_attribute)

    # Identify unique groups in the sensitive attribute
    groups = sensitive_attribute.unique()

    # Calculate the positive prediction rate for each group
    positive_rates = {}
    for group in groups:
        group_mask = sensitive_attribute == group
        group_mask = group_mask.reindex(predictions.index, fill_value=False)
        positive_rate = np.mean(predictions[group_mask] == positive_class)
        positive_rates[group] = positive_rate

    # Calculate demographic parity difference
    parity_difference = max(positive_rates.values()) - min(
        positive_rates.values())

    return parity_difference


def evaluation(train, test):
    model = 'xgboost'

    y_train = train['label']
    y_test = test['label']

    train.drop(['label'], axis=1, inplace=True)
    test.drop(['label'], axis=1, inplace=True)

    if model == 'xgboost':
        dtrain = xgb.DMatrix(train, label=y_train)
        dtest = xgb.DMatrix(test, label=y_test)

        # XGBoost parameters
        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "eta": 0.2932425687600753,
            "max_depth": 9,
            "use_label_encoder": False,
        }

        model = xgb.train(params, dtrain)

        predictions = model.predict(dtest) > 0.5  # Binary predictions
    elif model == 'rf':
        rf = RandomForestClassifier(n_estimators=100, random_state=420,
                                    oob_score=True, max_depth=20)
        rf.fit(train, y_train)

        predictions = rf.predict(test)
    accuracy = calculate_accuracy(predictions, y_test)
    dpd = demographic_parity_difference(
        predictions, test['DIS'])

    print(f"Accuracy: {accuracy}. DPD: {dpd}")


