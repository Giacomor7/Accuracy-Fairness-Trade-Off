import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier

from fair_training import xgboost_fair_training, random_forest_fair_training


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


def evaluation(train, test):
    #['xgboost', 'rf']
    model = 'rf'
    fairness_aware = False

    # In order of highest correlation coefficient with DIS:
    # ['DIS', 'DREM', 'DEAR', 'DEYE', 'AGEP', ...]
    features_to_drop = ['DIS', 'DREM', 'DEAR', 'DEYE', 'AGEP']

    y_train = train['label']
    y_test = test['label']

    train.drop(['label'], axis=1, inplace=True)
    test.drop(['label'], axis=1, inplace=True)

    train_dis = train['DIS']
    test_dis = test['DIS']

    train.drop(features_to_drop, axis=1, inplace=True)
    test.drop(features_to_drop, axis=1, inplace=True)

    if model == 'xgboost':
        dtrain = xgb.DMatrix(train, label=y_train)
        dtest = xgb.DMatrix(test, label=y_test)

        # XGBoost parameters
        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "eta": 0.016117651313141638,
            "max_depth": 8,
            "use_label_encoder": False,
        }

        if fairness_aware:
            model = xgboost_fair_training(train, y_train,
                                          sensitive_attribute=train_dis,
                                          **params)
            predictions = model.predict(dtest) > 0.5  # Binary predictions
        else:
            model = xgb.train(params, dtrain)
            predictions = model.predict(dtest) > 0.5  # Binary predictions


    elif model == 'rf':
        if fairness_aware:
            params = {
                "n_estimators": 354,
                "max_depth": 9,
                "random_state": 42,
            }

            rf = random_forest_fair_training(train, y_train, train_dis,
                                             **params)
        else:
            rf = RandomForestClassifier(n_estimators=153, random_state=420,
                                        oob_score=True, max_depth=13)
            rf.fit(train, y_train)

        predictions = rf.predict(test)
    accuracy = calculate_accuracy(predictions, y_test)
    dpd = demographic_parity_difference(
        predictions, test_dis)

    print(f"Accuracy: {accuracy}. DPD: {dpd}")

