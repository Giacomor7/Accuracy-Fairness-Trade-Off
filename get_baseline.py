import xgboost as xgb
import numpy as np

from evaluation import calculate_accuracy, demographic_parity_difference
from model_selection import training


def get_baseline(train_train_datasets, train_val_datasets):
    fairness_aware = True

    xgboost_results = []
    rf_results = []
    xgboost_fairness_results = []
    rf_fairness_results = []

    for i in range(5):
        train_dataset_i = train_train_datasets[i]
        train_labels = train_dataset_i['label']
        train_dataset_i.drop(['label'], axis=1, inplace=True)

        xgboost = training(train_dataset_i, train_labels, 'xgb',
                           fairness_aware)
        rf = training(train_dataset_i, train_labels, 'rf', fairness_aware)

        val_dataset_i = train_val_datasets[i]
        labels = val_dataset_i['label']
        val_dataset_i.drop('label', axis=1, inplace=True)

        dval = xgb.DMatrix(val_dataset_i, label=labels)

        xgboost_predictions = xgboost.predict(dval) > 0.5
        rf_predictions = rf.predict(val_dataset_i)

        xgboost_results.append(calculate_accuracy(xgboost_predictions, labels))
        rf_results.append(calculate_accuracy(rf_predictions, labels))
        xgboost_fairness_results.append(
            demographic_parity_difference(xgboost_predictions,
                                          val_dataset_i['DIS']))
        rf_fairness_results.append(
            demographic_parity_difference(rf_predictions,
                                          val_dataset_i['DIS']))

    print("xgboost mean accuracy: ", np.mean(xgboost_results))
    print("xgboost mean dpd: ", np.mean(xgboost_fairness_results))
    print("rf mean accuracy: ", np.mean(rf_results))
    print("rf mean dpd: ", np.mean(rf_fairness_results))