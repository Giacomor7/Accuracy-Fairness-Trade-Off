import numpy as np

from evaluation import calculate_accuracy
from model_selection import training


def get_baseline(train_train_datasets, train_val_datasets):
    xgboost_results = []
    rf_results = []

    for i in range(5):
        train_dataset_i = train_train_datasets[i]
        train_labels = train_dataset_i['label']
        train_dataset_i.drop(['label'], axis=1, inplace=True)

        xgboost = training(train_dataset_i, train_labels, 'xgb')
        rf = training(train_dataset_i, train_labels, 'rf')

        val_dataset_i = train_val_datasets[i]
        labels = val_dataset_i['label']
        val_dataset_i.drop('label', axis=1, inplace=True)

        xgboost_predictions = xgboost.predict(val_dataset_i)
        rf_predictions = rf.predict(val_dataset_i)

        xgboost_results.append(calculate_accuracy(xgboost_predictions, labels))
        rf_results.append(calculate_accuracy(rf_predictions, labels))

    print("xgboost mean accuracy: ", np.mean(xgboost_results))
    print("rf mean accuracy: ", np.mean(rf_results))