import folktables
import numpy as np
import pandas as pd
from aif360.datasets import StandardDataset

from data_download import download_data, employment_filter
from evaluation import calculate_accuracy
from model_selection import training
from train_test_split import split_data

if __name__ == "__main__":
    ACSEmployment = folktables.BasicProblem(
        features=[
            "AGEP",
            # age; for range of values of features please check Appendix B.4
            "SCHL",  # educational attainment
            "MAR",  # marital status
            "RELP",  # relationship
            "DIS",  # disability recode
            "ESP",  # employment status of parents
            "CIT",  # citizenship status
            "MIG",  # mobility status ( lived here 1 year ago)
            "MIL",  # military service
            "ANC",  # ancestry recode
            "NATIVITY",  # nativity
            "DEAR",  # hearing difficulty
            "DEYE",  # vision difficulty
            "DREM",  # cognitive difficulty
            "SEX",  # sex
            "RAC1P",  # recoded detailed race code
            "GCL",  # grandparents living with grandchildren
        ],
        target="ESR",  # employment status recode
        target_transform=lambda x: x == 1,
        group="DIS",
        preprocess=employment_filter,
        postprocess=lambda x: np.nan_to_num(x, -1),
    )

    try:
        data = pd.read_csv('employment_data.csv')
    except FileNotFoundError:
        data = download_data(ACSEmployment)

    # train_train and train_val are lists containing 5 dataframes each
    # test is a single dataframe
    train_train_datasets, train_val_datasets, test = split_data(data)

    xgboost_results = []
    rf_results = []

    for i in range(5):
        train_dataset_i = train_train_datasets[i]
        train_labels = train_dataset_i['label']
        train_dataset_i.drop(['label'], axis=1, inplace=True)

        xgboost = training(train_dataset_i, train_labels,'xgb')
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


    # Define fairness - related groups
    favorable_classes = [True]
    protected_attribute_names = [ACSEmployment.group]
    privileged_classes = np.array([[1]])

    # Create AIF360 StandardDataset
    data_for_aif = StandardDataset(
        data,
        "label",
        favorable_classes=favorable_classes,
        protected_attribute_names=protected_attribute_names,
        privileged_classes=privileged_classes,
    )

    # Define privileged and unprivileged groups
    privileged_groups = [{"DIS ": 1}]
    unprivileged_groups = [{"DIS ": 2}]