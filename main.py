import folktables
import numpy as np
import pandas as pd
from aif360.datasets import StandardDataset
from sklearn.model_selection import train_test_split

from data_download import download_data, employment_filter
from preprocessing import preprocess
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

    preprocessed_data = preprocess(data)

    # train_train and train_val are lists containing 5 dataframes each
    # test is a single dataframe
    train_train_datasets, train_val_datasets, test = split_data(data)


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