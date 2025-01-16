import folktables
import numpy as np
import pandas as pd
from aif360.datasets import StandardDataset
from folktables import ACSDataSource


def employment_filter(data):
    """
    Filters for the employment prediction task
    """
    df = data
    df = df[df["AGEP"] > 16]
    df = df[df["AGEP"] < 90]
    df = df[df["PWGTP"] >= 1]
    return df


def download_data(ACSEmployment):
    data_source = ACSDataSource(
        survey_year="2018", horizon="1-Year", survey="person"
    )
    acs_data = data_source.get_data(
        states=["FL"], download=True
    )  # data for Florida state
    features, label, group = ACSEmployment.df_to_numpy(acs_data)

    # Convert features and labels into a DataFrame
    data = pd.DataFrame(features, columns=ACSEmployment.features)
    data["label"] = label

    # save data to a csv file
    data.to_csv("employment_data.csv", index=False)
    print(data.head())

    return data

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