from folktables import ACSDataSource
import pandas as pd


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

def download_data_for_state(ACSEmployment, state):
    data_source = ACSDataSource(
        survey_year="2018", horizon="1-Year", survey="person"
    )
    acs_data = data_source.get_data(
        states=[state], download=True
    )  # data for provided state
    features, label, group = ACSEmployment.df_to_numpy(acs_data)

    # Convert features and labels into a DataFrame
    data = pd.DataFrame(features, columns=ACSEmployment.features)
    data["label"] = label

    # save data to a csv file
    data.to_csv(f"employment_data_{state}.csv", index=False)
    print(data.head())

    return data