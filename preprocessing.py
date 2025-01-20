def preprocess(data):
    """
    Data fields:
        AGEP (age): is left as is to preserve ordinal relationships
        SCHL (educational attainment): is left as is to preserve ordinal
            relationships
        MAR (marital status): one-hot encoded since data are categorical
        RELP (relationship): one-hot encoded since data are categorical
        DIS (disability recode): one-hot encoded since data are binary
        ESP (employment status of parents): one-hot encoded since data are
            categorical
        CIT (citizenship status): one-hot encoded since data are categorical
        MIG (mobility status): one-hot encoded since data are categorical
        MIL (military service): one-hot encoded since data are categorical
        ANC (ancestry recode): one-hot encoded since data are categorical
        NATIVITY: one-hot encoded since data are binary
        DEAR (hearing difficulty): one-hot encoded since data are binary
        DEYE (vision difficulty): one-hot encoded since data are binary
        DREM (cognitive difficulty): one-hot encoded since data are binary
        SEX: one-hot encoded since data are binary
        RAC1P (recoded detailed race code): one-hot encoded since data are
            categorical
        GCL (grandparents living with grandchildren): one-hot encoded since
            data are categorical

    :param data: employment dataset as pandas dataframe
    :return: pandas dataframe containing one-hot encoded data
    """
    pass