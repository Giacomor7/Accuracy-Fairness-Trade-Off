from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def random_forest(x, y):
    rf = RandomForestClassifier(n_estimators=100, oob_score=True,
                                random_state=123456)
    rf.fit(x, y)
    return rf


def xg_boost(x, y):
    xgb = XGBClassifier(objective='binary:logistic', random_state=420)
    xgb.fit(x, y)
    return xgb


def training(x, y, model_choice):
    """
    Train chosen model on supplied dataset
    :param data: training data
    :param model_choice: string - 'rf' or 'xgb'
    :return: trained model
    """
    if model_choice == 'rf':
        return random_forest(x, y)
    elif model_choice == 'xgb':
        return xg_boost(x, y)
    else:
        raise ValueError("Model choice must be either 'rf' or 'xgb'")