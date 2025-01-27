from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from fair_training import fair_training

def random_forest(x, y, fairness_aware):
    if fairness_aware:
        rf = fair_training(x, y, True)
    else:
        rf = RandomForestClassifier(n_estimators=100, oob_score=True,
                                    random_state=123456)
        rf.fit(x, y)
    return rf


def xg_boost(x, y, fairness_aware):
    if fairness_aware:
        xgboost = fair_training(x, y)
    else:
        xgboost = XGBClassifier(objective='binary:logistic', random_state=420)
        xgboost.fit(x, y)
    return xgboost


def training(x, y, model_choice, fairness_aware):
    """
    Train chosen model on supplied dataset
    :param fairness_aware: (bool) should reweighing be applied before training?
    :param x: training examples.
    :param y: training labels.
    :param model_choice: string - 'rf' or 'xgb'.
    :return: trained model.
    """
    if model_choice == 'rf':
        return random_forest(x, y, fairness_aware)
    elif model_choice == 'xgb':
        return xg_boost(x, y, fairness_aware)
    else:
        raise ValueError("Model choice must be either 'rf' or 'xgb'")