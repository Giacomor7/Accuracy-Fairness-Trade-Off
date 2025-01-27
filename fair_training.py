import numpy as np
import xgboost as xgb
from xgboost import XGBClassifier

MAX_FAIRNESS = False
WEIGHT = 1

def fair_training(x, y, sk=False, **xgb_params):
    dis = x['DIS']

    unique_groups, group_counts = np.unique(dis, return_counts=True)

    if MAX_FAIRNESS:
        group_weights = {group: 1.0 / count for group, count in
                         zip(unique_groups, group_counts)}
    else:
        group_weights = {1.0: WEIGHT, 2.0: 1}

    sample_weights = np.array([group_weights[group] for group in dis])

    dtrain = xgb.DMatrix(x, label=y, weight=sample_weights)

    default_params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": 6,
        "eta": 0.1,
        "subsample": 0.8,
    }
    default_params.update(xgb_params)

    if sk:
        model = XGBClassifier(**default_params)
        model.fit(x,y, sample_weight=sample_weights)
    else:
        model = xgb.train(params=default_params, dtrain=dtrain,
                          num_boost_round=100)

    return model