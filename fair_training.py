import numpy as np
import xgboost as xgb


def fair_training(x, y, **xgb_params):
    dis = x['DIS']

    unique_groups, group_counts = np.unique(dis, return_counts=True)
    group_weights = {group: 1.0 / count for group, count in
                     zip(unique_groups, group_counts)}

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

    model = xgb.train(params=default_params, dtrain=dtrain,
                      num_boost_round=100)

    return model