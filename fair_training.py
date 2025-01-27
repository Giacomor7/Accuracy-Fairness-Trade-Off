import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import pandas as pd

def calculate_sample_weights(dis, y):
    dis_label_df = pd.DataFrame(columns=['DIS', 'label'])
    dis_label_df['DIS'] = dis
    dis_label_df['label'] = y

    dis_value_counts = dis.value_counts()
    y_value_counts = y.value_counts()

    n_p = dis_value_counts[2.0] # 2.0 is able-bodied
    n_pp = len(dis_label_df[(dis_label_df['DIS'] == 2.0) & (
                dis_label_df['label'] == True)])
    n_np = len(dis_label_df[(dis_label_df['DIS'] == 2.0) & (
                dis_label_df['label'] == False)])
    n_up = dis_value_counts[1.0] # 1.0 is disabled
    n_pup = len(dis_label_df[(dis_label_df['DIS'] == 1.0) & (
                dis_label_df['label'] == True)])
    n_nup = len(dis_label_df[(dis_label_df['DIS'] == 1.0) & (
                dis_label_df['label'] == False)])
    n_pos = y_value_counts[True]
    n_neg = y_value_counts[False]
    n_total = len(dis)

    w_pp = (n_p / n_total) * (n_pos / n_pp)
    w_pup = (n_up / n_total) * (n_pos / n_pup)
    w_np = (n_p / n_total) * (n_neg / n_np)
    w_nup = (n_up / n_total) * (n_neg / n_nup)

    categories = (y.to_numpy() * 2) + dis.to_numpy()

    sample_weight_classes = {1 : w_nup, 2 : w_np, 3 : w_pup, 4 : w_pp}

    return np.array(
        [sample_weight_classes[category] for category in categories])

def xgboost_fair_training(x, y, sk=False, **xgb_params):
    dis = x['DIS']

    sample_weights = calculate_sample_weights(dis, y)

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

def random_forest_fair_training(x, y, **rf_params):
    dis = x['DIS']

    sample_weights = calculate_sample_weights(dis, y)

    default_params = {
        "n_estimators": 100,
        "max_depth": None,
        "random_state": 42,
    }
    default_params.update(rf_params)

    model = RandomForestClassifier(**default_params)
    model.fit(x, y, sample_weight=sample_weights)

    return model