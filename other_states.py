import xgboost as xgb
import pandas as pd

from data_download import download_data_for_state
from evaluation import calculate_accuracy, demographic_parity_difference
from fair_training import xgboost_fair_training

STATES = [
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID",
    "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS",
    "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK",
    "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV",
    "WI", "WY"
]

def other_states(train, ACSEmployment):
    y_train = train['label']
    train.drop(['label'], axis=1, inplace=True)
    dtrain = xgb.DMatrix(train, label=y_train)

    fair_params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "eta": 0.016117651313141638,
        "max_depth": 8,
        "use_label_encoder": False,
    }

    accurate_params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "eta": 0.12355187904055383,
        "max_depth": 10,
        "use_label_encoder": False,
    }

    fair_model = xgboost_fair_training(train, y_train, **fair_params)
    accurate_model = xgb.train(accurate_params, dtrain)

    results = pd.DataFrame(
        columns=['state', 'fair_model_accuracy', 'fair_model_dpd',
                 'accurate_model_accuracy', 'accurate_model_dpd'])

    for state in STATES:
        try:
            test = pd.read_csv(f"employment_data_{state}.csv")
        except FileNotFoundError:
            test = download_data_for_state(ACSEmployment, state)

        y_test = test['label']
        test.drop(['label'], axis=1, inplace=True)
        dtest = xgb.DMatrix(test, label=y_test)
        fair_predictions = fair_model.predict(dtest) > 0.5
        accurate_predictions = accurate_model.predict(dtest) > 0.5

        fair_model_accuracy = calculate_accuracy(fair_predictions, y_test)
        accurate_model_accuracy = calculate_accuracy(accurate_predictions,
                                                     y_test)

        fair_model_dpd = demographic_parity_difference(fair_predictions,
                                                       test['DIS'])
        accurate_model_dpd = demographic_parity_difference(
            accurate_predictions, test['DIS'])

        result = [state, fair_model_accuracy, fair_model_dpd,
                  accurate_model_accuracy, accurate_model_dpd]

        results.loc[state] = result

        print(result)

    results.to_csv("results.csv")
    print(results.to_html())