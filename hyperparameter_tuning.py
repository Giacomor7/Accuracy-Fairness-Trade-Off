import numpy as np
import optuna
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from fair_training import xgboost_fair_training, random_forest_fair_training


def hyperparameter_tuning(data):
    x_train, x_valid = train_test_split(data, test_size=0.3)
    y_train = x_train['label']
    y_valid = x_valid['label']

    x_train.drop(['label'], axis=1, inplace=True)
    x_valid.drop(['label'], axis=1, inplace=True)

    sa_valid = x_valid['DIS']

    def demographic_parity_difference(y_pred, sa):
        """
        Calculates the Demographic Parity Difference (DPD).
        """
        sa_0 = y_pred[sa == 1]
        sa_1 = y_pred[sa == 2]
        rate_0 = np.mean(sa_0)
        rate_1 = np.mean(sa_1)
        return abs(rate_0 - rate_1)

    def xg_boost_accuracy_objective(trial):
        # Suggest values for eta and max_depth
        eta = trial.suggest_float("eta", 0.01, 0.3,
                                  log=True)  # Log scale for learning rate
        max_depth = trial.suggest_int("max_depth", 3, 10)

        # Define XGBoost parameters
        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "eta": eta,
            "max_depth": max_depth,
            "use_label_encoder": False,
        }

        # Train XGBoost model
        dtrain = xgb.DMatrix(x_train, label=y_train)
        dvalid = xgb.DMatrix(x_valid, label=y_valid)
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=100,
            evals=[(dvalid, "validation")],
            early_stopping_rounds=10,
            verbose_eval=False,
        )

        # Predict and calculate accuracy
        preds = model.predict(dvalid)
        preds_binary = (preds > 0.5).astype(int)
        accuracy = accuracy_score(y_valid, preds_binary)
        return accuracy


    def xg_boost_fairness_objective(trial):
        """
        Objective function to minimize DPD using Optuna.
        """
        # Hyperparameter suggestions
        eta = trial.suggest_float("eta", 0.01, 0.3, log=True)  # Learning rate
        max_depth = trial.suggest_int("max_depth", 3, 10)  # Tree depth

        # XGBoost parameters
        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "eta": eta,
            "max_depth": max_depth,
            "use_label_encoder": False,
        }

        # Train XGBoost model
        dtrain = xgb.DMatrix(x_train, label=y_train)
        dvalid = xgb.DMatrix(x_valid, label=y_valid)
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=100,
            evals=[(dvalid, "validation")],
            early_stopping_rounds=10,
            verbose_eval=False,
        )

        # Predictions for the validation set
        y_pred = model.predict(dvalid) > 0.5  # Binary predictions

        # Calculate demographic parity difference
        dpd = demographic_parity_difference(y_pred, sa_valid)
        return dpd


    def random_forest_accuracy_objective(trial):
        # Suggest hyperparameter values
        n_estimators = trial.suggest_int("n_estimators", 50, 500)
        max_depth = trial.suggest_int("max_depth", 5, 30)

        # Create and train the Random Forest model
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )
        model.fit(x_train, y_train)

        # Predict and calculate accuracy
        y_pred = model.predict(x_valid)
        accuracy = accuracy_score(y_valid, y_pred)
        return accuracy


    def random_forest_fairness_objective(trial):
        """
        Objective function to minimize DPD for Random Forest.
        """
        # Hyperparameter suggestions
        n_estimators = trial.suggest_int("n_estimators", 50, 300, step=50)
        max_depth = trial.suggest_int("max_depth", 3, 20)

        # Train Random Forest model
        model = RandomForestClassifier(n_estimators=n_estimators,
                                       max_depth=max_depth, random_state=42)
        model.fit(x_train, y_train)

        # Predictions
        y_pred = model.predict(x_valid)

        # Calculate DPD
        dpd = demographic_parity_difference(y_pred, sa_valid)

        # Handle NaN values
        if np.isnan(dpd):
            return 1.0  # Large penalty for invalid trials

        return dpd


    def fair_xgboost_accuracy_objective(trial):
        # Suggest values for eta and max_depth
        eta = trial.suggest_float("eta", 0.01, 0.3,
                                  log=True)  # Log scale for learning rate
        max_depth = trial.suggest_int("max_depth", 3, 10)

        # Define XGBoost parameters
        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "eta": eta,
            "max_depth": max_depth,
            "use_label_encoder": False,
        }

        dvalid = xgb.DMatrix(x_valid, label=y_valid)
        model = xgboost_fair_training(x_train, y_train, dvalid=dvalid,
                                      **params)

        # Predict and calculate accuracy
        preds = model.predict(dvalid)
        preds_binary = (preds > 0.5).astype(int)
        accuracy = accuracy_score(y_valid, preds_binary)
        return accuracy


    def fair_random_forest_accuracy_objective(trial):
        # Suggest hyperparameter values
        n_estimators = trial.suggest_int("n_estimators", 50, 500)
        max_depth = trial.suggest_int("max_depth", 5, 30)

        params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "random_state": 42,
        }
        model = random_forest_fair_training(x_train, y_train, **params)

        # Predict and calculate accuracy
        y_pred = model.predict(x_valid)
        accuracy = accuracy_score(y_valid, y_pred)
        return accuracy


    def fair_random_forest_fairness_objective(trial):
        # Suggest hyperparameter values
        n_estimators = trial.suggest_int("n_estimators", 50, 500)
        max_depth = trial.suggest_int("max_depth", 5, 30)

        params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "random_state": 42,
        }
        model = random_forest_fair_training(x_train, y_train, **params)

        # Predictions
        y_pred = model.predict(x_valid)

        # Calculate DPD
        dpd = demographic_parity_difference(y_pred, sa_valid)

        # Handle NaN values
        if np.isnan(dpd):
            return 1.0  # Large penalty for invalid trials

        return dpd


    def fair_xgboost_fairness_objective(trial):
        # Suggest values for eta and max_depth
        eta = trial.suggest_float("eta", 0.01, 0.3,
                                  log=True)  # Log scale for learning rate
        max_depth = trial.suggest_int("max_depth", 3, 10)

        # Define XGBoost parameters
        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "eta": eta,
            "max_depth": max_depth,
            "use_label_encoder": False,
        }

        dvalid = xgb.DMatrix(x_valid, label=y_valid)
        model = xgboost_fair_training(x_train, y_train, dvalid=dvalid,
                                      **params)

        # Predictions for the validation set
        y_pred = model.predict(dvalid) > 0.5  # Binary predictions

        # Calculate demographic parity difference
        dpd = demographic_parity_difference(y_pred, sa_valid)
        return dpd


    def fair_xgboost_balanced_objective(trial):
        # Suggest values for eta and max_depth
        eta = trial.suggest_float("eta", 0.01, 0.3,
                                  log=True)  # Log scale for learning rate
        max_depth = trial.suggest_int("max_depth", 3, 10)

        # Define XGBoost parameters
        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "eta": eta,
            "max_depth": max_depth,
            "use_label_encoder": False,
        }

        dvalid = xgb.DMatrix(x_valid, label=y_valid)
        model = xgboost_fair_training(x_train, y_train, dvalid=dvalid,
                                      **params)

        # Predictions for the validation set
        y_pred = model.predict(dvalid) > 0.5  # Binary predictions

        # Calculate demographic parity difference
        dpd = demographic_parity_difference(y_pred, sa_valid)

        # Predict and calculate accuracy
        preds = model.predict(dvalid)
        preds_binary = (preds > 0.5).astype(int)
        accuracy = accuracy_score(y_valid, preds_binary)
        return accuracy - dpd


    def fair_random_forest_balanced_objective(trial):
        # Suggest hyperparameter values
        n_estimators = trial.suggest_int("n_estimators", 50, 500)
        max_depth = trial.suggest_int("max_depth", 5, 30)

        params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "random_state": 42,
        }
        model = random_forest_fair_training(x_train, y_train, **params)

        # Predictions
        y_pred = model.predict(x_valid)

        # Calculate DPD
        dpd = demographic_parity_difference(y_pred, sa_valid)

        # Handle NaN values
        if np.isnan(dpd):
            return 1.0  # Large penalty for invalid trials

        # Calculate accuracy
        accuracy = accuracy_score(y_valid, y_pred)

        return accuracy - dpd


    # Create a study (maximize accuracy / minimize DPD)
    study = optuna.create_study(direction="maximize")
    # Optimize the study against functions above
    study.optimize(fair_random_forest_balanced_objective, n_trials=50)

    # Print the best parameters and score
    print("Best parameters:", study.best_params)
    print("Best value:", study.best_value)

"""
Best hyperparameters:

Tuned for accuracy:
XGB:
Best parameters: {'eta': 0.12355187904055383, 'max_depth': 10}
RF:
Best parameters: {'n_estimators': 153, 'max_depth': 13}

For fairness:
XGB:
Best parameters: {'eta': 0.2932425687600753, 'max_depth': 9}
RF:
Best parameters: {'n_estimators': 100, 'max_depth': 20}

Fair models (after reweighing):

Tuned for accuracy:
XGB:
Best parameters: {'eta': 0.07624482588186494, 'max_depth': 10}
RF:
Best parameters: {'n_estimators': 397, 'max_depth': 18}

Tuned for fairness:
RF:
Best parameters: {'n_estimators': 354, 'max_depth': 9}
XGB:
Best parameters: {'eta': 0.016117651313141638, 'max_depth': 8}

Tuned for balance:
XGB:
Best parameters: {'eta': 0.2908419486944403, 'max_depth': 4}
RF:
Best parameters: {'n_estimators': 286, 'max_depth': 10}
"""
