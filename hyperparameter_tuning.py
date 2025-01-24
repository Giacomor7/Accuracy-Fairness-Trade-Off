import numpy as np
import optuna
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


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


    # Create a study (maximise accuracy / minimise DPD)
    study = optuna.create_study(direction="minimize")
    # Optimize the study against functions above
    study.optimize(random_forest_fairness_objective, n_trials=50)

    # Print the best parameters and score
    print("Best parameters:", study.best_params)
    print("Best value:", study.best_value)
