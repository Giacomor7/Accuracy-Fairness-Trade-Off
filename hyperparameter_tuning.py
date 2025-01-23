import optuna
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score


def hyperparameter_tuning(data):
    x_train, x_valid = train_test_split(data, test_size=0.3)
    y_train = x_train['label']
    y_valid = x_valid['label']

    x_train.drop(['label'], axis=1, inplace=True)
    x_valid.drop(['label'], axis=1, inplace=True)

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

    # Create a study
    study = optuna.create_study(direction="maximize")
    # Optimize the study
    study.optimize(xg_boost_accuracy_objective, n_trials=50)

    # Print the best parameters and score
    print("Best parameters:", study.best_params)
    print("Best accuracy:", study.best_value)
