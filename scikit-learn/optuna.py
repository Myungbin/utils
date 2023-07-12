import optuna
from optuna import Trial
from optuna.samplers import TPESampler
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error


def objective_xgb(trial: Trial, X, y):
    params = {
        "n_estimators": trial.suggest_int('n_estimators', 500, 4000),
        'max_depth': trial.suggest_int('max_depth', 8, 16),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 300),
        'gamma': trial.suggest_int('gamma', 1, 3),
        'learning_rate': 0.01,
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        'lambda': trial.suggest_loguniform('lambda', 1e-3, 10.0),
        'alpha': trial.suggest_loguniform('alpha', 1e-3, 10.0),
        'subsample': trial.suggest_categorical('subsample', [0.6, 0.7, 0.8, 1.0]),
        'random_state': 1103
    }

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=1103)

    model = XGBRegressor(**params, tree_method='gpu_hist', gpu_id=0)
    xgb = model.fit(X_train, y_train, verbose=False, eval_set=[(X_val, y_val)])
    y_pred = xgb.predict(X_val)
    score = mean_absolute_error(y_val, y_pred)
    return score


if __name__ == "__main__":
    study = optuna.create_study(direction='minimize', sampler=TPESampler())
    study.optimize(lambda trial: objective_xgb(trial, X, y), n_trials=100)
    print('Best trial: score {},\nparams {}'.format(
        study.best_trial.value, study.best_trial.params))

    params = study.best_trial.params
    xgb_model = XGBRegressor(**params, tree_method='gpu_hist', gpu_id=0).fit(X_train, y_train)
    y_pred = xgb_model.predict(test)
