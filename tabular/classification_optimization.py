def catboost_params_multiclass(trial):
    return {
        'iterations': trial.suggest_int('iterations', 100, 10000),
        'depth': trial.suggest_int('depth', 1, 16),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-4, 1e-1),
        'l2_leaf_reg': trial.suggest_loguniform('l2_leaf_reg', 1e-4, 10.0),
        'border_count': trial.suggest_int('border_count', 1, 255),
        'bagging_temperature': trial.suggest_loguniform('bagging_temperature', 0.01, 10.0),
        'random_strength': trial.suggest_loguniform('random_strength', 0.01, 10.0),
        'rsm': trial.suggest_uniform('rsm', 0.1, 1.0),
        'leaf_estimation_iterations': trial.suggest_int('leaf_estimation_iterations', 1, 20),
        'grow_policy': trial.suggest_categorical('grow_policy', ['SymmetricTree', 'Depthwise', 'Lossguide']),
        'bootstrap_type': trial.suggest_categorical('bootstrap_type', ['Bayesian', 'Bernoulli', 'MVS']),
        'scale_pos_weight': trial.suggest_loguniform('scale_pos_weight', 0.01, 10.0),
        'loss_function': 'MultiClass'  # or 'Logloss' for binary
    }


def lgbm_params_multiclass(trial, num_class):
    return {
        'num_leaves': trial.suggest_int('num_leaves', 2, 256),
        'max_depth': trial.suggest_int('max_depth', -1, 128),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-4, 1e-1),
        'n_estimators': trial.suggest_int('n_estimators', 100, 10000),
        'min_split_gain': trial.suggest_loguniform('min_split_gain', 1e-4, 10.0),
        'min_child_weight': trial.suggest_loguniform('min_child_weight', 1e-4, 10.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 1, 100),
        'subsample': trial.suggest_uniform('subsample', 0.1, 1.0),
        'subsample_freq': trial.suggest_int('subsample_freq', 0, 10),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.1, 1.0),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-4, 10.0),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-4, 10.0),
        'scale_pos_weight': trial.suggest_loguniform('scale_pos_weight', 0.01, 10.0),
        'objective': 'multiclass', # or 'binary' forr binary
        'num_class': num_class
    }


def xgb_params_multiclass(trial, num_class):
    return {
        'n_estimators': trial.suggest_int('n_estimators', 100, 10000),
        'max_depth': trial.suggest_int('max_depth', 1, 16),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-4, 1e-1),
        'subsample': trial.suggest_uniform('subsample', 0.1, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.1, 1.0),
        'colsample_bylevel': trial.suggest_uniform('colsample_bylevel', 0.1, 1.0),
        'colsample_bynode': trial.suggest_uniform('colsample_bynode', 0.1, 1.0),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-4, 10.0),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-4, 10.0),
        'min_child_weight': trial.suggest_loguniform('min_child_weight', 1e-4, 10.0),
        'gamma': trial.suggest_loguniform('gamma', 1e-4, 10.0),
        'scale_pos_weight': trial.suggest_loguniform('scale_pos_weight', 0.01, 10.0),
        'objective': 'multi:softprob',  # or 'binary:logistic' forr binary
        'num_class': num_class
    }

def objective_catboost_multiclass_cv(trial):
    params = catboost_params_multiclass(trial)
    model = CatBoostRegressor(**params, silent=True)
    score = cross_val_score(model, X, y, cv=5, scoring='neg_log_loss')
    log_loss = -score.mean()
    return log_loss

study_catboost = optuna.create_study(direction='minimize')
study_catboost.optimize(objective_catboost_multiclass_cv, n_trials=100)
print("Best CatBoost params:", study_catboost.best_params)
