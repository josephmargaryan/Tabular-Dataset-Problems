# Define categorical and continuous features
cat_feat = [...]  # List of categorical feature names
continuous_feat = [...]  # List of continuous feature names

# Preprocessing steps
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=24)

category_encoder = OrdinalEncoder(
    categories='auto',
    dtype=int,
    handle_unknown='use_encoded_value',
    unknown_value=-2,
    encoded_missing_value=-1,
)

X_train[cat_feat] = category_encoder.fit_transform(X_train[cat_feat])
X_test[cat_feat] = category_encoder.transform(X_test[cat_feat])

sc = StandardScaler()
X_train[continuous_feat] = sc.fit_transform(X_train[continuous_feat])
X_test[continuous_feat] = sc.transform(X_test[continuous_feat])

# Combine train and test sets for cross-validation
X = pd.concat([X_train, X_test], axis=0)
y = pd.concat([y_train, y_test], axis=0)

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 2, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'early_stopping_rounds': 50,  # Setting early stopping rounds here
        'eval_metric': 'auc'  # Setting evaluation metric here
    }

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    partial_auc_scores = []

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        over = SMOTE(sampling_strategy=0.1)
        under = RandomUnderSampler(sampling_strategy=0.5)
        steps = [('o', over), ('u', under)]
        pipeline = Pipeline(steps=steps)
        X_train, y_train = pipeline.fit_resample(X_train, y_train)

        model = xgb.XGBClassifier(**params, objective='binary:logistic', use_label_encoder=False)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        preds_proba = model.predict_proba(X_test)[:, 1]

        # Create dataframes for the custom metric
        solution_df = pd.DataFrame(data={'row_id': test_index, 'target': y_test})
        submission_df = pd.DataFrame(data={'row_id': test_index, 'prediction': preds_proba})

        partial_auc = score(solution_df, submission_df, row_id_column_name='row_id')
        partial_auc_scores.append(partial_auc)

    return np.mean(partial_auc_scores)

# Example usage with an Optuna study
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

print('Best trial:')
trial = study.best_trial

print('Value: ', trial.value)
print('Params: ')
for key, value in trial.params.items():
    print(f'    {key}: {value}')