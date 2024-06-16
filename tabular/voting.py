from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import VotingClassifier
import xgboost as xgb
import catboost as cb
import lightgbm as lgb

# Define the best hyperparameters for each model
xgb_params = {
    'n_estimators': 362,
    'max_depth': 6,
    'learning_rate': 0.2913829109037736,
    'subsample': 0.7089783313312756,
    'colsample_bytree': 0.9431404325494228,
    'gamma': 1.7558727716988485,
    'min_child_weight': 3,
    'use_label_encoder': False,
    'objective': 'multi:softprob',
    'num_class': 3
}

lgb_params = {
    'n_estimators': 263,
    'max_depth': 4,
    'learning_rate': 0.22543342013256146,
    'subsample': 0.6309810304198402,
    'colsample_bytree': 0.7264169869604262,
    'num_leaves': 105,
    'min_child_weight': 6.1012464211389466,
    'reg_alpha': 0.2549843049801755,
    'reg_lambda': 0.7477662021426192,
    'objective': 'multiclass',
    'num_class': 3,
    'verbose': -1
}

cat_params = {
    'iterations': 498,
    'depth': 5,
    'learning_rate': 0.2919233590114165,
    'l2_leaf_reg': 1.754246919249645,
    'border_count': 107,
    'random_strength': 0.5377214107717403,
    'bagging_temperature': 0.23468124747130156,
    'od_type': 'Iter',
    'od_wait': 37,
    'loss_function': 'MultiClass'
}

# Prepare your data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the models with the best hyperparameters
xgb_model = xgb.XGBClassifier(**xgb_params)
lgb_model = lgb.LGBMClassifier(**lgb_params)
cat_model = cb.CatBoostClassifier(**cat_params, verbose=0)

# Create the voting classifier
voting_clf = VotingClassifier(
    estimators=[
        ('xgb', xgb_model),
        ('lgb', lgb_model),
        ('cat', cat_model)
    ],
    voting='soft'  # Use 'hard' for majority voting
)

# Train the voting classifier
voting_clf.fit(X_train, y_train)

# Predict on the validation set
y_pred = voting_clf.predict(X_val)

# Evaluate the performance
accuracy = accuracy_score(y_val, y_pred)
print(f'Voting Classifier Accuracy: {accuracy:.4f}')