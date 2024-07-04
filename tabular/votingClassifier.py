from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import optuna

# Define your individual models
model_rf = RandomForestClassifier()
model_gb = GradientBoostingClassifier()
model_svc = SVC(probability=True)  # SVC needs probability=True for VotingClassifier

models = [('rf', model_rf), ('gb', model_gb), ('svc', model_svc)]

def objective(trial):
    # Define parameters to optimize (weights)
    w_rf = trial.suggest_float('w_rf', 0.0, 1.0)
    w_gb = trial.suggest_float('w_gb', 0.0, 1.0)
    w_svc = trial.suggest_float('w_svc', 0.0, 1.0)
    
    # Create VotingClassifier with specified weights
    voting_clf = VotingClassifier(estimators=models, voting='soft', 
                                  weights=[w_rf, w_gb, w_svc])

    # Evaluate the classifier using cross-validation
    cv_scores = cross_val_score(voting_clf, X, y, cv=3, scoring='roc_auc')
    avg_score = cv_scores.mean()
    
    return avg_score


# Create an Optuna study
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

# Print the best trial
best_trial = study.best_trial
print('Best ROC AUC:', best_trial.value)
print('Best weights:', best_trial.params)

best_weights = study.best_trial.params
print('Best weights:', best_weights)

best_w_rf = best_weights['w_rf']
best_w_gb = best_weights['w_gb']
best_w_svc = best_weights['w_svc']

best_voting_clf = VotingClassifier(estimators=models, voting='soft', weights=[best_w_rf, best_w_gb, best_w_svc])
best_voting_clf.fit(X_train, y_train)