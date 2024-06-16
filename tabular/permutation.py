import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

# Perform permutation importance
perm_importance = permutation_importance(model, X.iloc[0:1000, :], y[0:1000], n_repeats=5, random_state=42)

# Get feature importances
importances = perm_importance.importances_mean
std = perm_importance.importances_std
feature_names = X.columns

# Create DataFrame for feature importances
perm_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance_Mean': importances,
    'Importance_STD': std
})

# Sort DataFrame by importance (ascending)
perm_importance_df = perm_importance_df.sort_values(by='Importance_Mean', ascending=True)

# Plot permutation feature importances
plt.figure(figsize=(10, 6))
plt.barh(range(X.shape[1]), perm_importance_df['Importance_Mean'], xerr=perm_importance_df['Importance_STD'], align='center')
plt.yticks(range(X.shape[1]), perm_importance_df['Feature'])
plt.ylim([-1, X.shape[1]])
plt.xlabel('Permutation Importance')
plt.title('Permutation Feature Importances')
plt.show()

# Display the DataFrame with sorted importances
print("Sorted Permutation Feature Importances:")
print(perm_importance_df.head())

# Filter columns with permutation importance score of 0
irrelevant_features = perm_importance_df.loc[perm_importance_df['Importance_Mean'] == 0, 'Feature'].tolist()

# Drop irrelevant features
X = X.drop(columns=irrelevant_features, axis=1)