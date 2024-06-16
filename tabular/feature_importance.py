import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

model = RandomForestClassifier()
model.fit(X.iloc[0:1000, :], y[0:1000])

# Get feature importances
importances = model.feature_importances_
feature_names = X.columns

# Sort and plot feature importances
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), feature_names[indices], rotation=90)
plt.xlim([-1, X.shape[1]])
plt.show()
import_df = pd.DataFrame(list(zip(feature_names, importances)), 
                         columns=['Names', 'Scores']).sort_values(by=['Scores'], ascending=False)
import_df.head()