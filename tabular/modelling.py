class Modelling:
    def __init__(self, model: dict, X, y, scaler):
        self.model = model
        self.X = X
        self.y = y
        self.scaler = scaler
        
    def preprocess(self):
        X_scaled = self.scaler.fit_transform(self.X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, self.y, 
                                                            test_size=0.2, 
                                                            random_state=24)
        return X_train, X_test, y_train, y_test
    
    def plot(self):
        X_train, X_test, y_train, y_test = self.preprocess()
        names = []
        accuracy_list = []
        precision_list = []
        recall_list = []
        f1_list = []
        roc_auc_list = []
        
        for name, model in self.model.items():
            names.append(name)
            print(f"Training model: {name}")
            model.fit(X_train, y_train)
            
            # Using predict for most metrics
            predictions = model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            precision = precision_score(y_test, predictions, average='macro')
            recall = recall_score(y_test, predictions, average='macro')
            f1 = f1_score(y_test, predictions, average='macro')
            
            # Using predict_proba for ROC AUC
            if hasattr(model, "predict_proba"):
                prob_predictions = model.predict_proba(X_test)
                roc_auc = roc_auc_score(y_test, prob_predictions, multi_class='ovr')
            else:
                roc_auc = np.nan  # In case the model does not have predict_proba
            
            accuracy_list.append(accuracy)
            precision_list.append(precision)
            recall_list.append(recall)
            f1_list.append(f1)
            roc_auc_list.append(roc_auc)
        
        # Creating DataFrame to store the metrics
        df = pd.DataFrame({
            'model': names,
            'accuracy': accuracy_list,
            'precision': precision_list,
            'recall': recall_list,
            'f1_score': f1_list,
            'roc_auc': roc_auc_list
        })
        
        # Plotting the metrics
        fig, ax = plt.subplots(3, 2, figsize=(15, 15))
        
        sns.barplot(x='model', y='accuracy', data=df, ax=ax[0, 0])
        ax[0, 0].set_title('Accuracy')
        
        sns.barplot(x='model', y='precision', data=df, ax=ax[0, 1])
        ax[0, 1].set_title('Precision')
        
        sns.barplot(x='model', y='recall', data=df, ax=ax[1, 0])
        ax[1, 0].set_title('Recall')
        
        sns.barplot(x='model', y='f1_score', data=df, ax=ax[1, 1])
        ax[1, 1].set_title('F1 Score')
        
        sns.barplot(x='model', y='roc_auc', data=df, ax=ax[2, 0])
        ax[2, 0].set_title('ROC AUC')
        
        plt.tight_layout()
        plt.show()
        
        return df


model = {
    'XGB': xgb.XGBClassifier(),
    'LightGBM': lgb.LGBMClassifier(),
    'CatBoost': cb.CatBoostClassifier(verbose=0),
    'HistBoost': HistGradientBoostingClassifier()
}

scaler = StandardScaler()

model_testing = Modelling(model=model, X=X, y=y, scaler=scaler)
results = model_testing.plot()
print(results)