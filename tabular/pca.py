from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt

def scatter_plot_two_features(data, component1, component2, class_label):
    sns.scatterplot(data=data, x=component1, y=component2, hue=class_label, palette='viridis')
    plt.title(f'Scatter Plot of {component1} vs {component2}')
    plt.xlabel(f'Principal Component {component1}')
    plt.ylabel(f'Principal Component {component2}')
    plt.show()

# Assuming X_pca contains your PCA-transformed data and y contains your target labels
# Replace 'class_label' with the actual column name or index of your target labels in 'data'
scatter_plot_two_features(data=pd.DataFrame(X_pca, columns=['PC1', 'PC2']), 
                          component1='PC1', 
                          component2='PC2', 
                          class_label=y)