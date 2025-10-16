import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Miss', 'Beat'], yticklabels=['Miss', 'Beat'], ax=ax)
    ax.set_title('Confusion Matrix (Test Set)')
    return fig

def plot_feature_importance(features, importances):
    fi_df = pd.DataFrame({'feature': features, 'importance': importances}).sort_values('importance', ascending=False)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=fi_df, ax=ax, palette='viridis')
    ax.set_title('Most Predictive Linguistic Features')
    return fig

