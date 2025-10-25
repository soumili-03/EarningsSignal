import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

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
def plot_classification_report(y_true, y_pred):
    report = classification_report(y_true, y_pred, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    
    # Show precision, recall, f1-score for classes "Misses" (0) and "Beat" (1)
    metrics = ['precision', 'recall', 'f1-score']
    class_map = {'0.0': 'Misses', '1.0': 'Beat'}
    classes = ['0.0', '1.0']
    
    fig, ax = plt.subplots(figsize=(6, 4))
    bar_width = 0.25
    indices = np.arange(len(metrics))
    
    for i, cls in enumerate(classes):
        vals = [df_report.loc[cls, m] for m in metrics]
        ax.bar(indices + i*bar_width, vals, bar_width, label=class_map[cls])
    
    ax.set_xticks(indices + bar_width / 2)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1)
    ax.set_ylabel('Score')
    ax.set_title('Classification Metrics by Class')
    ax.legend()
    
    return fig
