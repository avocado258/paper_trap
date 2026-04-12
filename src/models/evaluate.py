import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y_true, y_pred, save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    
    labels = ['Real', 'Fake']
    
    tn, fp, fn, tp = cm.ravel()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    accuracy  = (tp + tn) / cm.sum()
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    fig, ax = plt.subplots(figsize=(7, 6))
    
    sns.heatmap(
        cm,
        annot=False,
        fmt='d',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        linewidths=0.5,
        linecolor='gray',
        ax=ax
    )
    
    cell_annotations = [
        (0, 0, tn,  'TN', 'Correctly cleared\nreal papers',     'darkgreen'),
        (0, 1, fp,  'FP', 'Real paper wrongly\naccused (❗)',    'darkred'),
        (1, 0, fn,  'FN', 'Fake paper\nmissed',                 'darkorange'),
        (1, 1, tp,  'TP', 'Correctly caught\nfake papers',      'darkgreen'),
    ]
    
    for row, col, val, tag, desc, color in cell_annotations:
        ax.text(col + 0.5, row + 0.30, str(val),
                ha='center', va='center', fontsize=22, fontweight='bold', color=color)
        ax.text(col + 0.5, row + 0.58, tag,
                ha='center', va='center', fontsize=11, fontweight='bold',
                color='white',
                bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.75, edgecolor='none'))
        ax.text(col + 0.5, row + 0.82, desc,
                ha='center', va='center', fontsize=8, color='dimgray')
    
    ax.set_xlabel('Predicted Class', fontsize=12, labelpad=10)
    ax.set_ylabel('Actual Class',    fontsize=12, labelpad=10)
    ax.set_title('PaperTrap — Confusion Matrix', fontsize=14, fontweight='bold', pad=15)
    
    metrics_text = (
        f'Precision: {precision:.2%}  |  '
        f'Recall: {recall:.2%}  |  '
        f'Accuracy: {accuracy:.2%}  |  '
        f'F1: {f1:.2%}'
    )
    fig.text(0.5, 0.01, metrics_text, ha='center', fontsize=9, color='dimgray')
    
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()
    return cm