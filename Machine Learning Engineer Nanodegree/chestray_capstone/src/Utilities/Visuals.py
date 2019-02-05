import numpy as np 
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

def draw_roc_auc(title, pred_labels, actual_Y, predicted_Y):
    fig, c_ax = plt.subplots(1,1, figsize = (10, 10))
    for (idx, c_label) in enumerate(pred_labels):
        fpr, tpr, thresholds = roc_curve(actual_Y[:,idx].astype(int), predicted_Y[:,idx])
        c_ax.plot(fpr, tpr, label = '%s (AUC:%0.2f)'  % (c_label, auc(fpr, tpr)))
    c_ax.legend()
    c_ax.set_xlabel('False Positive Rate')
    c_ax.set_ylabel('True Positive Rate')
    c_ax.set_title(title)