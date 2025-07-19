#!/usr/bin/env python3
# visualization/visualize_metrics.py

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, auc,
    confusion_matrix, precision_recall_curve
)
from sklearn.calibration import calibration_curve

def main():
    # Cargar predicciones
    df = pd.read_csv('prediccion_prestamos.csv')
    y_true = df['loan_status_true']
    y_pred = df['loan_status_pred']
    y_proba = df['prob_fully_paid (%)'] / 100.0

    # 1. Curva ROC
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve (AUC = {roc_auc:.4f})')
    plt.savefig('roc_curve.png')
    plt.close()
    print("✅ roc_curve.png guardada")

    # 2. Matriz de Confusión
    cm = confusion_matrix(y_true, y_pred)
    plt.figure()
    plt.imshow(cm, interpolation='nearest')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.colorbar()
    tick_marks = [0, 1]
    plt.xticks(tick_marks, ['Charged Off', 'Fully Paid'], rotation=45)
    plt.yticks(tick_marks, ['Charged Off', 'Fully Paid'])
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()
    print("✅ confusion_matrix.png guardada")

    # 3. Curva Precisión-Recall
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.savefig('precision_recall_curve.png')
    plt.close()
    print("✅ precision_recall_curve.png guardada")

    # 4. Curva de Calibración
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10)
    plt.figure()
    plt.plot(prob_pred, prob_true, marker='o')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Calibration Curve')
    plt.savefig('calibration_curve.png')
    plt.close()
    print("✅ calibration_curve.png guardada")

    # 5. Histograma de Probabilidades
    plt.figure()
    plt.hist([y_proba[y_true == 0], y_proba[y_true == 1]], bins=10, stacked=True)
    plt.xlabel('Predicted Probability of Fully Paid')
    plt.ylabel('Count')
    plt.title('Probability Distribution by Class')
    plt.legend(['Charged Off', 'Fully Paid'])
    plt.savefig('prob_distribution.png')
    plt.close()
    print("✅ prob_distribution.png guardada")

if __name__ == "__main__":
    main()
