#!/usr/bin/env python3
# training/train_model.py

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score
)

def main():
    # 2. Cargar datos
    df_train = pd.read_csv('train2_prestamos.csv')
    df_test  = pd.read_csv('test2_prestamos.csv')

    # 3. Preparar variables
    X_train   = df_train.drop(columns=['id','loan_status'])
    y_train   = df_train['loan_status'].astype(int)
    X_test    = df_test.drop(columns=['id','loan_status'])
    y_true    = df_test['loan_status'].astype(int)
    ids_test  = df_test['id']

    # 4. Entrenar modelo
    modelo = LogisticRegression(max_iter=1000, random_state=42)
    modelo.fit(X_train, y_train)

    # 5. Predecir
    y_pred  = modelo.predict(X_test)
    y_proba = modelo.predict_proba(X_test)[:,1]

    # 6. DataFrame resultados (incluye todas las columnas de X_test)
    df_res = X_test.copy()
    df_res['id']                   = ids_test.values
    df_res['loan_status_pred']     = y_pred
    df_res['prob_charged_off (%)'] = (100 * (1 - y_proba)).round(1)
    df_res['prob_fully_paid (%)']  = (100 * y_proba).round(1)
    df_res['loan_status_true']     = y_true.values

    # 7. Métricas
    print("\n--- Métricas en Test ---")
    print("Matriz de Confusión:")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification Report:")
    print(classification_report(
        y_true, y_pred,
        target_names=['Charged Off','Fully Paid'],
        digits=4
    ))
    print(f"ROC AUC: {roc_auc_score(y_true, y_proba):.4f}")

    # 8. Guardar resultados
    df_res.to_csv('prediccion_prestamos.csv', index=False)
    print("\n✅ Archivo 'prediccion_prestamos.csv' generado.")

if __name__ == "__main__":
    main()
