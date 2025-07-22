# --- 1. IMPORTAR LIBRERÍAS NECESARIAS ---
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)

# --- 2. CARGAR DATOS ENTRENAMIENTO Y PRUEBA ---
df_train = pd.read_csv('train2_prestamos.csv')
df_test  = pd.read_csv('test2_prestamos.csv')

# --- 3. SEPARAR VARIABLES ---
X_train  = df_train.drop(columns=['id', 'loan_status'])
y_train  = df_train['loan_status'].astype(int)
X_test   = df_test.drop(columns=['id', 'loan_status'])
ids_test = df_test['id']
y_true   = df_test['loan_status'].astype(int)

# --- 4. ENTRENAR MODELO XGBoost ---
modelo = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)
modelo.fit(X_train, y_train)

# --- 5. GENERAR PREDICCIONES Y PROBABILIDADES ---
y_pred  = modelo.predict(X_test)
y_proba = modelo.predict_proba(X_test)[:, 1]

# --- 6. GUARDAR PREDICCIONES ---
df_resultados = X_test.copy()
df_resultados['id']                    = ids_test.values
df_resultados['loan_status_pred']     = y_pred
df_resultados['prob_charged_off (%)'] = (100 * (1 - y_proba)).round(1)
df_resultados['prob_fully_paid (%)']  = (100 * y_proba).round(1)
df_resultados['loan_status_true']     = y_true.values
df_resultados.to_csv('prediccion_prestamos.csv', index=False)

# --- 7. GUARDAR MÉTRICAS ---
metrics = {
    "accuracy": round(accuracy_score(y_true, y_pred), 4),
    "precision": round(precision_score(y_true, y_pred), 4),
    "recall": round(recall_score(y_true, y_pred), 4),
    "f1_score": round(f1_score(y_true, y_pred), 4),
    "roc_auc": round(roc_auc_score(y_true, y_proba), 4),
    "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
}
# Guardar en JSON para informes del pipeline
import json
with open('metricas_xgboost.json', 'w') as f:
    json.dump(metrics, f, indent=4)

# --- 8. OPCIONAL: Imprimir resumen en consola ---
print("\n--- Métricas en Test ---")
print(json.dumps(metrics, indent=4))
print("\nClassification Report:\n")
print(classification_report(
    y_true,
    y_pred,
    target_names=['Charged Off', 'Fully Paid'],
    digits=4
))
