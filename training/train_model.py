import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)

# --- 1. CARGAR DATOS DESDE ARCHIVOS LOCALES ---
df_train = pd.read_csv('train2_prestamos.csv')
df_test  = pd.read_csv('test2_prestamos.csv')

# --- 2. PREPARAR VARIABLES ---
X_train  = df_train.drop(columns=['id', 'loan_status'])
y_train  = df_train['loan_status'].astype(int)
X_test   = df_test.drop(columns=['id', 'loan_status'])
ids_test = df_test['id']
y_true   = df_test['loan_status'].astype(int)

# --- 3. ENTRENAR MODELO ---
modelo = LogisticRegression(max_iter=1000, random_state=42)
modelo.fit(X_train, y_train)

# --- 4. PREDICCIONES ---
y_pred  = modelo.predict(X_test)
y_proba = modelo.predict_proba(X_test)[:, 1]

# --- 5. RESULTADOS ---
df_resultados = X_test.copy()
df_resultados['id']                   = ids_test.values
df_resultados['loan_status_pred']     = y_pred
df_resultados['prob_charged_off (%)'] = (100 * (1 - y_proba)).round(1)
df_resultados['prob_fully_paid (%)']  = (100 * y_proba).round(1)
df_resultados['loan_status_true']     = y_true.values

# --- 6. MÉTRICAS ---
conf_mat  = confusion_matrix(y_true, y_pred)
accuracy  = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall    = recall_score(y_true, y_pred)
f1        = f1_score(y_true, y_pred)
roc_auc   = roc_auc_score(y_true, y_proba)

print("--- Métricas en Test ---")
print("Matriz de Confusión:")
print(conf_mat)

print("Classification Report:")
print(classification_report(
    y_true,
    y_pred,
    target_names=['Charged Off', 'Fully Paid'],
    digits=4
))

print(f"ROC AUC: {roc_auc:.4f}")

# --- 7. GUARDAR RESULTADOS ---
df_resultados.to_csv('prediccion_prestamos.csv', index=False)
print("Archivo 'prediccion_prestamos.csv' guardado exitosamente.")
