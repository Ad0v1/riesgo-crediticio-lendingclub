# ============ IMPORTS ============
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# ============ PASO 0: LECTURA DIRECTA ============
df = pd.read_csv('balanced_dataset.csv', low_memory=False)

# ============ PASO 1: INGENIERÍA DE FECHAS ============
df['last_credit_pull_d'] = pd.to_datetime(df['last_credit_pull_d'], errors='coerce')
df['last_pull_year'] = df['last_credit_pull_d'].dt.year
df['last_pull_month'] = df['last_credit_pull_d'].dt.month
ref = df['last_credit_pull_d'].max()
df['months_since_last_pull'] = ((ref.year - df['last_credit_pull_d'].dt.year) * 12 +
                                (ref.month - df['last_credit_pull_d'].dt.month))

# ============ PASO 2: FILTRADO INICIAL ============
keep_cols = [
    'id', 'loan_status',
    'loan_amnt', 'term', 'int_rate', 'sub_grade', 'emp_length',
    'home_ownership', 'annual_inc', 'verification_status', 'purpose',
    'addr_state', 'dti', 'delinq_2yrs', 'installment', 'collections_12_mths_ex_med',
    'open_acc', 'inq_last_6mths', 'total_acc', 'revol_bal',
    'revol_util', 'total_rev_hi_lim', 'tot_cur_bal', 'tot_coll_amt',
    'pub_rec', 'acc_now_delinq',
    'last_pull_year', 'last_pull_month', 'months_since_last_pull'
]
missing = set(keep_cols) - set(df.columns)
if missing:
    raise ValueError(f"Faltan columnas en el CSV: {missing}")

df = df[keep_cols]
df = df[df['loan_status'].isin(['Fully Paid', 'Charged Off'])]
df.to_csv('step2_filtered_dataset.csv', index=False)
print(f"P2 completado → {df.shape}")

# ============ PASO 3: MARCAR NaN ============
df = df.replace(['', ' ', 'NA', 'N/A', None], np.nan)
df = df.where(pd.notnull(df), np.nan)
df.to_csv('step3_nan_replaced.csv', index=False)
print(f"P3 completado → {df.shape}")

# ============ PASO 4: CONVERSIÓN DE TIPOS ============
cat_cols = [
    'term', 'sub_grade', 'emp_length', 'home_ownership',
    'verification_status', 'purpose', 'addr_state', 'loan_status'
]
for c in cat_cols:
    df[c] = df[c].astype('category')
df.to_csv('step4_types_converted.csv', index=False)
print(f"P4 completado → {df.shape}")

# ============ PASO 5: IMPUTACIÓN ============
num_cols = [
    'loan_amnt', 'int_rate', 'annual_inc', 'dti',
    'delinq_2yrs', 'installment',
    'collections_12_mths_ex_med', 'open_acc', 'inq_last_6mths',
    'total_acc', 'revol_bal', 'revol_util', 'total_rev_hi_lim',
    'tot_cur_bal', 'tot_coll_amt', 'pub_rec', 'acc_now_delinq',
    'last_pull_year', 'last_pull_month', 'months_since_last_pull'
]
for c in num_cols:
    df[c] = df[c].fillna(df[c].median())

cat_impute_cols = [
    'term', 'sub_grade', 'emp_length', 'home_ownership',
    'verification_status', 'purpose', 'addr_state'
]
for c in cat_impute_cols:
    df[c] = df[c].fillna(df[c].mode()[0])

df.to_csv('step5_imputed.csv', index=False)
print(f"P5 completado → {df.shape}")

# ============ PASO 6: SNAPSHOT ============
df.to_csv('step6_snapshot.csv', index=False)
print("P6 snapshot guardado")

# ============ PASO 7: SEPARACIÓN X/y ============
le = LabelEncoder()
y = le.fit_transform(df['loan_status'])
X = df.drop(columns=['loan_status']).copy()
pd.DataFrame(y, columns=['loan_status']).to_csv('step7_y.csv', index=False)
X.to_csv('step7_X.csv', index=False)
print(f"P7 completado → X={X.shape}, y={len(y)}")

# ============ PASO 8: ESTANDARIZACIÓN ============
X = pd.read_csv('step7_X.csv')
num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
if 'id' in num_cols:
    num_cols.remove('id')
scaler = StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])
X.to_csv('step8_X_scaled.csv', index=False)
print(f"P8 completado → escalado de {len(num_cols)} vars")

# ============ PASO 9: CODIFICACIÓN ============
X = pd.read_csv('step8_X_scaled.csv')
X['term']       = LabelEncoder().fit_transform(X['term'])
X['sub_grade']  = LabelEncoder().fit_transform(X['sub_grade'])
X['emp_length'] = LabelEncoder().fit_transform(X['emp_length'])
X = pd.get_dummies(
    X,
    columns=['verification_status', 'home_ownership', 'purpose', 'addr_state'],
    drop_first=False
)
X.to_csv('step9_X_encoded.csv', index=False)
print(f"P9 completado → {X.shape}")

# ============ PASO 13: EXPORT FINAL ============
X = pd.read_csv('step9_X_encoded.csv')
X['loan_status'] = pd.read_csv('step7_y.csv')['loan_status']
X.to_csv('step13_final_dataset.csv', index=False)
X.to_parquet('step13_final_dataset.parquet', index=False)
print("P13 completado → datasets exportados")

# ============ PASO 15: TRAIN/TEST Y ARCHIVOS SOLICITADOS ============
df_final = pd.read_csv('step13_final_dataset.csv')
X = df_final.drop(columns=['loan_status']).copy()
y = df_final['loan_status']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
train = pd.concat([X_train, y_train], axis=1)
test  = pd.concat([X_test, y_test], axis=1)
train.drop(columns=['id']).to_csv('train1_prestamos.csv', index=False)
train.to_csv('train2_prestamos.csv', index=False)
test.drop(columns=['id','loan_status']).to_csv('test1_prestamos.csv', index=False)
test.to_csv('test2_prestamos.csv', index=False)
print("P15 completado → archivos train1/2_prestamos & test1/2_prestamos generados")

