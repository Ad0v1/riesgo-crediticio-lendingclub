# Proyecto de Predicción de Riesgo Crediticio - LendingClub

Este repositorio implementa un **pipeline de Machine Learning** para predecir el riesgo de incumplimiento de préstamos (loan status) usando datos de LendingClub. El flujo completo abarca:

1. **Preprocesamiento** de datos crudos
2. **Balanceo** de clases para evitar sesgos
3. **Entrenamiento** de un modelo XGBoost optimizado
4. **Evaluación** de desempeño (métricas clásicas y ROC AUC)
5. **Inferencia**: predicción sobre nuevos registros via script de consola o interfaz web

---

## Estructura del repositorio

```
/ (raíz)
├─ data/
│  ├─ raw/                # Datasets originales sin procesar (train.csv, nuevos_registros.csv)
│  └─ processed/          # Conjuntos procesados y balanceados (balanced_dataset.csv)
├─ preprocessing/
│  └─ custom_preprocessing.py  # Función `transformar_dataset_crudo(df)`
├─ training/
│  └─ train_model.py      # Script de entrenamiento y guardado de model_pipeline.pkl
├─ evaluation/
│  └─ evaluate.py         # Cálculo de métricas y generación de reportes (CSV/JSON)
├─ app/
│  ├─ models/             # Modelo serializado `model_pipeline.pkl`
│  ├─ static/css/         # Estilos para la interfaz web
│  ├─ templates/          # Plantillas HTML (index.html)
│  └─ app.py              # Servidor Flask para inferencia web
├─ requirements.txt       # Dependencias Python
└─ README.md              # Documentación de uso de este pipeline
```

---

## Requisitos previos

- Python 3.8+
- pip

Instala las librerías necesarias:

```bash
pip install -r requirements.txt
```

---

## Uso del pipeline

### 1. Preprocesamiento y balanceo

Procesa el dataset crudo y genera el conjunto balanceado:

```bash
python -c "from preprocessing.custom_preprocessing import transformar_dataset_crudo; import pandas as pd

df = pd.read_csv('data/raw/train.csv')
processed = transformar_dataset_crudo(df)
processed.to_csv('data/processed/balanced_dataset.csv', index=False)"
```

### 2. Entrenamiento del modelo

Ejecuta el script de entrenamiento para generar `model_pipeline.pkl` en `training/`:

```bash
python training/train_model.py \
  --input data/processed/balanced_dataset.csv \
  --output training/model_pipeline.pkl
```

### 3. Evaluación del modelo

Calcula métricas y guarda reportes en CSV y JSON:

```bash
python evaluation/evaluate.py \
  --model training/model_pipeline.pkl \
  --test data/processed/balanced_dataset.csv \
  --out_dir evaluation/reports/
```

### 4. Inferencia por consola

Predice sobre un nuevo archivo crudo `nuevos_registros.csv`:

````bash
python -c "
import pandas as pd; from joblib import load

# Carga y transformación
model = load('training/model_pipeline.pkl')
df = pd.read_csv('data/raw/nuevos_registros.csv')
preds = model.predict(df)
probs = model.predict_proba(df)

# Resultado
output = df.copy()
output['Predicción'] = preds
output['Prob_Charged_Off'] = probs[:,0] * 100
output['Prob_Fully_Paid'] = probs[:,1] * 100
output.to_csv('data/processed/predicciones.csv', index=False)
"```

### 5. Inferencia vía web

Arranca el servidor Flask y accede a la interfaz:

```bash
cd app
python app.py
````

- Visita `http://127.0.0.1:5000`
- Sube un archivo CSV crudo y obtén la tabla de predicciones y descarga del CSV completo.

---

## ¿Qué hace este pipeline?

- ``:

  - Transforma columnas de fecha y crea variables derivadas (p.ej. `months_since_last_pull`).
  - Codifica categorías y escala variables numéricas.

- **Balanceo**:

  - Asegura proporción equilibrada entre clases "Fully Paid" y "Charged Off".

- ``:

  - Ajusta un `XGBClassifier` con validación interna y parámetros optimizados.
  - Genera un pipeline completo con preprocesamiento embebido.

- ``:

  - Calcula accuracy, precision, recall, F1-score y ROC AUC.
  - Genera reportes en CSV y JSON para análisis comparativo.

- ``:

  - Integra todo en una API REST simple para subir datos sin procesar y recibir las predicciones.

---

## Contribuciones

1. Crea un *fork* del repositorio.
2. Abre un *pull request* describiendo tus cambios.

---

## Licencia

Este proyecto está bajo la licencia MIT.

