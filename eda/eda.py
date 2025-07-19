# 1. Cargar archivo desde local
from google.colab import files
uploaded = files.upload()

# 2. Cargar archivo CSV en un DataFrame
import pandas as pd
df = pd.read_csv("balanced_dataset.csv")

# 3. Ver estructura general del DataFrame
print("\n--- Info general del DataFrame ---")
print(df.info())

# 4. Mostrar todas las filas sin recortes
pd.set_option('display.max_rows', None)

# 5. Ver valores nulos por columna
print("\n--- Valores nulos por columna ---")
print(df.isnull().sum())

# 6. Ver filas duplicadas
print("\n--- Total de filas duplicadas ---")
print(df.duplicated().sum())

# 7. Ver valores únicos por columna
print("\n--- Valores únicos por columna ---")
print(df.nunique())

# 8. Estadísticas descriptivas (solo para variables numéricas)
print("\n--- Estadísticas descriptivas ---")
print(df.describe())

# 9. Instalar y generar un reporte HTML de EDA (pandas-profiling)
!pip install -q ydata-profiling

from ydata_profiling import ProfileReport
profile = ProfileReport(df, title="Reporte EDA - dataset_100k", explorative=True)

# 10. Mostrar reporte en el notebook
profile.to_notebook_iframe()

# 11. (Opcional) Guardar el reporte en HTML
profile.to_file("reporte_eda_dataset.html")
