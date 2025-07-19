#!/usr/bin/env python3
# eda/eda.py

import pandas as pd
from ydata_profiling import ProfileReport

def main():
    # 1. Carga el dataset
    df = pd.read_csv("balanced_dataset.csv")

    # 2. Info rápida
    print("\n--- Info general del DataFrame ---")
    df.info()

    # 3. Valores nulos
    print("\n--- Valores nulos por columna ---")
    print(df.isnull().sum())

    # 4. Filas duplicadas
    print("\n--- Total de filas duplicadas ---")
    print(df.duplicated().sum())

    # 5. Valores únicos por columna
    print("\n--- Valores únicos por columna ---")
    print(df.nunique())

    # 6. Estadísticas descriptivas
    print("\n--- Estadísticas descriptivas ---")
    print(df.describe())

    # 7. Genera el reporte HTML
    profile = ProfileReport(
        df,
        title="Reporte EDA - balanced_dataset",
        explorative=True
    )
    profile.to_file("reporte_eda_dataset.html")
    print("\n✅ Reporte EDA guardado en 'reporte_eda_dataset.html'")

if __name__ == "__main__":
    main()
