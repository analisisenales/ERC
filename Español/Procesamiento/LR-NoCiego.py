# -*- coding: utf-8 -*-
"""
Regresión Logística con biomarcadores íntegros
Enfoque: Clasificación mediante modelos lineales multiclase utilizando el expediente completo.
"""

import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Paso 1: Carga de datos de los pacientes desde el archivo fuente.
print("Cargando la base de datos completa...")
directorio_actual = os.path.dirname(os.path.abspath(__file__))
archivo_limpio = os.path.join(directorio_actual, 'Unificacion_tablas.csv')

try:
    df = pd.read_csv(archivo_limpio)
except Exception as e:
    print(f"❌ Error al cargar el CSV: {e}")
    raise

# Paso 2: Generación de la "Ground Truth" o variable objetivo mediante el cálculo de eGFR.
df.dropna(subset=['Creatinina_umol_L', 'Edad', 'Genero'], inplace=True)

df['Creatinina_mg_dL'] = df['Creatinina_umol_L'] / 88.4

def calcular_egfr(row):
    scr = row['Creatinina_mg_dL']
    edad = row['Edad']
    sexo = row['Genero']
    kappa = 0.7 if sexo == 2 else 0.9
    alpha = -0.241 if sexo == 2 else -0.302
    mult_sexo = 1.012 if sexo == 2 else 1.0
    return 142 * min(scr/kappa, 1)**alpha * max(scr/kappa, 1)**-1.200 * 0.9938**edad * mult_sexo

df['eGFR_Real'] = df.apply(calcular_egfr, axis=1)

def clasificar_estadio(egfr):
    if egfr >= 60: return "G1-G2 (Sano/Leve)"
    elif egfr >= 30: return "G3 (Moderado)"
    else: return "G4-G5 (Grave)"

df['Target_Estadio'] = df['eGFR_Real'].apply(clasificar_estadio)

# Paso 3: Preparación de datos para Machine Learning.
# Al ser la versión completa, se preservan las variables de cálculo directo dentro del conjunto X.
columnas_excluir = ['ID_Participante', 'Target_Estadio', 'eGFR_Real']

X = df.drop(columns=columnas_excluir)
y = df['Target_Estadio']

# División de los datos: 75% para entrenamiento, 25% para prueba (test).
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# Paso 4: Construcción del Pipeline de preprocesamiento (Imputación y Escalamiento).
print("\n🛠️ Rellenando valores faltantes y escalando variables...")

# A. Imputación de valores faltantes utilizando la mediana de cada característica.
imputador = SimpleImputer(strategy='median')
X_train_imputed = imputador.fit_transform(X_train)
X_test_imputed = imputador.transform(X_test)

# B. Escalamiento estadístico z-score (indispensable para la estabilidad de modelos lineales).
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

# Paso 5: Inicialización y entrenamiento del modelo clasificador Logistic Regression.
print("🧠 Entrenando Regresión Logística con todas las variables...")
# Se especifica max_iter=1000 para garantizar la convergencia del optimizador numérico.
logreg_model = LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000)
logreg_model.fit(X_train_scaled, y_train)

# Paso 6: Evaluación del modelo, cálculo de métricas de rendimiento y visualización de resultados.
y_pred = logreg_model.predict(X_test_scaled)

print("\n" + "="*50)
accuracy = accuracy_score(y_test, y_pred)
print(f"🏆 PRECISIÓN (ACCURACY) Regresión Logística Completa: {accuracy * 100:.2f}%")
print("="*50)
print("\n📋 Reporte Detallado:")
print(classification_report(y_test, y_pred))

# Paso 7: Extracción y consolidación de los coeficientes del modelo (Importancia de Variables).
# Se calcula la media del valor absoluto de los coeficientes a través de todas las clases objetivo.
coeficientes_absolutos = np.abs(logreg_model.coef_).mean(axis=0)

df_importancia = pd.DataFrame({
    'Biomarcador': X.columns, 
    'Importancia': coeficientes_absolutos
})
df_importancia = df_importancia.sort_values(by='Importancia', ascending=False)

# Generación del gráfico de barras para análisis de coeficientes lineales.
plt.figure(figsize=(10, 8))
sns.barplot(x='Importancia', y='Biomarcador', data=df_importancia, palette='magma')
plt.title('Pesos de Variables en Regresión Logística\n(Modelo con biomarcadores íntegros)', fontsize=14)
plt.xlabel('Magnitud Media Absoluta del Coeficiente', fontsize=12)
plt.ylabel('Estudio Clínico / Biomarcador', fontsize=12)
plt.tight_layout()
plt.show()