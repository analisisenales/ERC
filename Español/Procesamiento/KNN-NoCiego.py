# -*- coding: utf-8 -*-
"""
KNN con biomarcadores íntegros.
Enfoque: Clasificación por distancias usando todos los biomarcadores disponibles (Modelo Completo).
"""

import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Paso 1: Carga de datos de los pacientes desde el archivo fuente.
print("Cargando la base de datos completa...")
directorio_actual = os.path.dirname(os.path.abspath(__file__))
archivo_limpio = os.path.join(directorio_actual, 'Unificacion_tablas.csv')

try:
    df = pd.read_csv(archivo_limpio)
    print(f"📄 Pacientes iniciales en el archivo: {len(df)}")
except Exception as e:
    print(f"❌ Error al cargar el CSV: {e}")
    raise

# Paso 2: Generación de la "Ground Truth" o variable objetivo.
# Es indispensable eliminar registros sin creatinina, edad o género para permitir el cálculo correcto de eGFR.
df.dropna(subset=['Creatinina_umol_L', 'Edad', 'Genero'], inplace=True)
print(f"✅ Pacientes conservados para el experimento: {len(df)}")

# Cálculo de la tasa de filtración glomerular estimada (eGFR) real internamente.
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
# Se excluyen las variables objetivo para evitar sesgos o fuga de datos en la evaluación.
columnas_excluir = ['ID_Participante', 'Target_Estadio', 'eGFR_Real']
X = df.drop(columns=columnas_excluir)
y = df['Target_Estadio']

# División de los datos: 75% para entrenamiento, 25% para prueba (test).
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# Paso 4: Construcción del Pipeline de preprocesamiento (Imputación y Escalamiento).
print("\n🛠️ Rellenando valores faltantes y escalando variables...")

# A. Imputación de valores faltantes utilizando la mediana.
imputador = SimpleImputer(strategy='median')
X_train_imputed = imputador.fit_transform(X_train)
X_test_imputed = imputador.transform(X_test)

# B. Escalamiento de características para homogeneizar la magnitud de todas las variables numéricas.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

# Paso 5: Inicialización y entrenamiento del modelo clasificador KNN.
print("🧠 Entrenando KNN con todos los biomarcadores (datos imputados y escalados)...")
knn_model = KNeighborsClassifier(n_neighbors=5) 
knn_model.fit(X_train_scaled, y_train)

# Paso 6: Evaluación del modelo, cálculo de métricas de rendimiento y visualización de resultados.
y_pred = knn_model.predict(X_test_scaled)

print("\n" + "="*50)
accuracy = accuracy_score(y_test, y_pred)
print(f"🏆 PRECISIÓN (ACCURACY) KNN Completo: {accuracy * 100:.2f}%")
print("="*50)
print("\n📋 Reporte Detallado:")
print(classification_report(y_test, y_pred))

# Generación de la Matriz de Confusión para análisis visual de predicciones.
plt.figure(figsize=(7, 5))
cm = confusion_matrix(y_test, y_pred, labels=["G1-G2 (Sano/Leve)", "G3 (Moderado)", "G4-G5 (Grave)"])
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', 
            xticklabels=["G1-G2", "G3", "G4-G5"], 
            yticklabels=["G1-G2", "G3", "G4-G5"])
plt.title(f'Matriz de Confusión KNN (Modelo Completo)\nPrecisión: {accuracy:.2f}')
plt.ylabel('Estadio Real (Medicina)')
plt.xlabel('Predicción de la IA')
plt.show()