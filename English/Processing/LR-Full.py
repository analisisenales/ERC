# -*- coding: utf-8 -*-
"""
Logistic Regression with complete biomarkers
Approach: Classification using multiclass linear models using the complete dossier.
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

# Step 1: Loading patient data from the source file.
print("Loading the complete database...")
directorio_actual = os.path.dirname(os.path.abspath(__file__))
archivo_limpio = os.path.join(directorio_actual, 'Unificacion_tablas.csv')

try:
    df = pd.read_csv(archivo_limpio)
except Exception as e:
    print(f"❌ Error loading CSV: {e}")
    raise

# Step 2: Generating the "Ground Truth" or target variable by calculating eGFR.
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
    if egfr >= 60: return "G1-G2 (Healthy/Mild)"
    elif egfr >= 30: return "G3 (Moderate)"
    else: return "G4-G5 (Severe)"

df['Target_Estadio'] = df['eGFR_Real'].apply(clasificar_estadio)

# Step 3: Preparing data for Machine Learning.
# As it is the full version, the direct calculation variables within set X are preserved.
columnas_excluir = ['ID_Participante', 'Target_Estadio', 'eGFR_Real']

X = df.drop(columns=columnas_excluir)
y = df['Target_Estadio']

# División de los datos: 75% para entrenamiento, 25% para prueba (test).
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# Step 4: Building the Preprocessing Pipeline (Imputation and Scaling).
print("\n🛠️ Filling missing data and scaling variables...")

# A. Imputation of missing values ​​using the median of each characteristic.
imputador = SimpleImputer(strategy='median')
X_train_imputed = imputador.fit_transform(X_train)
X_test_imputed = imputador.transform(X_test)

# B. Statistical z-score scaling (essential for the stability of linear models).
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

# Step 5: Initializing and training the Logistic Regression classifier.
print("🧠 Training Logistic Regression with all variables...")
# Se especifica max_iter=1000 para garantizar la convergencia del optimizador numérico.
logreg_model = LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000)
logreg_model.fit(X_train_scaled, y_train)

# Step 6: Model evaluation, calculation of performance metrics, and visualization of results.
y_pred = logreg_model.predict(X_test_scaled)

print("\n" + "="*50)
accuracy = accuracy_score(y_test, y_pred)
print(f"🏆 Complete Logistic Regression ACCURACY: {accuracy * 100:.2f}%")
print("="*50)
print("\n📋 Detailed Report:")
print(classification_report(y_test, y_pred))

# Step 7: Extraction and consolidation of the model coefficients (Feature Importance).
# The average of the absolute value of the coefficients is calculated across all target classes.
coeficientes_absolutos = np.abs(logreg_model.coef_).mean(axis=0)

df_importancia = pd.DataFrame({
    'Biomarcador': X.columns, 
    'Importancia': coeficientes_absolutos
})
df_importancia = df_importancia.sort_values(by='Importancia', ascending=False)

# Generation of the bar chart for linear coefficient analysis.
plt.figure(figsize=(10, 8))
sns.barplot(x='Importancia', y='Biomarcador', data=df_importancia, palette='magma')
plt.title('Variable Weights in Logistic Regression\n(Complete Model with Imputed Data)', fontsize=14)
plt.xlabel('Mean Absolute Coefficient Magnitude', fontsize=12)
plt.ylabel('Clinical Study / Biomarker', fontsize=12)
plt.tight_layout()
plt.show()