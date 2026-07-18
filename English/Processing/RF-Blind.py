# -*- coding: utf-8 -*-
"""
Random Forest without primary biomarkers.
Focus: Discovery of alternative biomarkers for CKD prediction using a blinded model.
"""

import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer
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
# Records with key data are retained only to generate the correct rating.
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

# Step 3: Preparing data for Machine Learning previniendo la fuga de datos (Data Leakage).
# Directly calculated variables are explicitly excluded to force the model to predict
# using exclusively secondary biomarkers.
columnas_excluir = [
    'ID_Participante', 
    'Target_Estadio', 
    'eGFR_Real', 
    'Creatinina_mg_dL',
    'Creatinina_umol_L',
    'Edad',
    'Genero'
]

X = df.drop(columns=columnas_excluir)
y = df['Target_Estadio']

# Data division: 75% for training, 25% for testing.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# Step 4: Building the Preprocessing Pipeline using Statistical Imputation.
print("\n🛠️ Filling missing values in secondary biomarkers...")
imputador = SimpleImputer(strategy='median')
# Se conserva el formato DataFrame para mantener los nombres de las columnas para la evaluación posterior.
X_train_imputed = pd.DataFrame(imputador.fit_transform(X_train), columns=X_train.columns)
X_test_imputed = pd.DataFrame(imputador.transform(X_test), columns=X_test.columns)

# Step 5: Initializing and training the Random Forest classifier.
print("🧠 Training blind AI (finding hidden relationships in blood data)...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_model.fit(X_train_imputed, y_train)

# Step 6: Model evaluation, calculation of performance metrics, and visualization of results.
y_pred = rf_model.predict(X_test_imputed)

print("\n" + "="*50)
accuracy = accuracy_score(y_test, y_pred)
print(f"🏆 Blind Random Forest ACCURACY: {accuracy * 100:.2f}%")
print("="*50)
print("\n📋 Detailed Report:")
print(classification_report(y_test, y_pred))

# Step 7: Extraction and visualization of feature importance.
importancias = rf_model.feature_importances_
df_importancia = pd.DataFrame({'Biomarcador': X.columns, 'Importancia': importancias})
df_importancia = df_importancia.sort_values(by='Importancia', ascending=False)

plt.figure(figsize=(10, 8))
sns.barplot(x='Importancia', y='Biomarcador', data=df_importancia, palette='rocket')
plt.title('Discovery of Alternative Biomarkers for CKD\n(Blind Model)', fontsize=14)
plt.xlabel('Importance Level (Fraction of 100%)', fontsize=12)
plt.ylabel('Clinical Study / Biomarker', fontsize=12)
plt.tight_layout()
plt.show()