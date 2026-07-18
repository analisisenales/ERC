# -*- coding: utf-8 -*-
"""
Random Forest with complete biomarkers
Approach: Classification by tree assemblies using the complete record.
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
    # The initial data load is performed without applying global deletions of missing values.
    df = pd.read_csv(archivo_limpio)
    pacientes_iniciales = len(df)
    print(f"📄 Initial patients in the file: {pacientes_iniciales}")
except Exception as e:
    print(f"❌ Error loading CSV: {e}")
    raise

# Step 2: Generating the "Ground Truth" or target variable by calculating eGFR.
# Records with key data are retained only to generate the correct rating.
df.dropna(subset=['Creatinina_umol_L', 'Edad', 'Genero'], inplace=True)
print(f"✅ Patients retained after validating ground truth requirements: {len(df)}")

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
columnas_excluir = ['ID_Participante', 'Target_Estadio', 'eGFR_Real']
X = df.drop(columns=columnas_excluir)
y = df['Target_Estadio']

# Data division: 75% for training, 25% for testing.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# Step 4: Building the Preprocessing Pipeline using Statistical Imputation.
print("\n🛠️ Applying Statistical Imputation (Median) to incomplete records...")
# The median is used for its robustness against atypical or extreme clinical values.
imputador = SimpleImputer(strategy='median')

# The fit is performed exclusively on the training set to prevent data leakage.
X_train_imputed = pd.DataFrame(imputador.fit_transform(X_train), columns=X_train.columns)
X_test_imputed = pd.DataFrame(imputador.transform(X_test), columns=X_test.columns)

# Step 5: Initializing and training the Random Forest classifier.
print("🧠 Training Random Forest with recovered data...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_model.fit(X_train_imputed, y_train)

# Step 6: Model evaluation, calculation of performance metrics, and visualization of results.
y_pred = rf_model.predict(X_test_imputed)

print("\n" + "="*50)
accuracy = accuracy_score(y_test, y_pred)
print(f"🏆 Random Forest ACCURACY: {accuracy * 100:.2f}%")
print("="*50)
print("\n📋 Detailed Report:")
print(classification_report(y_test, y_pred))

# Step 7: Extraction and visualization of feature importance.
importancias = rf_model.feature_importances_
df_importancia = pd.DataFrame({'Biomarcador': X.columns, 'Importancia': importancias})
df_importancia = df_importancia.sort_values(by='Importancia', ascending=False)

plt.figure(figsize=(10, 8))
sns.barplot(x='Importancia', y='Biomarcador', data=df_importancia, palette='mako')
plt.title('Biomarker Importance in CKD Diagnosis\n(Random Forest with Imputed Data)', fontsize=14)
plt.xlabel('Importance Level (Fraction of 100%)', fontsize=12)
plt.ylabel('Clinical Study / Biomarker', fontsize=12)
plt.tight_layout()
plt.show()