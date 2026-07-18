# -*- coding: utf-8 -*-
"""
KNN with intact biomarkers. 
Approach: Classification by distances using all available biomarkers.
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

# Step 1: Loading patient data from the source file.
print("Loading the complete database...")
directorio_actual = os.path.dirname(os.path.abspath(__file__))
archivo_limpio = os.path.join(directorio_actual, 'Unificacion_tablas.csv')

try:
    df = pd.read_csv(archivo_limpio)
    print(f"📄 Initial patients in the file: {len(df)}")
except Exception as e:
    print(f"❌ Error loading CSV: {e}")
    raise

# Step 2: Generating the "Ground Truth" or target variable.
# It is essential to remove records without creatinine, age, or gender to allow for the correct calculation of eGFR.
df.dropna(subset=['Creatinina_umol_L', 'Edad', 'Genero'], inplace=True)
print(f"✅ Patients retained for the experiment: {len(df)}")

# Calculation of the actual estimated glomerular filtration rate (eGFR) internally.
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
# Target variables are excluded to avoid bias or data leaks.
columnas_excluir = ['ID_Participante', 'Target_Estadio', 'eGFR_Real']
X = df.drop(columns=columnas_excluir)
y = df['Target_Estadio']

# Data division: 75% for training, 25% for testing.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# Step 4: Building the Preprocessing Pipeline (Imputation and Scaling).
print("\n🛠️ Filling missing data and scaling variables...")

# A. Imputation of missing values ​​using the median of each characteristic.
imputador = SimpleImputer(strategy='median')
X_train_imputed = imputador.fit_transform(X_train)
X_test_imputed = imputador.transform(X_test)

# B. Feature scaling to homogenize the magnitude of all numerical variables.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

# Step 5: Initializing and training the KNN classifier.
print("🧠 Training KNN with recovered and scaled data...")
knn_model = KNeighborsClassifier(n_neighbors=5) 
knn_model.fit(X_train_scaled, y_train)

# Step 6: Model evaluation, calculation of performance metrics, and visualization of results.
y_pred = knn_model.predict(X_test_scaled)

print("\n" + "="*50)
accuracy = accuracy_score(y_test, y_pred)
print(f"🏆 Imputed KNN ACCURACY: {accuracy * 100:.2f}%")
print("="*50)
print("\n📋 Detailed Report:")
print(classification_report(y_test, y_pred))

# Generation of the Confusion Matrix for visual analysis of predictions.
plt.figure(figsize=(7, 5))
cm = confusion_matrix(y_test, y_pred, labels=["G1-G2 (Healthy/Mild)", "G3 (Moderate)", "G4-G5 (Severe)"])
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', 
            xticklabels=["G1-G2", "G3", "G4-G5"], 
            yticklabels=["G1-G2", "G3", "G4-G5"])
plt.title(f'KNN Confusion Matrix (Imputed Data)\nAccuracy: {accuracy:.2f}')
plt.ylabel('Current Stage (Medicine)')
plt.xlabel('AI Prediction')
plt.show()