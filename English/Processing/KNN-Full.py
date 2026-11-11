# -*- coding: utf-8 -*-
"""
KNN with complete biomarkers.
Focus: Distance-based classification using all available biomarkers (Baseline Model).
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

# Step 1: Load patient data from the source file.
print("Loading the complete database...")
current_directory = os.path.dirname(os.path.abspath(__file__))
clean_file = os.path.join(current_directory, 'merged_tables.csv')

try:
    df = pd.read_csv(clean_file)
    print(f"📄 Initial patients in the file: {len(df)}")
except Exception as e:
    print(f"❌ Error loading CSV: {e}")
    raise

# Step 2: Generate the "Ground Truth" or target variable.
# It is essential to remove records without creatinine, age, or gender for the correct eGFR calculation.
df.dropna(subset=['Creatinine, refrigerated serum (umol/L)', 'Age', 'Gender'], inplace=True)
print(f"✅ Patients retained for the experiment: {len(df)}")

# Internal calculation of the real estimated Glomerular Filtration Rate (eGFR).
df['Creatinine_mg_dL'] = df['Creatinine, refrigerated serum (umol/L)'] / 88.4

def calculate_egfr(row):
    scr = row['Creatinine_mg_dL']
    age = row['Age']
    sex = row['Gender']
    
    kappa = 0.7 if sex == 2 else 0.9
    alpha = -0.241 if sex == 2 else -0.302
    sex_multiplier = 1.012 if sex == 2 else 1.0
    
    return 142 * min(scr/kappa, 1)**alpha * max(scr/kappa, 1)**-1.200 * 0.9938**age * sex_multiplier

df['eGFR_Real'] = df.apply(calculate_egfr, axis=1)

def classify_stage(egfr):
    if egfr >= 60: return "G1-G2 (Healthy/Mild)"
    elif egfr >= 30: return "G3 (Moderate)"
    else: return "G4-G5 (Severe)"

df['Target_Stage'] = df['eGFR_Real'].apply(classify_stage)

# Step 3: Data preparation for Machine Learning.
# Target variables are excluded to prevent bias or Data Leakage.
columns_to_exclude = ['Participant_ID', 'Target_Stage', 'eGFR_Real']
X = df.drop(columns=columns_to_exclude)
y = df['Target_Stage']

# Data splitting: 75% for training, 25% for testing.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# Step 4: Preprocessing Pipeline (Imputation and Scaling).
print("\n🛠️ Filling missing data and scaling variables...")

# A. Imputation of missing values using the median for each feature.
imputer = SimpleImputer(strategy='median')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# B. Feature scaling to homogenize the magnitude of all numerical variables.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

# Step 5: Initialization and training of the KNN classifier model.
print("🧠 Training KNN with all biomarkers (imputed and scaled data)...")
knn_model = KNeighborsClassifier(n_neighbors=5) 
knn_model.fit(X_train_scaled, y_train)

# Step 6: Model evaluation, performance metrics calculation, and visualization.
y_pred = knn_model.predict(X_test_scaled)

print("\n" + "="*50)
accuracy = accuracy_score(y_test, y_pred)
print(f"🏆 Complete KNN ACCURACY: {accuracy * 100:.2f}%")
print("="*50)
print("\n📋 Detailed Report:")
print(classification_report(y_test, y_pred))

# Generation of the Confusion Matrix for visual analysis.
plt.figure(figsize=(7, 5))
cm = confusion_matrix(y_test, y_pred, labels=["G1-G2 (Healthy/Mild)", "G3 (Moderate)", "G4-G5 (Severe)"])
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', 
            xticklabels=["G1-G2", "G3", "G4-G5"], 
            yticklabels=["G1-G2", "G3", "G4-G5"])
plt.title(f'KNN Confusion Matrix (Complete Model)\nAccuracy: {accuracy:.2f}')
plt.ylabel('Actual Stage (Medicine)')
plt.xlabel('AI Prediction')
plt.show()