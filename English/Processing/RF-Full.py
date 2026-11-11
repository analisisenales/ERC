# -*- coding: utf-8 -*-
"""
Random Forest with complete biomarkers.
Focus: Classification via tree ensembles using the complete medical record.
"""

import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load patient data from the source file.
print("Loading the complete database...")
current_directory = os.path.dirname(os.path.abspath(__file__))
clean_file = os.path.join(current_directory, 'merged_tables.csv')

try:
    # Initial data loading is performed without applying global missing value deletions.
    df = pd.read_csv(clean_file)
    initial_patients = len(df)
    print(f"📄 Initial patients in the file: {initial_patients}")
except Exception as e:
    print(f"❌ Error loading CSV: {e}")
    raise

# Step 2: Generate the "Ground Truth" or target variable through eGFR calculation.
# Only records lacking the mandatory variables for the equation are excluded.
df.dropna(subset=['Creatinine, refrigerated serum (umol/L)', 'Age', 'Gender'], inplace=True)
print(f"✅ Patients retained after validating ground truth requirements: {len(df)}")

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
columns_to_exclude = ['Participant_ID', 'Target_Stage', 'eGFR_Real']
X = df.drop(columns=columns_to_exclude)
y = df['Target_Stage']

# Data splitting: 75% for training, 25% for testing.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# Step 4: Preprocessing Pipeline through Statistical Imputation.
print("\n🛠️ Applying Statistical Imputation (Median) to incomplete records...")
# Median is used due to its robustness against atypical or extreme clinical values.
imputer = SimpleImputer(strategy='median')

# The fit is exclusively performed on the training set to prevent data leakage.
X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

# Step 5: Initialization and training of the Random Forest classifier model.
print("🧠 Training Random Forest with recovered data...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_model.fit(X_train_imputed, y_train)

# Step 6: Model evaluation, performance metrics calculation, and visualization.
y_pred = rf_model.predict(X_test_imputed)

print("\n" + "="*50)
accuracy = accuracy_score(y_test, y_pred)
print(f"🏆 Random Forest ACCURACY: {accuracy * 100:.2f}%")
print("="*50)
print("\n📋 Detailed Report:")
print(classification_report(y_test, y_pred))

# Step 7: Extraction and visualization of feature importance.
importances = rf_model.feature_importances_
df_importance = pd.DataFrame({'Biomarker': X.columns, 'Importance': importances})
df_importance = df_importance.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 8))
sns.barplot(x='Importance', y='Biomarker', data=df_importance, palette='mako')
plt.title('Biomarker Importance in CKD Diagnosis\n(Random Forest with complete biomarkers)', fontsize=14)
plt.xlabel('Importance Level (Fraction of 100%)', fontsize=12)
plt.ylabel('Clinical Study / Biomarker', fontsize=12)
plt.tight_layout()
plt.show()