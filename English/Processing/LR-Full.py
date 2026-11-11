# -*- coding: utf-8 -*-
"""
Logistic Regression with complete biomarkers.
Focus: Multi-class linear model classification using the entire medical record.
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

# Step 1: Load patient data from the source file.
print("Loading the complete database...")
current_directory = os.path.dirname(os.path.abspath(__file__))
clean_file = os.path.join(current_directory, 'merged_tables.csv')

try:
    df = pd.read_csv(clean_file)
except Exception as e:
    print(f"❌ Error loading CSV: {e}")
    raise

# Step 2: Generate the "Ground Truth" or target variable through eGFR calculation.
df.dropna(subset=['Creatinine, refrigerated serum (umol/L)', 'Age', 'Gender'], inplace=True)

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
# Since this is the complete version, direct calculation variables are preserved within the X set.
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

# B. Z-score statistical scaling (indispensable for linear model stability).
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

# Step 5: Initialization and training of the Logistic Regression classifier model.
print("🧠 Training Logistic Regression with all variables...")
# max_iter=1000 is specified to guarantee the convergence of the numerical optimizer.
logreg_model = LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000)
logreg_model.fit(X_train_scaled, y_train)

# Step 6: Model evaluation, performance metrics calculation, and visualization.
y_pred = logreg_model.predict(X_test_scaled)

print("\n" + "="*50)
accuracy = accuracy_score(y_test, y_pred)
print(f"🏆 Complete Logistic Regression ACCURACY: {accuracy * 100:.2f}%")
print("="*50)
print("\n📋 Detailed Report:")
print(classification_report(y_test, y_pred))

# Step 7: Extraction and consolidation of model coefficients (Feature Importance).
# The mean of the absolute value of the coefficients is calculated across all target classes.
absolute_coefficients = np.abs(logreg_model.coef_).mean(axis=0)

df_importance = pd.DataFrame({
    'Biomarker': X.columns, 
    'Importance': absolute_coefficients
})
df_importance = df_importance.sort_values(by='Importance', ascending=False)

# Generation of the bar chart for linear coefficient analysis.
plt.figure(figsize=(10, 8))
sns.barplot(x='Importance', y='Biomarker', data=df_importance, palette='magma')
plt.title('Variable Weights in Logistic Regression\n(Model with complete biomarkers)', fontsize=14)
plt.xlabel('Mean Absolute Coefficient Magnitude', fontsize=12)
plt.ylabel('Clinical Study / Biomarker', fontsize=12)
plt.tight_layout()
plt.show()