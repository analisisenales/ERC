# -*- coding: utf-8 -*-
"""
Random Forest con biomarcadores íntegros
Enfoque: Clasificación mediante ensambles de árboles utilizando el expediente completo.
"""

import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

# Paso 1: Carga de datos de los pacientes desde el archivo fuente.
print("Cargando la base de datos completa...")
directorio_actual = os.path.dirname(os.path.abspath(__file__))
archivo_limpio = os.path.join(directorio_actual, 'Unificacion_tablas.csv')

try:
    # Se realiza la carga inicial de los datos sin aplicar eliminaciones globales de valores faltantes.
    df = pd.read_csv(archivo_limpio)
    pacientes_iniciales = len(df)
    print(f"📄 Pacientes iniciales en el archivo: {pacientes_iniciales}")
except Exception as e:
    print(f"❌ Error al cargar el CSV: {e}")
    raise

# Paso 2: Generación de la "Ground Truth" o variable objetivo mediante el cálculo de eGFR.
# Se excluyen únicamente los registros que carecen de las variables mandatorias para la ecuación.
df.dropna(subset=['Creatinina_umol_L', 'Edad', 'Genero'], inplace=True)
print(f"✅ Pacientes retenidos tras validar requisitos de ground truth: {len(df)}")

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
columnas_excluir = ['ID_Participante', 'Target_Estadio', 'eGFR_Real']
X = df.drop(columns=columnas_excluir)
y = df['Target_Estadio']

# División de los datos: 75% para entrenamiento, 25% para prueba (test).
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# Paso 4: Construcción del Pipeline de preprocesamiento mediante Imputación Estadística.
print("\n🛠️ Aplicando Imputación Estadística (Mediana) a registros incompletos...")
# Se utiliza la mediana por su robustez ante valores clínicos atípicos o extremos.
imputador = SimpleImputer(strategy='median')

# El ajuste (fit) se realiza exclusivamente sobre el conjunto de entrenamiento para prevenir la fuga de datos.
X_train_imputed = pd.DataFrame(imputador.fit_transform(X_train), columns=X_train.columns)
X_test_imputed = pd.DataFrame(imputador.transform(X_test), columns=X_test.columns)

# Paso 5: Inicialización y entrenamiento del modelo clasificador Random Forest.
print("🧠 Entrenando Random Forest con datos recuperados...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_model.fit(X_train_imputed, y_train)

# Paso 6: Evaluación del modelo, cálculo de métricas de rendimiento y visualización de resultados.
y_pred = rf_model.predict(X_test_imputed)

print("\n" + "="*50)
accuracy = accuracy_score(y_test, y_pred)
print(f"🏆 PRECISIÓN (ACCURACY) Random Forest Completo: {accuracy * 100:.2f}%")
print("="*50)
print("\n📋 Reporte Detallado:")
print(classification_report(y_test, y_pred))

# Paso 7: Extracción y visualización de la importancia de las características (Importancia de Variables).
importancias = rf_model.feature_importances_
df_importancia = pd.DataFrame({'Biomarcador': X.columns, 'Importancia': importancias})
df_importancia = df_importancia.sort_values(by='Importancia', ascending=False)

plt.figure(figsize=(10, 8))
sns.barplot(x='Importancia', y='Biomarcador', data=df_importancia, palette='mako')
plt.title('Importancia de Biomarcadores en Diagnóstico de ERC\n(Random Forest con biomarcadores íntegros)', fontsize=14)
plt.xlabel('Nivel de Importancia (Fracción del 100%)', fontsize=12)
plt.ylabel('Estudio Clínico / Biomarcador', fontsize=12)
plt.tight_layout()
plt.show()