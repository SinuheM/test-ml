import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Generar datos de ejemplo
np.random.seed(42)
n_samples = 1000

# Variables predictoras
temperatura = np.random.normal(25, 5, n_samples)
humedad = np.random.normal(60, 10, n_samples)
velocidad_viento = np.random.normal(15, 5, n_samples)

# Variable objetivo (precipitación) con algo de ruido
precipitacion = (0.3 * temperatura + 0.5 * humedad + 0.2 * velocidad_viento + 
                np.random.normal(0, 5, n_samples))

data = pd.read_csv('TodoMes.csv')

print(data.head())

X = data[['ALLSKY_SFC_UVA', 'RH2M', 'PRECTOTCORR', 'T2M']]
y = data['Rancha']

""" # Crear DataFrame
data = pd.DataFrame({
    'temperatura': temperatura,
    'humedad': humedad,
    'velocidad_viento': velocidad_viento,
    'precipitacion': precipitacion
})

# Dividir los datos en conjuntos de entrenamiento y prueba
X = data[['temperatura', 'humedad', 'velocidad_viento']]
y = data['precipitacion'] """
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear y entrenar el modelo
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# Realizar predicciones
y_pred = modelo.predict(X_test)

# Evaluar el modelo
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Imprimir resultados
print("Coeficientes:")
for feature, coef in zip(X.columns, modelo.coef_):
    print(f"{feature}: {coef:.4f}")
print(f"Intercepto: {modelo.intercept_:.4f}")
print(f"Error cuadrático medio: {mse:.4f}")
print(f"R-cuadrado: {r2:.4f}")

# Guardar el modelo entrenado
joblib.dump(modelo, 'modelo_entrenado.pkl')

# Visualización de resultados
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Valores reales")
plt.ylabel("Predicciones")
plt.title("Valores reales vs Predicciones")
plt.tight_layout()
plt.show()

# Gráfico de correlación
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title("Matriz de Correlación")
plt.tight_layout()
plt.show()
