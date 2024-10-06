from flask import Flask, request, jsonify
import joblib
import numpy as np

# Cargar el modelo entrenado
modelo = joblib.load('modelo_entrenado.pkl')

# Inicializar la aplicación Flask
app = Flask(__name__)

@app.route('/predecir', methods=['POST'])
def predecir():
    # Obtener los datos JSON del cuerpo de la solicitud
    datos = request.get_json()

    # Extraer las variables de entrada (temperatura, humedad, velocidad del viento)
    temperature = datos['temperature']
    precipitation = datos['precipitation']
    radiation = datos['radiation']
    humidity = datos['humidity']
    # velocidad_viento = datos['velocidad_viento']

    # Crear el array de entrada para el modelo
    entrada = np.array([[radiation, humidity, precipitation, temperature]])

    # Realizar la predicción
    prediccion = modelo.predict(entrada)

    # Retornar la predicción en formato JSON
    return jsonify({'prediccion': prediccion[0]})

if __name__ == '__main__':
    app.run(debug=True)

# curl -X POST http://127.0.0.1:5000/predecir -H "Content-Type: application/json" -d '{"temperature": 25, "humidity": 60, "radiation": 15, "precipitation": 40}'
