from flask import Flask, render_template, request
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os

model_path = os.path.join(os.path.dirname(__file__), 'models', 'modelo_regresion.pkl')
model = joblib.load(model_path)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    Nitrógeno = int(request.form['Nitrógeno'])
    Fósforo = int(request.form['Fósforo'])
    Potasio = int(request.form['Potasio'])
    Temperatura = int(request.form['Temperatura'])
    Humedad = int(request.form['Humedad'])
    PH_Suelo = int(request.form['PH_Suelo'])
    Precipitación = int(request.form['Precipitación'])

    new_samples = np.array([[Nitrógeno, Fósforo, Potasio, Temperatura, Humedad, PH_Suelo, Precipitación]])

    escalas = scaler.transform(new_samples)

    prediction = model.predict([escalas])

    mensaje = ""
    mensaje += f"La Clasificacion es: {prediction[0]}"

    return render_template('result.html', predi=mensaje)

if __name__ == '__main__':
    app.run()