from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = load_model("diabetes_model.h5")

# Load the scaler
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
columns = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI",
           "DiabetesPedigreeFunction", "Age", "Outcome"]
df = pd.read_csv(url, names=columns)
X = df.iloc[:, :-1]
scaler = StandardScaler().fit(X)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    data = [data[feature] for feature in ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]]
    data = np.array(data).reshape(1, -1)
    data = scaler.transform(data)
    prediction = model.predict(data)
    output = "Positive" if prediction[0][0] > 0.5 else "Negative"
    return jsonify(prediction_text=f'Diabetes Prediction: {output}')

if __name__ == "__main__":
    app.run(debug=True)