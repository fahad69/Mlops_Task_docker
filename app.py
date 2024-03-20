from flask import Flask, request, jsonify
import pickle
import os

app = Flask(__name__)
current_dir = os.path.dirname(__file__)
model_path = os.path.join(os.getcwd(), 'model.pkl')
with open(model_path, 'rb') as file:
    model = pickle.load(file)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    pregnancies = data['pregnancies']
    glucose = data['glucose']
    blood_pressure = data['blood_pressure']
    skin_thickness = data['skin_thickness']
    insulin = data['insulin']
    bmi = data['bmi']
    diabetes_pedigree_function = data['diabetes_pedigree_function']
    age = data['age']
    prediction_data = [[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]]
    prediction = model.predict(prediction_data)
    return jsonify({'outcome': prediction.tolist()})

if __name__ == '__main__':
    app.run()
