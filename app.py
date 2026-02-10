from flask import Flask, render_template, request
import torch
import torch.nn as nn
import numpy as np
import pickle

app = Flask(__name__)

# ======================
# ANN MODEL
# ======================
class ANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(24, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.sigmoid(self.fc4(x))
        return x

# ======================
# LOAD MODEL + SCALER
# ======================
model = ANN()
model.load_state_dict(torch.load("diabetes_model.pth", map_location="cpu"))
model.eval()

scaler = pickle.load(open("scaler.pkl", "rb"))

# ======================
# ROUTES
# ======================
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        gender = int(request.form['gender'])
        age = float(request.form['age'])
        bmi = float(request.form['bmi'])
        systolic_bp = float(request.form['systolic_bp'])
        diastolic_bp = float(request.form['diastolic_bp'])
        cholesterol = float(request.form['cholesterol'])
        smoking = int(request.form['smoking'])
        alcohol = float(request.form['alcohol'])
        physical_activity = float(request.form['physical_activity'])
        family_history = int(request.form['family_history'])
        bp_history = int(request.form['bp_history'])

        features = np.zeros(24)
        features[0] = age
        features[1] = alcohol
        features[2] = physical_activity
        features[6] = bmi
        features[8] = systolic_bp
        features[9] = diastolic_bp
        features[11] = cholesterol
        features[15] = gender
        features[19] = smoking
        features[21] = family_history
        features[22] = bp_history

        features = scaler.transform([features])
        features_tensor = torch.tensor(features, dtype=torch.float32)

        with torch.no_grad():
            output = model(features_tensor)
            prediction = (output > 0.5).item()

        result = (
            "Diabetes Detected (High Risk)"
            if prediction == 1
            else "No Diabetes Detected (Low Risk)"
        )

        return render_template("index.html", prediction_text=result)

    except Exception as e:
        return render_template("index.html", prediction_text=str(e))

if __name__ == "__main__":
    app.run(debug=True)
