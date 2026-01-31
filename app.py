from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
try:
    # Ensure this file is in the same directory as app.py
    model = pickle.load(open('diabetes_model.pkl', 'rb'))
except FileNotFoundError:
    print("Error: diabetes_model.pkl file not found! Please check your directory.")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # 1. Retrieve data from the English HTML form
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

            # 2. Map inputs to the 24-feature vector required by the model
            # Based on your training columns structure
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

            # 3. Perform Prediction
            prediction = model.predict([features])

            # 4. Determine Result in English
            if prediction[0] == 1:
                result = "Diabetes Detected (High Risk)"
            else:
                result = "No Diabetes Detected (Low Risk)"

            return render_template('index.html', prediction_text=result)

        except Exception as e:
            # Error handling in English
            return render_template('index.html', prediction_text=f"Input Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)