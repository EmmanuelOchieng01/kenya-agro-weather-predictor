from flask import Flask, render_template, request, jsonify
import pickle
import json
import numpy as np
import os

app = Flask(__name__)

# Load models
with open('models/best_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('models/encoders.pkl', 'rb') as f:
    encoders = pickle.load(f)
with open('models/metadata.json', 'r') as f:
    metadata = json.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        crop_enc = encoders['crop'].transform([data['crop']])[0]
        region_enc = encoders['region'].transform([data['region']])[0]
        season_enc = encoders['season'].transform([data['season']])[0]

        features = np.array([[
            data.get('year', 2025), crop_enc, region_enc, season_enc,
            data.get('temp_mean', 25) + 5, data.get('temp_mean', 25) - 5,
            data.get('temp_mean', 25), data.get('precipitation', 600),
            3, 18000, 200, 6.5, 0.15, 1.5
        ]])

        prediction = model.predict(features)[0]

        return jsonify({
            "success": True,
            "prediction": round(prediction, 2),
            "confidence": "high",
            "recommendation": f"Expected yield: {round(prediction, 2)} kg/ha",
            "crop": data['crop'],
            "region": data['region']
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
