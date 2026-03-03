from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import os

app = Flask(__name__)

# This logic automatically finds the model in the same folder as this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = r'C:\Users\ARJUN\OneDrive\Desktop\printing-internship\ink_model_pipeline.pkl'
MODEL_PATH = os.path.join(BASE_DIR, MODEL_NAME)

# Load the model once when the server starts
try:
    # Ensure you have exported 'ink_model_pipeline.pkl' from your notebook
    model = joblib.load(MODEL_PATH)
    print("✅ Success: ML Model Loaded.")
except Exception as e:
    print(f"❌ Error: Could not find or load {MODEL_NAME} in {BASE_DIR}")
    print(f"Details: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # 1. Capture the initial values
        de_before = float(data['Delta_E_before'])
        # Since we don't have the 'after' value yet (that's what we are predicting),
        # your model likely used a placeholder or the 'reduction' was a training feature.
        # Based on your notebook cell 6, it was calculated as:
        # DeltaE_reduction = Delta_E_before - Delta_E_after
        
        # NOTE: If this was a feature during training, you must provide it.
        # Usually, for a prediction app, we calculate the reduction based on the
        # user's intended target or use a default of 0 if it's an 'after' prediction.
        # Based on your specific model requirements:
        de_reduction = de_before - float(data.get('Delta_E_after', 0)) 

        # 2. Create the DataFrame with the missing column included
        input_df = pd.DataFrame([{
            'Press_ID': int(data['Press_ID']),
            'Job_Number': str(data['Job_Number']),
            'Paper_type': str(data['Paper_type']),
            'Zone_number': float(data['Zone_number']),
            'Color': str(data['Color']),
            'Ink_key_zero_setting': float(data['Ink_key_zero_setting']),
            'Delta_E_before': de_before,
            'initial_density': float(data['initial_density']),
            'final_density': float(data['final_density']),
            'initial_ink_key_setting': float(data['initial_ink_key_setting']),
            'final_ink_key_setting': float(data['final_ink_key_setting']),
            'DeltaE_reduction': de_reduction  # <--- THIS WAS MISSING
        }])

        # 3. Run the Prediction
        prediction = model.predict(input_df)
        
        return jsonify({
            "status": "success",
            "prediction": round(float(prediction[0]), 3)
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})
if __name__ == '__main__':
    # port 5000 is the default for Flask
    app.run(debug=True, port=5000)