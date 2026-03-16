from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import json
import os

app = Flask(__name__)

# Paths - Ensure these point to your current desktop setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = r'C:\Users\ARJUN\OneDrive\Desktop\printing-internship\ink_key_stacking_model.pkl' 
MODEL_PATH = os.path.join(BASE_DIR, MODEL_NAME)
CONFIG_PATH = os.path.join(BASE_DIR, 'press_settings.json')
EXCEL_PATH = os.path.join(BASE_DIR, 'print_logs.xlsx')

# MAPPINGS: Exact match to your pakka_data.map() logic
PAPER_MAP = {'Coated': 0, 'Uncoated': 1}
COLOR_MAP = {'Cyan': 0, 'Magenta': 1, 'Yellow': 2, 'Black': 3}

def get_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, 'r') as f: return json.load(f)
    return {"density_relation": 0.5, "zero_setting": 0, "target_de": 2.5}

try:
    model = joblib.load(MODEL_PATH)
    print("✅ Stacking Model Loaded Successfully.")
except Exception as e:
    print(f"❌ Load Error: {e}")

@app.route('/')
def index():
    return render_template('index.html', config=get_config())

@app.route('/settings', methods=['GET', 'POST'])
def settings():
    if request.method == 'POST':
        config = {
            "density_relation": float(request.form['density_relation']),
            "zero_setting": int(request.form['zero_setting']),
            "target_de": float(request.form['target_de'])
        }
        with open(CONFIG_PATH, 'w') as f: json.dump(config, f)
        return render_template('settings.html', config=config, msg="Settings Saved!")
    return render_template('settings.html', config=get_config())

@app.route('/predict_all', methods=['POST'])
def predict_all():
    try:
        zones_input = request.json['zones']
        config = get_config()
        results = []
        log_entries = []

        for zone in zones_input:
            # Match categorical numbers
            paper_val = PAPER_MAP.get(zone['paper_type'], 0)
            color_val = COLOR_MAP.get(zone['color'], 0)
            
            # Align exact 11 columns from your notebook's training set
            de_before = float(zone['de_before'])
            target_de = float(config['target_de'])
            
            features = pd.DataFrame([{
                'Press ID': 1,
                'Job Number': int(zone['job_number']),
                'Paper type': paper_val,
                'Zone number': int(zone['zone_no']),
                'Color': color_val,
                'Ink key zero setting': int(config['zero_setting']),
                'Delta E before': de_before,
                'Delta E after': target_de,
                'initial density': 1.2, # Static default for example
                'final density': 1.2 + float(config['density_relation']),
                'initial ink key setting': float(zone['init_key'])
            }])

            # Run prediction through the StackingRegressor
            prediction = model.predict(features)
            final_key = round(float(prediction[0]), 2)
            results.append({"zone_no": zone['zone_no'], "predicted_key": final_key})

        return jsonify({"status": "success", "results": results})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)