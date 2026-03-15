from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import json
import os

app = Flask(__name__)

# File Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'ink_model_pipeline.pkl')
EXCEL_PATH = os.path.join(BASE_DIR, 'print_logs.xlsx')
CONFIG_PATH = os.path.join(BASE_DIR, 'press_settings.json')

def get_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, 'r') as f: return json.load(f)
    return {"density_relation": 0.5, "zero_setting": 0.0, "target_de": 2.5}

@app.route('/')
def index():
    return render_template('index.html', config=get_config())

@app.route('/settings', methods=['GET', 'POST'])
def settings():
    if request.method == 'POST':
        config = {"density_relation": float(request.form['density_relation']),
                  "zero_setting": float(request.form['zero_setting']),
                  "target_de": float(request.form['target_de'])}
        with open(CONFIG_PATH, 'w') as f: json.dump(config, f)
        return render_template('settings.html', config=config, msg="System Calibrated!")
    return render_template('settings.html', config=get_config())

@app.route('/predict_all', methods=['POST'])
def predict_all():
    try:
        zones_input = request.json['zones']
        config = get_config()
        model = joblib.load(MODEL_PATH)
        results = []
        log_entries = []

        for zone in zones_input:
            de_before = float(zone['de_before'])
            # Feature Engineering for the model
            input_df = pd.DataFrame([{
                'Press_ID': 1,
                'Job_Number': str(zone['job_number']),
                'Paper_type': str(zone['paper_type']),
                'Zone_number': float(zone['zone_no']),
                'Color': str(zone['color']),
                'Ink_key_zero_setting': float(config['zero_setting']),
                'Delta_E_before': de_before,
                'initial_density': float(zone.get('init_dens', 0) or 0),
                'final_density': float(zone.get('init_dens', 0) or 0) + float(config['density_relation']),
                'initial_ink_key_setting': float(zone['init_key']),
                'DeltaE_reduction': de_before - float(config['target_de'])
            }])

            prediction = model.predict(input_df)
            final_key = round(float(prediction[0]), 2)
            results.append({"zone_no": zone['zone_no'], "predicted_key": final_key})
            
            input_df['final_ink_key_setting'] = final_key
            log_entries.append(input_df)

        if log_entries:
            new_data = pd.concat(log_entries, ignore_index=True)
            if os.path.exists(EXCEL_PATH):
                existing = pd.read_excel(EXCEL_PATH)
                pd.concat([existing, new_data], ignore_index=True).to_excel(EXCEL_PATH, index=False)
            else:
                new_data.to_excel(EXCEL_PATH, index=False)

        return jsonify({"status": "success", "results": results})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/retrain', methods=['POST'])
def retrain():
    try:
        data = pd.read_excel(EXCEL_PATH).dropna(subset=['final_ink_key_setting'])
        X = data.drop(['final_ink_key_setting', 'Delta_E_after'], axis=1, errors='ignore')
        y = data['final_ink_key_setting']
        model_pipe = joblib.load(MODEL_PATH)
        model_pipe.fit(X, y)
        joblib.dump(model_pipe, MODEL_PATH)
        return jsonify({"status": "success", "message": f"Learned from {len(data)} readings!"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)