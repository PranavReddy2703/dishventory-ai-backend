from flask import Flask, request, jsonify
import pandas as pd
from io import BytesIO
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enables CORS so your frontend can call this API


@app.route('/predict', methods=['POST'])
def predict():
    if 'data-file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['data-file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.filename.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file)
        else:
            return jsonify({'error': 'Unsupported file format'}), 400

        print(df.head())  # For debugging in console

        # Return dummy response for now
        return jsonify({
            "forecast": "Forecast result will appear here.",
            "raw_materials": "Raw material suggestions will appear here."
        })
    except Exception as e:
        print(e)
        return jsonify({"error": "Failed to process file."}), 500


if __name__ == "__main__":
    app.run(debug=True)
