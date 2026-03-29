from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
# Load the trained model and imputer
model = joblib.load('logistic_regression_model.pkl')
imputer = joblib.load('simple_imputer.pkl')

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        df_pred = pd.DataFrame([data])

        # Ensure correct column order
        feature_columns = [
            'Pregnancies', 'Glucose', 'BloodPressure', 
            'SkinThickness', 'Insulin', 'BMI', 
            'DiabetesPedigreeFunction', 'Age'
        ]
        df_pred = df_pred[feature_columns]

        # Apply imputation
        columns_to_impute = [
            'Glucose', 'BloodPressure', 
            'SkinThickness', 'Insulin', 'BMI'
        ]
        df_pred[columns_to_impute] = imputer.transform(df_pred[columns_to_impute])

        prediction = model.predict(df_pred)
        output = {'prediction': int(prediction[0])}

        return jsonify(output)

    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True)
