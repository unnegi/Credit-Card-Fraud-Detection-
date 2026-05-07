import os
import pandas as pd
import numpy as np
from flask import Flask, request, render_template, url_for, Response
import joblib
import io
import csv

app = Flask(__name__)
app.secret_key = 'fraud-detection-secret-key-2024'  # Needed for flash messages and session

def analyze_risk_factors(transaction, prediction, proba):
    """Analyze risk factors for a transaction"""
    risk_factors = []
    risk_score = 0
    
    # Amount-based risk
    amount = transaction.get('Amount', 0)
    if amount > 10000:
        risk_factors.append({
            'factor': 'High Transaction Amount',
            'description': f'Transaction amount (${amount:.2f}) exceeds $10,000 threshold',
            'severity': 'high'
        })
        risk_score += 30
    elif amount > 5000:
        risk_factors.append({
            'factor': 'Elevated Transaction Amount',
            'description': f'Transaction amount (${amount:.2f}) is above $5,000',
            'severity': 'medium'
        })
        risk_score += 15
    
    # Time-based risk (if transaction happens at unusual hours)
    time = transaction.get('Time', 0)
    hour = (time % 86400) / 3600  # Convert to hour of day
    if hour < 6 or hour > 22:
        risk_factors.append({
            'factor': 'Unusual Transaction Time',
            'description': f'Transaction occurred at {hour:.1f}:00 (outside normal business hours)',
            'severity': 'medium'
        })
        risk_score += 20
    
    # V feature analysis (PCA components - look for extreme values)
    v_features = {f'V{i}': transaction.get(f'V{i}', 0) for i in range(1, 29)}
    extreme_values = []
    for key, value in v_features.items():
        if abs(value) > 3:  # More than 3 standard deviations
            extreme_values.append((key, value))
    
    if len(extreme_values) > 5:
        risk_factors.append({
            'factor': 'Multiple Anomalous Features',
            'description': f'{len(extreme_values)} features show extreme values (possible data manipulation)',
            'severity': 'high'
        })
        risk_score += 25
    
    # Model confidence
    fraud_prob = proba[1] if len(proba) > 1 else proba[0]
    if fraud_prob > 0.9:
        risk_factors.append({
            'factor': 'Very High Fraud Probability',
            'description': f'Model confidence: {fraud_prob*100:.2f}%',
            'severity': 'critical'
        })
        risk_score += 40
    elif fraud_prob > 0.7:
        risk_factors.append({
            'factor': 'High Fraud Probability',
            'description': f'Model confidence: {fraud_prob*100:.2f}%',
            'severity': 'high'
        })
        risk_score += 25
    
    # Calculate final risk level
    if risk_score >= 70:
        risk_level = 'Critical'
    elif risk_score >= 50:
        risk_level = 'High'
    elif risk_score >= 30:
        risk_level = 'Medium'
    else:
        risk_level = 'Low'
    
    return {
        'factors': risk_factors,
        'risk_score': min(risk_score, 100),
        'risk_level': risk_level,
        'recommendation': get_recommendation(risk_level, prediction)
    }

def get_recommendation(risk_level, prediction):
    """Get recommendation based on risk level"""
    if prediction == 1:
        if risk_level == 'Critical':
            return 'Immediate transaction blocking recommended. Contact cardholder immediately.'
        elif risk_level == 'High':
            return 'Transaction should be flagged for manual review. Consider additional authentication.'
        else:
            return 'Transaction flagged. Monitor for suspicious patterns.'
    else:
        if risk_level == 'Low':
            return 'Transaction appears legitimate. Normal processing recommended.'
        else:
            return 'Transaction appears legitimate but monitor for patterns.'

# --- Load the saved model and scalers ---
print("Loading model and scalers...")
try:
    model_path = os.path.join(os.path.dirname(__file__), 'models', 'fraud_detection_model.pkl')
    scaler_path = os.path.join(os.path.dirname(__file__), 'models', 'scaler.pkl')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
    
    model = joblib.load(model_path)
    scalers = joblib.load(scaler_path)
    
    # Handle both old format (single scaler) and new format (dict of scalers)
    if isinstance(scalers, dict):
        scaler_amount = scalers['amount']
        scaler_time = scalers['time']
    else:
        # Legacy support: if only one scaler is saved, use it for both (not ideal but won't crash)
        print("Warning: Using legacy single scaler format. Consider retraining model with separate scalers.")
        scaler_amount = scalers
        scaler_time = scalers
    
    print("Model and scalers loaded successfully.")
except Exception as e:
    print(f"ERROR: Failed to load model or scalers: {str(e)}")
    print("Please ensure the model files exist in the 'models' directory.")
    raise

# Allowed file extension
ALLOWED_EXTENSIONS = {'csv'}
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- Define the landing page route ---
@app.route('/')
def home():
    return render_template('landing.html')

# --- Define the analysis page route ---
@app.route('/analyze')
def analyze():
    # Clear any previous results/reset state on fresh page load
    return render_template('index.html', batch_predictions=None, batch_summary=None, single_prediction=None, error_csv=None)

# --- CSV Template Download Route ---
@app.route('/download_template')
def download_template():
    try:
        # Create a CSV template with sample headers
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write header row
        headers = ['Time', 'Amount'] + [f'V{i}' for i in range(1, 29)]
        writer.writerow(headers)
        
        # Write a sample row with zeros
        sample_row = [0, 0.0] + [0.0] * 28
        writer.writerow(sample_row)
        
        output.seek(0)
        
        return Response(
            output.getvalue(),
            mimetype='text/csv',
            headers={'Content-Disposition': 'attachment; filename=fraud_guard_template.csv'}
        )
    except Exception as e:
        print(f"Error generating template: {str(e)}")
        return Response(
            "Error generating template. Please try again.",
            status=500,
            mimetype='text/plain'
        )

# --- Define the prediction route ---
@app.route('/predict', methods=['POST'])
def predict():
    action = request.form.get('action')

    if action == 'predict_single':
        try:
            # --- Single Prediction Logic ---
            form_values = [request.form[f'V{i}'] for i in range(1, 29)] + [request.form['Time'], request.form['Amount']]
            # Check if all fields are filled
            if any(val == '' for val in form_values):
                empty_fields = [f'V{i}' if i < 28 else ('Time' if i == 28 else 'Amount') for i, val in enumerate(form_values) if val == '']
                error_msg = f"Error: Please fill all fields. Missing: {', '.join(empty_fields[:5])}{'...' if len(empty_fields) > 5 else ''}"
                return render_template('index.html', batch_predictions=None, batch_summary=None, single_prediction={'text': error_msg, 'prob': ''}, error_csv=None)

            # Validate and convert form values to float with error handling
            try:
                v_features = [float(request.form[f'V{i}']) for i in range(1, 29)]
                time_val = float(request.form['Time'])
                amount_val = float(request.form['Amount'])
            except ValueError as e:
                error_msg = f"Error: Invalid numeric value in form fields. Please ensure all fields contain valid numbers. ({str(e)})"
                return render_template('index.html', batch_predictions=None, batch_summary=None, single_prediction={'text': error_msg, 'prob': ''}, error_csv=None)

            # Scale Amount and Time using their respective scalers
            scaled_amount_array = scaler_amount.transform(np.array([[amount_val]]))
            scaled_time_array = scaler_time.transform(np.array([[time_val]]))
            scaled_amount = scaled_amount_array[0, 0]
            scaled_time = scaled_time_array[0, 0]

            final_features_list = v_features + [scaled_amount, scaled_time]
            final_features = np.array(final_features_list).reshape(1, -1)

            # Make prediction with error handling
            try:
                prediction = model.predict(final_features)
                prediction_proba = model.predict_proba(final_features)
            except Exception as e:
                error_msg = f"Error during model prediction: {str(e)}. Please check your input values."
                print(f"Prediction error: {error_msg}")
                return render_template('index.html', batch_predictions=None, batch_summary=None, single_prediction={'text': error_msg, 'prob': ''}, error_csv=None)

            if prediction[0] == 1:
                result_text = "This transaction is likely to be FRAUDULENT."
                probability = f"Model Confidence: {prediction_proba[0][1]*100:.2f}%"
            else:
                result_text = "This transaction appears to be LEGITIMATE."
                probability = f"Confidence: {prediction_proba[0][0]*100:.2f}%"
            
            single_pred_result = {'text': result_text, 'prob': probability}
            print(f"✓✓✓ Single prediction result: {single_pred_result}")  # Debug
            print(f"   Text: {result_text}")
            print(f"   Prob: {probability}")
            # Render template with results
            response = render_template('index.html', batch_predictions=None, batch_summary=None, single_prediction=single_pred_result, error_csv=None)
            print(f"✓ Template rendered with single_prediction: {single_pred_result is not None}")
            return response

        except Exception as e:
            import traceback
            error_msg = f"An error occurred: {str(e)}"
            print(f"Error in single prediction: {error_msg}")
            print(traceback.format_exc())
            return render_template('index.html', batch_predictions=None, batch_summary=None, single_prediction={'text': error_msg, 'prob': ''}, error_csv=None)

    elif action == 'predict_csv':
        try:
            # --- Batch Prediction Logic ---
            if 'csv_file' not in request.files:
                return render_template('index.html', batch_predictions=None, batch_summary=None, single_prediction=None, error_csv='No file part in the request.')
            
            file = request.files['csv_file']
            if file.filename == '':
                return render_template('index.html', batch_predictions=None, batch_summary=None, single_prediction=None, error_csv='No file selected.')

            if file and allowed_file(file.filename):
                try:
                    df = pd.read_csv(file)
                    
                    # Check if dataframe is empty
                    if df.empty:
                        return render_template('index.html', batch_predictions=None, batch_summary=None, single_prediction=None, error_csv='CSV file is empty. Please upload a file with transaction data.')
                    
                    # Check for required columns
                    required_cols = ['Time', 'Amount'] + [f'V{i}' for i in range(1, 29)]
                    missing_cols = [col for col in required_cols if col not in df.columns]
                    if missing_cols:
                        return render_template('index.html', batch_predictions=None, batch_summary=None, single_prediction=None, error_csv=f'CSV file is missing required columns: {", ".join(missing_cols)}')
                    
                    # Check for non-numeric values in required columns
                    for col in required_cols:
                        if col in df.columns:
                            # Try to convert to numeric, coercing errors to NaN
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                            if df[col].isna().any():
                                return render_template('index.html', batch_predictions=None, batch_summary=None, single_prediction=None, error_csv=f'Column "{col}" contains non-numeric values. Please ensure all values are numbers.')
                    
                    # Check if dataframe has rows after cleaning
                    if df.empty:
                        return render_template('index.html', batch_predictions=None, batch_summary=None, single_prediction=None, error_csv='CSV file has no valid data rows after processing.')
                    
                except pd.errors.EmptyDataError:
                    return render_template('index.html', batch_predictions=None, batch_summary=None, single_prediction=None, error_csv='CSV file is empty or invalid.')
                except pd.errors.ParserError as e:
                    return render_template('index.html', batch_predictions=None, batch_summary=None, single_prediction=None, error_csv=f'Error parsing CSV file: {str(e)}')
                except Exception as e:
                    return render_template('index.html', batch_predictions=None, batch_summary=None, single_prediction=None, error_csv=f'Error reading CSV file: {str(e)}')

                # Store original Time and Amount for display
                original_time = df['Time']
                original_amount = df['Amount']

                # Preprocess the entire DataFrame - scale Amount and Time using their respective scalers
                # Note: scaler.transform() expects 2D array with shape (n_samples, n_features)
                df['scaled_amount'] = scaler_amount.transform(df[['Amount']].values).flatten()
                df['scaled_time'] = scaler_time.transform(df[['Time']].values).flatten()
                
                # Prepare the final feature set in the correct order for prediction
                prediction_features = df[[f'V{i}' for i in range(1, 29)] + ['scaled_amount', 'scaled_time']]
                
                # Get predictions and probabilities for the whole file with error handling
                try:
                    predictions = model.predict(prediction_features)
                    predictions_proba = model.predict_proba(prediction_features)
                except Exception as e:
                    return render_template('index.html', batch_predictions=None, batch_summary=None, single_prediction=None, error_csv=f'Error during model prediction: {str(e)}. Please check your CSV file format and data values.')

                # Format results for display
                results = []
                fraud_count = 0
                legit_count = 0
                total_amount = 0
                fraud_amount = 0
                
                for i in range(len(df)):
                    if predictions[i] == 1:
                        pred_text = "FRAUDULENT"
                        prob = f"{predictions_proba[i][1]*100:.2f}"
                        fraud_count += 1
                        fraud_amount += original_amount.iloc[i]
                    else:
                        pred_text = "LEGITIMATE"
                        prob = f"{predictions_proba[i][0]*100:.2f}"
                        legit_count += 1
                    
                    total_amount += original_amount.iloc[i]
                    
                    # Analyze risk factors for fraud detection
                    risk_factors = analyze_risk_factors(df.iloc[i], predictions[i], predictions_proba[i])
                    
                    results.append({
                        'time': original_time.iloc[i],
                        'amount': f"{original_amount.iloc[i]:.2f}",
                        'prediction': pred_text,
                        'probability': prob,
                        'risk_factors': risk_factors,
                        'index': i
                    })
                
                # Calculate summary statistics
                total_count = len(results)
                fraud_percentage = (fraud_count / total_count * 100) if total_count > 0 else 0
                legit_percentage = (legit_count / total_count * 100) if total_count > 0 else 0
                
                summary = {
                    'total_transactions': total_count,
                    'fraudulent': fraud_count,
                    'legitimate': legit_count,
                    'fraud_percentage': f"{fraud_percentage:.2f}",
                    'legit_percentage': f"{legit_percentage:.2f}",
                    'total_amount': f"{total_amount:.2f}",
                    'fraud_amount': f"{fraud_amount:.2f}"
                }
                
                return render_template('index.html', batch_predictions=results, batch_summary=summary, single_prediction=None, error_csv=None)
            else:
                return render_template('index.html', batch_predictions=None, batch_summary=None, single_prediction=None, error_csv='Invalid file type. Please upload a CSV file.')

        except Exception as e:
            import traceback
            error_msg = f"An error occurred while processing the file: {str(e)}"
            print(f"Error in CSV prediction: {error_msg}")
            print(traceback.format_exc())
            return render_template('index.html', batch_predictions=None, batch_summary=None, single_prediction=None, error_csv=error_msg)
    else:
        # No action or unrecognized action
        if action:
            return render_template('index.html', batch_predictions=None, batch_summary=None, single_prediction=None, error_csv=f"Unrecognized action: {action}")
        else:
            return render_template('index.html', batch_predictions=None, batch_summary=None, single_prediction=None, error_csv="No action specified. Please use one of the buttons to submit.")


if __name__ == '__main__':
    app.run(debug=True)