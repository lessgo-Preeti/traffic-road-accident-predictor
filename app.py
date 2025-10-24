from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import json
import pandas as pd
import numpy as np
from werkzeug.exceptions import BadRequest
import os
from datetime import datetime

# Initialize Flask app
app = Flask(__name__, static_folder='static')
CORS(app)  # Enable CORS for all routes

# Load model and metadata
MODEL_PATH = 'models/model.pkl'
METADATA_PATH = 'models/model_metadata.json'

print("Loading model...")
model = joblib.load(MODEL_PATH)
print("✓ Model loaded successfully!")

with open(METADATA_PATH, 'r') as f:
    metadata = json.load(f)

FEATURES = metadata['dataset']['features']
CATEGORICAL_FEATURES = metadata['dataset']['categorical_features']
NUMERICAL_FEATURES = metadata['dataset']['numerical_features']

# ==================== ROUTES ====================

@app.route('/')
def home():
    """Serve the main HTML page"""
    return send_from_directory('static', 'index.html')

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model': metadata['best_model'],
        'timestamp': datetime.now().isoformat()
    }), 200

@app.route('/info', methods=['GET'])
def info():
    """Get model information and metadata"""
    return jsonify({
        'model_name': metadata['best_model'],
        'training_date': metadata['training_date'],
        'metrics': metadata['best_model_metrics'],
        'features': {
            'total': len(FEATURES),
            'categorical': CATEGORICAL_FEATURES,
            'numerical': NUMERICAL_FEATURES
        },
        'dataset_info': {
            'train_size': metadata['dataset']['train_size'],
            'test_size': metadata['dataset']['test_size']
        }
    }), 200

@app.route('/schema', methods=['GET'])
def schema():
    """Get input schema with example"""
    # Create example based on actual features
    example = {}
    
    # Add numerical features with default values
    for feat in NUMERICAL_FEATURES:
        if 'hour' in feat.lower():
            example[feat] = 18
        elif 'day' in feat.lower() or 'dow' in feat.lower():
            example[feat] = 4
        elif 'month' in feat.lower():
            example[feat] = 12
        elif 'weekend' in feat.lower() or 'rush' in feat.lower() or 'night' in feat.lower() or 'monsoon' in feat.lower():
            example[feat] = 1
        elif 'junction' in feat.lower() or 'urban' in feat.lower() or 'highway' in feat.lower() or 'signal' in feat.lower() or 'crossing' in feat.lower():
            example[feat] = 1
        else:
            example[feat] = 0
    
    # Add categorical features with default values
    for feat in CATEGORICAL_FEATURES:
        if 'weather' in feat.lower():
            example[feat] = 'Rain'
        elif 'road' in feat.lower():
            example[feat] = 'arterial'
        else:
            example[feat] = 'unknown'
    
    return jsonify({
        'features': FEATURES,
        'categorical_features': CATEGORICAL_FEATURES,
        'numerical_features': NUMERICAL_FEATURES,
        'example': example
    }), 200

@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint"""
    try:
        # Get JSON data
        data = request.get_json(silent=True)
        
        if data is None:
            raise BadRequest("Invalid or missing JSON payload")
        
        # Validate all required features are present
        missing_features = [f for f in FEATURES if f not in data]
        if missing_features:
            return jsonify({
                'error': 'Missing required features',
                'missing_features': missing_features
            }), 400
        
        # Create dataframe with proper column order
        input_data = {feat: [data[feat]] for feat in FEATURES}
        input_df = pd.DataFrame(input_data)
        
        # Make prediction
        prediction_proba = model.predict_proba(input_df)[0]
        prediction_class = int(model.predict(input_df)[0])
        
        # Calculate risk level
        risk_probability = float(prediction_proba[1])
        
        if risk_probability >= 0.7:
            risk_level = 'High Risk'
            risk_color = 'red'
        elif risk_probability >= 0.4:
            risk_level = 'Medium Risk'
            risk_color = 'orange'
        else:
            risk_level = 'Low Risk'
            risk_color = 'green'
        
        # Return response
        return jsonify({
            'prediction': {
                'class': prediction_class,
                'risk_level': risk_level,
                'risk_color': risk_color,
                'risk_probability': round(risk_probability * 100, 2),
                'low_risk_probability': round(prediction_proba[0] * 100, 2),
                'high_risk_probability': round(prediction_proba[1] * 100, 2)
            },
            'input': data,
            'model': metadata['best_model'],
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except BadRequest as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({
            'error': 'Internal server error',
            'details': str(e)
        }), 500

@app.route('/batch', methods=['POST'])
def batch_predict():
    """Batch prediction endpoint for multiple inputs"""
    try:
        data = request.get_json(silent=True)
        
        if data is None or 'inputs' not in data:
            raise BadRequest("Provide 'inputs' array in JSON")
        
        inputs = data['inputs']
        
        if not isinstance(inputs, list):
            raise BadRequest("'inputs' must be an array")
        
        results = []
        
        for idx, single_input in enumerate(inputs):
            try:
                # Validate features
                missing = [f for f in FEATURES if f not in single_input]
                if missing:
                    results.append({
                        'index': idx,
                        'error': f'Missing features: {missing}'
                    })
                    continue
                
                # Create dataframe
                input_df = pd.DataFrame([{f: single_input[f] for f in FEATURES}])
                
                # Predict
                proba = model.predict_proba(input_df)[0]
                pred_class = int(model.predict(input_df)[0])
                
                results.append({
                    'index': idx,
                    'prediction_class': pred_class,
                    'risk_probability': round(float(proba[1]) * 100, 2)
                })
                
            except Exception as e:
                results.append({
                    'index': idx,
                    'error': str(e)
                })
        
        return jsonify({
            'results': results,
            'total': len(inputs),
            'successful': sum(1 for r in results if 'error' not in r)
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ==================== ERROR HANDLERS ====================

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error'}), 500

# ==================== MAIN ====================

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port, debug=True)
