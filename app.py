from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

# Create Flask app
app = Flask(__name__)

# Load the trained AutoML model
model = joblib.load('automl_model.joblib')

# Define the home route to serve a form for input
@app.route('/')
def home():
    return render_template('index.html')

# Define a route to handle prediction requests
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the form (example: a simple numeric input)
        input_data = [float(x) for x in request.form.values()]
        input_array = np.array(input_data).reshape(1, -1)  # Reshape for the model
        
        # Make prediction using the loaded model
        prediction = model.predict(input_array)
        
        # Return result as JSON
        return jsonify({'prediction': int(prediction[0])})
    
    except Exception as e:
        return jsonify({'error': str(e)})

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
