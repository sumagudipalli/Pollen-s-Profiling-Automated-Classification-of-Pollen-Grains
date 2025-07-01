import os
import numpy as np
from flask import Flask, request, render_template, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

# Initialize the Flask application
app = Flask(__name__)

# --- Configuration ---
MODEL_PATH = '../pollen_model.keras'  # Path to the model relative to app.py
UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# These are the class names from your dataset.
# IMPORTANT: The order MUST be the same as the one used during training.
# You can get the correct order from your LabelEncoder: le.classes_
CLASS_NAMES = ['anadenanthera', 'arecaceae', 'croton', 'syagrus', 'tridax']


# --- Model Loading ---
# Load your trained model
try:
    model = load_model(MODEL_PATH)
    print(f"Model loaded from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# --- Image Preprocessing Function ---
def model_predict(img_path, model):
    # Load the image
    img = image.load_img(img_path, target_size=(128, 128))
    
    # Preprocess the image
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Make prediction
    preds = model.predict(img_array)
    return preds

# --- Routes ---
@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload():
    if model is None:
        return "Model not loaded. Please check the server logs.", 500

    # Get the file from post request
    f = request.files['file']

    # Save the file to ./uploads
    basepath = os.path.dirname(__file__)
    filepath = os.path.join(basepath, app.config['UPLOAD_FOLDER'], secure_filename(f.filename))
    f.save(filepath)

    # Make prediction
    preds = model_predict(filepath, model)

    # Process your result for human
    pred_class_index = np.argmax(preds)
    result = CLASS_NAMES[pred_class_index]
    
    return render_template('prediction.html', result=result)

# This is for serving files from the 'uploads' directory
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# --- Run the App ---
if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True, port=5001)