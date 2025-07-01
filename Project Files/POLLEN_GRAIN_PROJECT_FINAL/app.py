from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
model = load_model('model.h5')
labels = ['class1', 'class2', 'class3']  # Replace with actual pollen classes

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400
    file = request.files['file']
    path = os.path.join('flask/uploads', file.filename)
    file.save(path)
    img = image.load_img(path, target_size=(150, 150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    result = labels[np.argmax(prediction)]
    return render_template('prediction.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
