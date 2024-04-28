import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow_addons as tfa
from flask import Flask, request, render_template

app = Flask(__name__)

# Load the trained VGG19 model with custom_objects
custom_objects = {'F1Score': tfa.metrics.F1Score}
best_model = load_model('ensemble_model.h5', custom_objects=custom_objects)

# Define a dictionary to map class indices to class labels
label_dict = {
    0: 'bacterial_leaf_blight',
    1: 'bacterial_leaf_streak',
    2: 'bacterial_panicle_blight',
    3: 'blast',
    4: 'brown_spot',
    5: 'dead_heart',
    6: 'downey_mildew',
    7: 'hispa',
    8: 'normal',
    9: 'tungro',
    # Add more class labels as needed
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return render_template('index.html', result="No file selected.")
    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', result="No file selected.")

    if file:
        # Save the uploaded image temporarily
        uploaded_image_path = "C:/Users/hii/Desktop/Project/uploads/uploaded_image.jpg"
        file.save(uploaded_image_path)
        # Classify the uploaded image
        img = image.load_img(uploaded_image_path, target_size=(256, 256))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = img / 255.0  # Normalize pixel values, matching what you did during training
        predictions = best_model.predict(img)
        predicted_class = np.argmax(predictions, axis=1)
        predicted_category = label_dict[predicted_class[0]]
        return render_template('index.html', result=predicted_category, image_path=uploaded_image_path)

if __name__ == '__main__':
    app.run(debug=True)
