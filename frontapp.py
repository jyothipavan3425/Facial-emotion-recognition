import os
import joblib
import cv2
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

# Initialize the Flask app and enable CORS
app = Flask(__name__)
CORS(app)

# Load the trained models
lda_model = joblib.load('lda_model.pkl')
svm_model = joblib.load('svm_model.pkl')


# List of emotions
emotions = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Set the upload folder and allowed extensions
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'gif'}

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Function to check if the file has an allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Image Preprocessing Function
def preprocess_image(filepath):
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (64, 64))  # Resize as per the training setup
    return img.flatten() / 255.0  # Flatten and normalize the image

# Route to upload the image and predict emotion
@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part in the request'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected for uploading'}), 400
        
        if file and allowed_file(file.filename):
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)

            # Preprocess the image and make a prediction
            processed_img = preprocess_image(filepath)
            if processed_img is not None:
                # Apply LDA transformation and prediction
                processed_img = np.reshape(processed_img, (1, -1))  # Reshape for the model
                processed_img_lda = lda_model.transform(processed_img)
                predicted_label = svm_model.predict(processed_img_lda)

                predicted_emotion = emotions[predicted_label[0]]

                return jsonify({
                    'message': 'File uploaded successfully!',
                    'filename': file.filename,
                    'predicted_emotion': predicted_emotion
                }), 200
            else:
                return jsonify({'error': 'Error in processing the image'}), 500
        else:
            return jsonify({'error': 'Invalid file type'}), 400

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
