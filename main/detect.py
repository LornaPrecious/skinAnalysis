import cv2
import numpy as np
import tensorflow as tf
import os

# Load the trained model
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Get the project root directory
MODEL_PATH = os.path.join(BASE_DIR, "skin_classifier_model.keras")  # Adjust if needed

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")

model = tf.keras.models.load_model(MODEL_PATH)

# Image parameters (must match training)
img_size = (128, 128)

# Class labels (assuming same order as training)
class_labels = ['acne', 'eczema', 'melasma', 'psoriasis', 'normal'] 

def preprocess_image(image_path):
    """ Preprocesses the captured or uploaded image for model prediction """
    image = cv2.imread(image_path)  
    image = cv2.resize(image, img_size) 
    image = image / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  
    return image

def predict_skin_condition(image_path):
    """ Predicts the skin condition from the given image """
    processed_image = preprocess_image(image_path)
    predictions = model.predict(processed_image)
    predicted_class = np.argmax(predictions)  # Get highest probability class
    predicted_label = class_labels[predicted_class]  # Get class label
    return predicted_label

def capture_and_predict():
    """ Captures an image from the webcam and predicts the skin condition """
    video_capture = cv2.VideoCapture(0)
    
    while True:
        check, frame_image = video_capture.read()
        if not check:
            break

        # Predict the skin condition
        predicted_label = predict_skin_condition(frame_image)

        # Display prediction
        cv2.putText(frame_image, f"Prediction: {predicted_label}", (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show video feed with prediction
        cv2.imshow('Skin Analysis', frame_image)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release camera and close OpenCV windows
    video_capture.release()
    cv2.destroyAllWindows()

def upload_and_predict(image_path):
        predicted_label = predict_skin_condition(image_path)
        
        # Read image for display
        image = cv2.imread(image_path)
        cv2.putText(image, f"Prediction: {predicted_label}", (10, 50), 
                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Uploaded Image Analysis", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return predicted_label


