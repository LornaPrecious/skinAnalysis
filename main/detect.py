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

#Getting class names
path = "C:/Users/HP/Documents/Machine Learning/project/skinAnalysis/dataset/Dataset/train"
class_names = sorted(os.listdir(path))
#print('classes: ', class_names)

def preprocess_image(image_path):
    """ Preprocesses the captured or uploaded image for model prediction """
    img = tf.keras.utils.load_img(
    image_path, target_size=(192, 192)
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch
    return img_array

def predict_skin_condition(image_path):
    """ Predicts the skin condition from the given image """
    processed_image = preprocess_image(image_path)
    predictions = model.predict(processed_image)
    
    # Apply softmax to get confidence scores
    score = tf.nn.softmax(predictions[0]).numpy()
    
    # Get class with highest probability
    predicted_class_index = np.argmax(score)
    
    predicted_class_name = class_names[predicted_class_index]

    return predicted_class_name

def upload_and_predict(image_path):
        predicted_label = predict_skin_condition(image_path)

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


    