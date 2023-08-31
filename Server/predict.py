import logging
import pickle
import io
import cv2
import numpy as np
import tensorflow as tf
from keras.models import Model

class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck", "fish",
               "flowers", "fruits and vegetables", "people", "trees"]

# Configure logging
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the deep learning model
model = tf.keras.models.load_model('../model/best_model.h5')

# Load the anomaly detection model
anomalys_detector = pickle.load(open('../model/anomalys_detector_v2.sav', 'rb'))

# Define image preprocessing functions

def resize_image(img):
    resized_image = cv2.resize(img, (32, 32))
    return resized_image


def normalize_and_standarlize(img):
    # Scale pixel values between 0 and 1
    img = img.astype(np.float32) / 255.0
    # Standardize
    mean_per_channel = np.mean(img, axis=(0, 1))
    std_per_channel = np.std(img, axis=(0, 1))
    img = (img - mean_per_channel) / std_per_channel

    return img

def preprocess_image(image):
    try:
        # Convert the PIL Image to bytes format
        with io.BytesIO() as byte_io:
            # Convert the image to RGB mode (remove the alpha channel if present)
            image_rgb = image.convert("RGB")
            image_rgb.save(byte_io, format='JPEG')
            image_content = byte_io.getvalue()

        img = cv2.imdecode(np.frombuffer(image_content, dtype=np.uint8), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB format
        img = resize_image(img)
        img = normalize_and_standarlize(img)
        img = np.expand_dims(img, axis=0)
        return img
    except Exception as e:
        logging.error(f"Error in preprocess_image: {str(e)}")

def anomalys_detection(image):
    try:
        anomalys_detector = pickle.load(open('../model/anomalys_detector_v2.sav', 'rb'))
        _model=Model(inputs=model.input,outputs=model.get_layer('dense_5').output)
        med_prediction=_model.predict(image)
        is_anom=anomalys_detector.predict(med_prediction)
        return is_anom ==-1
    except Exception as e:
        logging.error(f"Error in anomalys_detection: {str(e)}")

def predict_image(image_content):
    try:
        img = preprocess_image(image_content)
        cv2.imwrite("saved_image_after_processing.jpg", img)
        is_anomaly=anomalys_detection(img)
        if is_anomaly==True:
            return 'ood'
        prediction = model.predict(img)[0]
        predicted_class = np.argmax(prediction)
        logging.info(f"Prediction: {prediction}")
        return class_names[predicted_class]
    except Exception as e:
        logging.error(f"Error in predict_image: {str(e)}")


def calculate_average_entropy_and_histogram(image):
    
    
    avg_histograms = np.zeros((3, 10)) 

    for channel in range(3): 
        pixel_values = image[:, :, channel].flatten()
        histogram, _ = np.histogram(pixel_values, bins=10, range=(0, 256))
        avg_histograms[channel] = histogram   

    probabilities = avg_histograms / np.sum(avg_histograms, axis=1, keepdims=True)
    
    entropy = -np.sum(probabilities * np.log2(np.maximum(probabilities, 1e-10)), axis=1)

    average_entropy = np.mean(entropy)

    return average_entropy, avg_histograms


def get_avg_histogram(label):
    loaded_data = np.load('../model/histograms_all_classes.npz')
    avg_histogram = loaded_data[label]
    return avg_histogram

def get_distance(image,prediction,img_histogram):
    avg_histogram=get_avg_histogram(prediction)
    distance = np.sqrt(np.sum((img_histogram - avg_histogram)**2))
    return distance
