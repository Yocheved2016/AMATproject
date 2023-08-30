import io
import tensorflow as tf
import numpy as np
import cv2
import pickle
from keras.models import Model
import sklearn
class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck", "fish",
               "flowers", "fruits and vegetables", "people", "trees"]

model = tf.keras.models.load_model('best_model.h5')


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


from PIL import Image

def preprocess_image(image):
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

def anomalys_detection(image):

    anomalys_detector = pickle.load(open('../model/anomalys_detector_v3_clf_.sav', 'rb'))
    _model=Model(inputs=model.input,outputs=model.get_layer('dense_5').output)
    med_prediction=_model.predict(image)
    is_anom=anomalys_detector.predict(med_prediction)

    return is_anom ==-1

def predict_image(image_content):
    img = preprocess_image(image_content)
    is_anomaly=anomalys_detection(img)
    if is_anomaly==True:
        return 'ood'
    prediction = model.predict(img)[0]
    predicted_class = np.argmax(prediction)
    print(prediction)
    return class_names[predicted_class]
