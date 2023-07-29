import base64
import io
from PIL import Image
import cv2
import numpy as np
from keras.models import load_model


class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, image = self.video.read()
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()


img_width = 1100
img_height = 650
scale_factor = 0.5


def image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def base64_to_image(base64_string):
    image_bytes = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(image_bytes))


def turn_image_to_square(path):
    img = cv2.imread(path)
    height, width, channels = img.shape
    diff = (width - height) // 2
    left = diff
    right = width - diff
    # slice the side edges in order to make the square
    img = img[:, left:right]
    cv2.imwrite(path, img)


def resize_image(path):
    image = cv2.imread(path)
    resized_image = cv2.resize(image, (32, 32))
    cv2.imwrite(path, resized_image)


def get_prediction(model_path, img):
    model = load_model(model_path)
    print(model)
    img_path = 'image.png'
    cv2.imwrite(img_path, img)
    turn_image_to_square(img_path)
    resize_image(img_path)
    img = cv2.imread(img_path)
    img = np.expand_dims(img, axis=0)  # Add a batch dimension
    probabilities = model.predict(img)
    predicted_class = np.argmax(probabilities)
    return predicted_class
