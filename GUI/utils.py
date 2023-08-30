import base64
import io
from PIL import Image
import cv2
import requests
from GUI.predict_image import predict_image

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

def predict(image):
    predicted_class = predict_image(image)
    # response = requests.post("http://localhost:8000/predict", {image: image})
    # if response.status_code == 200:
    #     data = response.json()
    #     predicted_class = data
    # else:
    #     predicted_class = 'internal error'
    return f' {predicted_class}.'

def user_feedback(image, correctness, correct_class=None):
    # response = requests.post("http://localhost:8000/send_feedback", {image: image,
    #                                                                  correctness: correctness,
    #                                                                  correct_class: correct_class})
    # print(response)
    pass


classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck", "fish",
               "flowers", "fruits and vegetables", "people", "trees"]