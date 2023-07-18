import cv2
import numpy as np
from CreateCsv import create_or_add_csv
def turn_image_to_square(path):

    img = cv2.imread(path)

    height, width, channels = img.shape
    diff=(width-height)//2
    left=diff
    right=width-diff
    #slice the side edges in order to make the square
    img=img[:,left:right]
    cv2.imwrite(path, img)


def resize_image(path):

    image = cv2.imread(path)
    resized_image = cv2.resize(image, (32, 32))
    cv2.imwrite(path, resized_image)


def add_image(path,lable):
    turn_image_to_square(path)
    resize_image(path)
    #transform lable
    #write it to csv

from sklearn import