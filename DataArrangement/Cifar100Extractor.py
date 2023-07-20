import pickle
import cv2
import numpy as np
import zope.interface
import matplotlib.pyplot as plt
from .DataExtractor import DataExtractor
from scipy.ndimage import rotate


@zope.interface.implementer(DataExtractor)
class Cifar100Extractor:

    def __init__(self, filePaths, images_path, labels_images=[]):
        array_size = (0, 32, 32, 3)
        self.images = np.empty(array_size)
        self.labels = np.empty(0)
        self.file_names = np.empty(0)
        self.filePaths = filePaths
        self.images_path = images_path
        self.labels_images = labels_images

    def read_binary(self, filePath):
        with open(filePath, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    def read_all(self):
        for filePath in self.filePaths:
            dict_ = self.read_binary(filePath)
            data = dict_[b'data'].reshape(len(dict_[b'data']), 3, 32, 32).transpose(0, 2, 3, 1)
            self.images = np.concatenate((self.images, data), axis=0)
            self.labels = np.concatenate((self.labels, dict_[b'coarse_labels']), axis=0)
            self.file_names = np.concatenate((self.file_names, dict_[b'filenames']), axis=0)

    def write_images(self):
        images_filtered = [i for i, label in enumerate(self.labels) if label in self.labels_images]

        for i in images_filtered:
            img_rgb = cv2.cvtColor(self.images[i].astype('uint8'), cv2.COLOR_RGB2BGR)
            cv2.imwrite(f'{self.images_path}/{self.file_names[i].decode("ascii")}', img_rgb)

            rotated_image = self.data_augmentation(img_rgb)  # Assuming this function returns a rotated image

            rotated_file_name = f'rotated_{self.file_names[i].decode("ascii")}'
            cv2.imwrite(f'{self.images_path}/{rotated_file_name}', rotated_image)

            self.labels = np.concatenate((self.labels, [self.labels[i]]))
            self.file_names = np.concatenate((self.file_names, [rotated_file_name.encode("ascii")]))

    def extract_data(self):
        self.read_all()
        self.write_images()


    def data_augmentation(self, image):
        height, width = image.shape[:2]
        diagonal = np.sqrt(height ** 2 + width ** 2)
        padding = int((diagonal - min(height, width)) / 2)

        # Add padding to the image using BORDER_REFLECT or BORDER_REPLICATE mode
        padded_image = cv2.copyMakeBorder(image, padding, padding, padding, padding, cv2.BORDER_REFLECT)

        # Rotate the padded image by 30 degrees
        rotated = rotate(padded_image, angle=30, reshape=False)

        # Remove the padding from the rotated image
        rotated_cropped = rotated[padding:-padding, padding:-padding]

        return rotated_cropped
