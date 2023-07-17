import pickle
import cv2
import numpy as np
import zope.interface
from DateExtractor import DateExtractor

@zope.interface.implementer(DateExtractor)
class Cifar100Extractor:

    def __init__(self,filePaths,images_path,labels_images=[]):
        array_size = (0, 32, 32, 3)
        self.images = np.empty(array_size)
        self.labels = np.empty(0)
        self.file_names = np.empty(0)
        self.filePaths=filePaths
        self.images_path=images_path
        self.labels_images=labels_images
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
            img_rgb=cv2.cvtColor(self.images[i].astype('uint8') ,cv2.COLOR_RGB2BGR)
            cv2.imwrite(f'{self.images_path}/{self.file_names[i].decode("ascii")}', img_rgb)


    def extract_data(self):
        self.read_all()
        self.write_images()