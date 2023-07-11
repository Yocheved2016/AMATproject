
import pickle
import cv2
import numpy as np


class DataPreProcessing:

    def __init__(self):
        array_size = (0, 32, 32, 3)
        self.images = np.empty(array_size)
        self.labels = np.empty(0)
        self.file_names = np.empty(0)

    def read_binary(self, filePath):
            with open(filePath, 'rb') as fo:
                dict= pickle.load(fo, encoding='bytes')
            return dict


    def read_all(self, filePaths):
        for filePath in filePaths:
            dict_= self.read_binary(filePath)
            self.images = np.append (self.images ,dict_[b'data'].reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1))
            self.labels = np.append(self.labels,dict_[b'labels'])
            self.file_names = np.append(self.file_names,dict_[b'filenames'])


    def write_images(self,path):

        for i in range(len(self.images)):
            img=self.images[i]
            path = f'{path}/{self.file_names[i].decode("ascii")}'
            cv2.imwrite(path, img)
            a=5








