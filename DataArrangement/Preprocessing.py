from sklearn import preprocessing
import numpy as np



def normalize_data(data):
   
    data = data.astype('float32')
    normalized_data = data / 255.0
    return normalized_data
