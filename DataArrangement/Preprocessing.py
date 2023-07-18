from sklearn import preprocessing
import numpy as np



def normalize_data(data):
    shape_data = data.shape
    flatten_data = data.flatten()
    normalized_data = preprocessing.normalize([flatten_data])
    normalized_data = normalized_data.reshape(shape_data)

    return normalized_data