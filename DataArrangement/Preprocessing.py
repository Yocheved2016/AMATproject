from sklearn import preprocessing
import numpy as np



def normalizeData(data):

    x_array = np.array([2,3,5,6,7,4,8,7,6])
    normalized_data = preprocessing.normalize([data])
    print(normalized_data)
    return normalized_data