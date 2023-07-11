from DataPreProcessing import DataPreProcessing

import pickle
def write_images():
   file_paths=['../cifar-10-batches-py/data_batch_1','../cifar-10-batches-py/data_batch_2','../cifar-10-batches-py/data_batch_3','../cifar-10-batches-py/data_batch_4',
    '../cifar-10-batches-py/data_batch_5','../cifar-10-batches-py/data_batch_5','../cifar-10-batches-py/test_batch']
   dataPreProcessing = DataPreProcessing()
   dataPreProcessing.read_all(file_paths)
   print(dataPreProcessing.images.shape)
   print(dataPreProcessing.images[0].shape)
   # print(dataPreProcessing.file_names)
   # print(dataPreProcessing.labels)
   dataPreProcessing.write_images('../images')


def read_binary( filePath):
    with open(filePath, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #read_binary('../cifar-100-python/train')
     write_images()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
