
from Cifar10Extractor import  Cifar10Extractor
from Cifar100Extractor import Cifar100Extractor


def extract_data():
    cifar10_file_paths = ['../cifar-10-batches-py/data_batch_1',
                          '../cifar-10-batches-py/data_batch_2',
                          '../cifar-10-batches-py/data_batch_3',
                          '../cifar-10-batches-py/data_batch_4',
                          '../cifar-10-batches-py/data_batch_5',
                         '../cifar-10-batches-py/test_batch']

    cifar100_file_paths=['./cifar-100-python/train','./cifar-100-python/test']

    # labels:[b'fish', b'flowers', b'fruit_and_vegetables', b'people', b'trees']
    cifar100_labels=[1,4,2,14,17]

    #extract data for cifar-10
    cifar10Extractor=Cifar10Extractor()

    #read the image
    cifar10Extractor.read_all(cifar10_file_paths)
    #write the image
    cifar10Extractor.write_images('../images')

    # extract data for cifar-100

    cifar100Extractor=Cifar100Extractor()

    # read the image
    cifar10Extractor.read_all(cifar100_file_paths)
    #write the image
    cifar10Extractor.write_images('../images',cifar100_labels)




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    extract_data()



