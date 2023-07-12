
from Cifar10Extractor import  Cifar10Extractor
from Cifar100Extractor import Cifar100Extractor
from CreateCsv import create_or_add_csv


def extract_data():
    cifar10_file_paths = ['../cifar-10-batches-py/data_batch_1',
                          '../cifar-10-batches-py/data_batch_2',
                          '../cifar-10-batches-py/data_batch_3',
                          '../cifar-10-batches-py/data_batch_4',
                          '../cifar-10-batches-py/data_batch_5',
                          '../cifar-10-batches-py/test_batch']

    cifar100_file_paths=['../cifar-100-python/train','../cifar-100-python/test']

    # labels:[b'fish', b'flowers', b'fruit_and_vegetables', b'people', b'trees']
    cifar100_labels=[1,4,2,14,17]

    #extract data for cifar-10
    cifar10Extractor=Cifar10Extractor()

    # #read the image
    cifar10Extractor.read_all(cifar10_file_paths)
    cifar10Extractor.write_images('../images')
    create_or_add_csv(cifar10Extractor.file_names,cifar10Extractor.labels,'cifar10','../cifar_data.csv')


    # extract data for cifar-100

    cifar100Extractor=Cifar100Extractor()

    # read the image
    cifar100Extractor.read_all(cifar100_file_paths)

    #write the image
    cifar100Extractor.write_images('../images',cifar100_labels)
    create_or_add_csv(cifar100Extractor.file_names,cifar100Extractor.labels,'cifar100','../cifar_data.csv')






# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    extract_data()



