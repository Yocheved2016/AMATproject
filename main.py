from Cifar10Extractor import Cifar10Extractor
from Cifar100Extractor import Cifar100Extractor
from CreateCsv import create_or_add_csv
from Visualization import DataVisualization


def arrange_data():

    images_path = '../images'
    cifar10_file_paths = ['../cifar-10-batches-py/data_batch_1',
                          '../cifar-10-batches-py/data_batch_2',
                          '../cifar-10-batches-py/data_batch_3',
                          '../cifar-10-batches-py/data_batch_4',
                          '../cifar-10-batches-py/data_batch_5',
                          '../cifar-10-batches-py/test_batch']

    cifar100_file_paths = ['../cifar-100-python/train',
                           '../cifar-100-python/test']

    # labels:[b'fish', b'flowers', b'fruit_and_vegetables', b'people', b'trees']
    cifar100_labels = [1, 4, 2, 14, 17]

    # extract data for cifar-10
    cifar10Extractor = Cifar10Extractor(cifar10_file_paths,images_path)
    cifar10Extractor.extract_data()

    create_or_add_csv(cifar10Extractor.file_names, cifar10Extractor.labels, 'cifar10', '../cifar_data.csv')


    # extract data for cifar-100
    cifar100Extractor = Cifar100Extractor(cifar100_file_paths,images_path,cifar100_labels)
    cifar100Extractor.extract_data()

    create_or_add_csv(cifar100Extractor.file_names, cifar100Extractor.labels, 'cifar100', '../cifar_data.csv')



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #extract_data()
    dv = DataVisualization()
    dv.tSNE('../cifar_data_update.csv', '../images')

