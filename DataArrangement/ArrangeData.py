from DataArrangement.Cifar100Extractor import Cifar100Extractor
from DataArrangement.Cifar10Extractor import Cifar10Extractor
from DataArrangement.CreateCsv import create_or_add_csv


def arrange_data(cifar10_file_paths,cifar100_file_paths,cifar100_labels,images_path):

    # images_path = '../images'

    #
    # cifar100_file_paths = ['../cifar-100-python/train',
    #                        '../cifar-100-python/test']

    # labels:[b'fish', b'flowers', b'fruit_and_vegetables', b'people', b'trees']
    # cifar100_labels = [1, 4, 2, 14, 17]

    # extract data for cifar-10



    cifar10Extractor = Cifar10Extractor(cifar10_file_paths,images_path)
    cifar10Extractor.extract_data()

    #create_or_add_csv(cifar10Extractor.file_names, cifar10Extractor.labels, 'cifar10', "../data/data.csv")


    # extract data for cifar-100
    cifar100Extractor = Cifar100Extractor(cifar100_file_paths,images_path,cifar100_labels)
    cifar100Extractor.extract_data()

   # create_or_add_csv(cifar100Extractor.file_names, cifar100Extractor.labels, 'cifar100', "../data/data.csv")



