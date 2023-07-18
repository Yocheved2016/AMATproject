from .Cifar100Extractor import Cifar100Extractor
from .Cifar10Extractor import Cifar10Extractor
from .CreateCsv import create_or_add_csv, train_test_validation_split


def arrange_data(cifar10_file_paths,cifar100_file_paths,cifar100_labels,images_path):

    # extract data for cifar-10

    cifar10Extractor = Cifar10Extractor(cifar10_file_paths,images_path)
    cifar10Extractor.extract_data()

    cifar100Extractor = Cifar100Extractor(cifar100_file_paths,images_path,cifar100_labels)
    cifar100Extractor.extract_data()

    create_or_add_csv(cifar10Extractor.file_names, cifar10Extractor.labels, 'cifar10',"./data/data.csv")

    create_or_add_csv(cifar100Extractor.file_names, cifar100Extractor.labels, 'cifar100', "./data/data.csv")


