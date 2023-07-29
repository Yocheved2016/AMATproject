from DataArrangement.CreateCsv import train_test_validation_split
from DataArrangement.ArrangeData import arrange_data
from Visualization.Visualization import DataVisualization

if __name__ == '__main__':
    cifar10_file_paths = ['./../cifar-10-batches-py/data_batch_1',
                          './../cifar-10-batches-py/data_batch_2',
                          './../cifar-10-batches-py/data_batch_3',
                          './../cifar-10-batches-py/data_batch_4',
                          './../cifar-10-batches-py/data_batch_5',
                          './../cifar-10-batches-py/test_batch']

    cifar100_file_paths = ['./../cifar-100-python/train',
                           './../cifar-100-python/test']

    cifar100_labels = [1, 4, 2, 14, 17]

    arrange_data(cifar10_file_paths, cifar100_file_paths, cifar100_labels, "./../images")

    train_test_validation_split(0.15, 0.15)

    # dv = DataVisualization()
    # dv.train_image_samples('./data/data.csv','labels.csv')
