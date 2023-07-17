from DataArrangement.ArrangeData import arrange_data


# Press the green button in the gutter to run the script.
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

    arrange_data(cifar10_file_paths,cifar100_file_paths,cifar100_labels,"./../images")


