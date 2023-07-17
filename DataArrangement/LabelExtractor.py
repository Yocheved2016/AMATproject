import pickle
class LabelMapperCifar10:

    def __init__(self):
        self.meta_path = '../../cifar-10-batches-py/batches.meta'

    def load_meta(self, file_path):
        with open(file_path, 'rb') as fo:
            meta_dict = pickle.load(fo, encoding = 'bytes')
        return meta_dict


    def get_mappings(self):
        meta_dict = self.load_meta(self.meta_path)
        label_names = meta_dict[b'label_names']
        label_mapping = {label_num: label_name.decode('utf-8') for label_num, label_name in enumerate(label_names)}
        return label_mapping


def load_cifar100_labels(file_path):
    with open(file_path, 'rb') as fo:
        label_dict = pickle.load(fo, encoding='bytes')
    return label_dict[b'fine_label_names'], label_dict[b'coarse_label_names']


