import pickle
import csv

def load_cifar10_labels(file_path):

    with open(file_path, 'rb') as fo:
        meta_dict = pickle.load(fo, encoding='bytes')
    label_names = meta_dict[b'label_names']
    label_mapping = {label_num: label_name.decode('utf-8') for label_num, label_name in enumerate(label_names)}
    return label_mapping


def load_cifar100_labels(file_path):
    with open(file_path, 'rb') as fo:
        label_dict = pickle.load(fo, encoding='bytes')
    return label_dict[b'fine_label_names'], label_dict[b'coarse_label_names']


def write_lable_map_csv(csv_path):
    cifar10file_path = '../cifar-10-batches-py/batches.meta'
    dict = load_cifar10_labels(cifar10file_path)
    dict[10] = 'fish'
    dict[11] = 'flowers'
    dict[12] = 'fruit_and_vegetables'
    dict[13] = 'people'
    dict[14] = 'trees'
    fieldnames = ['label', 'label_name']
    with open(csv_path, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for key, value in dict.items():
            writer.writerow({'Index':key, 'Label':value})
    print(f"Successfully wrote the dictionary to the CSV file: {csv_path}.")


write_lable_map_csv('labels.csv')
