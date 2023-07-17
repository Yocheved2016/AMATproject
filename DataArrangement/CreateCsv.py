import csv
import pandas as pd
import os

def create_or_add_csv(filenames, labels, origin, output_file):
    if len(filenames) != len(labels):
        raise ValueError("The input arrays must have the same length.")

    data = []
    for filename, label in zip(filenames, labels):
        data.append({
            'filename': filename.decode("ascii"),
            'label': label,
            'origin': origin
        })

    desired_labels_cifar10 = list(range(10))  # 0-9
    desired_labels_cifar100 = [1, 4, 2, 14, 17]
    data_df = pd.DataFrame(data)
    filtered_df = data_df[(data_df['origin'] == 'cifar10') & data_df['label'].isin(desired_labels_cifar10) |
                          (data_df['origin'] == 'cifar100') & data_df['label'].isin(desired_labels_cifar100)]

    file_exists = os.path.isfile(output_file)

    with open(output_file, mode='a', newline='') as file:
        fieldnames = ['filename', 'label', 'origin']
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerows(filtered_df.to_dict('records'))


def train_test_validation_split(csv_path, test_size, validation_size):
    df = pd.read_csv(csv_path)
    # shuffle the df
    df = df.sample(frac=1, random_state=42)

    # split the df
    test_index = int(len(df) * test_size)
    validation_index = int(len(df) * (test_size + validation_size))

    test_df = df.iloc[:test_index]
    validation_df = df.iloc[test_index:validation_index]
    train_df = df.iloc[validation_index:]

    # write each part to csv
    test_df.to_csv("test_data.csv", index=False)
    validation_df.to_csv("validation_data.csv", index=False)
    train_df.to_csv("train_data.csv", index=False)
