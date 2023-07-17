import pandas as pd

def filter_csv():
    # Read the CSV file
    df = pd.read_csv('../cifar_data.csv')

    # Filter based on CIFAR-10 labels (0-9) and CIFAR-100 labels [1, 4, 2, 14, 17]
    desired_labels_cifar10 = list(range(10))  # 0-9
    desired_labels_cifar100 = [1, 4, 2, 14, 17]
    filtered_df = df[(df['origin'] == 'cifar10') & df['label'].isin(desired_labels_cifar10) |
    (df['origin'] == 'cifar100') & df['label'].isin(desired_labels_cifar100)]

    # Save the filtered DataFrame to a new CSV file
    filtered_df.to_csv('../cifar_data_update.csv', index=False)