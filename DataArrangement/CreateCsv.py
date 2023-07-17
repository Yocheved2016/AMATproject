import csv


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

    file_exists = False
    try:
        with open(output_file, mode='r'):
            file_exists = True
    except FileNotFoundError:
        pass

    with open(output_file, mode='a', newline='') as file:
        fieldnames = ['filename', 'label', 'origin']
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerows(data)