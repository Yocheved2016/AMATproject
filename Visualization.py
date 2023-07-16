import pandas as pd
import matplotlib.pyplot as plt

def classes_bar(csv_file):
    df = pd.read_csv(csv_file)
    label_counts = df["label"].value_counts()

    label_counts.plot.bar()

    # Show the plot
    plt.show()

