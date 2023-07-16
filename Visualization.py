import pandas as pd
import cv2
from matplotlib import pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

class DataVisualization:
    def tSNE(self, csvPath, Imagespath):
        df = pd.read_csv(csvPath)
        images = [Imagespath + '/' + image for image in df['filename']]
        labels = df['label']

        # Read and preprocess image data
        X = []
        for image in images:
            img = cv2.imread(image)
            img = img.flatten()  # Flatten the image into a 1D vector
            X.append(img)

        X = np.array(X)  # Convert the list to a NumPy array

        tsne = TSNE(n_components=2, random_state=42, perplexity=3)
        X_tsne = tsne.fit_transform(X)

        # Get unique class labels and assign a color to each class
        unique_labels = np.unique(labels)
        num_classes = len(unique_labels)
        colors = plt.cm.get_cmap('tab10', num_classes)

        # Plot the t-SNE representation with colored classes
        fig, ax = plt.subplots()
        for i, label in enumerate(unique_labels):
            indices = np.where(labels == label)
            ax.scatter(X_tsne[indices, 0], X_tsne[indices, 1], color=colors(i), label=label)

        plt.title("t-SNE Visualization with CIFAR-10 Classes")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        ax.legend()

        # Calculate accuracy based on ground truth labels
        predictions = np.argmax(X_tsne, axis=1)
        accuracy = np.mean(predictions == labels)
        print("Accuracy:", accuracy)

        plt.savefig('../plot.jpg')
        plt.show()
        def bar_plot(csv_file):
            df = pd.read_csv(csv_file)
            label_counts = df["label"].value_counts()

            label_counts.plot.bar()

            # Show the plot
            plt.show()

