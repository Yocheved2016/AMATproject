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

    def bar_plot(self, csv_file):
        df = pd.read_csv(csv_file)
        label_counts = df["label"].value_counts()

        label_counts.plot.bar()

        # Show the plot
        plt.show()

    def train_image_samples(self, csv_file, label_mapping_csv):
        df = pd.read_csv(csv_file)
        label_mapping_df = pd.read_csv(label_mapping_csv)

        df = pd.merge(df, label_mapping_df, on='label')

        df = df.sort_values(by=['label'])

        classes = df['label'].unique()
        for class_label in classes:
            class_images = df[df['label'] == class_label].head(10)
            fig, axes = plt.subplots(2, 5, figsize=(12, 6))
            class_name = class_images.iloc[0]['label_name']  # Retrieve the label name
            fig.suptitle('Class {}'.format(class_name))

            for i, (_, row) in enumerate(class_images.iterrows()):
                img_path = row['filename']
                img = cv2.imread(f'./../images/{img_path}')

                ax = axes[i // 5, i % 5]
                ax.imshow(img)
                ax.axis('off')


            plt.tight_layout()
            plt.show()

            plt.close()
