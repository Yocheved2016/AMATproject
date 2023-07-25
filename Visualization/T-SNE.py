import pandas as pd
import cv2
from matplotlib import pyplot as plt
import numpy as np
from sklearn.manifold import TSNE


class DataVisualization:
    
    def extract_features_from_the_cnn(images):
        pass



    def tSNE(data):
            tsne = TSNE(n_components=2, random_state=42, perplexity=3)
            data_2d = tsne.fit_transform(data)
            return  data_2d


    def visualizeData(data_2d,labels):

        # Get unique class labels and assign a color to each class
            unique_labels = np.unique(labels)
            num_classes = len(unique_labels)
            colors = plt.cm.get_cmap('tab10', num_classes)
            # Plot the t-SNE representation with colored classes
            fig, ax = plt.subplots()
            for i, label in enumerate(unique_labels):
                indices = np.where(labels == label)
                ax.scatter(data_2d[indices, 0], data_2d[indices, 1], color=colors(i), label=label)

            plt.title("t-SNE Visualization with 15 Classes")
            plt.xlabel("Dimension 1")
            plt.ylabel("Dimension 2")
            ax.legend()
            plt.savefig('../plot.jpg')
            plt.show() 
