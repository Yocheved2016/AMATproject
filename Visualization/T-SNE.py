import pandas as pd
import cv2
from matplotlib import pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from keras.models import load_model,Model

class DataVisualization:
    
    
    def __init__(self,data_path,model_path):
 
        self.data_path=data_path
        self.model_path=model_path
        self.x_test=np.array([])
        self.labels=np.array([])
        self.test_representations_2d=[]
        self.features=[]
        self.load_dataset()
        self.extract_features_from_the_model()
      
    
    def load_dataset(self):
       
        # load dataset
        data = np.load(self.data_path)
        x_test = data['test']
        y_test = data['ytest']
        
        #normalize the data
        mean_per_channel_test = np.mean(x_test, axis=(0, 1))
        std_per_channel_test = np.std(x_test, axis=(0, 1))
        x_test = (x_test - mean_per_channel_test) / std_per_channel_test
        self.x_test=x_test
        self.labels=y_test

    
    def extract_features_from_the_model(self):
        
        model = load_model(self.model_path)
        intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer('dense_5').output)        
        for img in self.x_test:
            img = img.reshape(1,32, 32, 3)
            self.features.append(intermediate_layer_model.predict(img))


    def tsne(self):
        # Apply t-SNE
        
        tsne = TSNE(n_components=2, random_state=0)
        self.test_representations_2d=tsne.fit_transform(np.array(self.features).reshape(1000,15))
         
    def create_plot(self,title=None):
    
        # Create a plot
        plt.figure(figsize=(10,10))
        scatter = plt.scatter(self.test_representations_2d[:, 0], self.test_representations_2d[:, 1], c=self.labels.flatten(), cmap='tab10')

        # Create a legend with class names
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        legend1 = plt.legend(*scatter.legend_elements(num=10), title="Classes")
        plt.gca().add_artist(legend1)

        # Convert labels to class names
        class_indices = [int(label.get_text().split("{")[-1].split("}")[0]) for label in legend1.texts]
        for t, class_index in zip(legend1.texts, class_indices):
            t.set_text(class_names[class_index])

        plt.savefig('../plot.jpg')
        plt.show()    
        



# dataVisualization=DataVisualization('./cfar10_modified_1000.npz','./best_model.h5')
# dataVisualization.tsne()
# dataVisualization.create_plot()
