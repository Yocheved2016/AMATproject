# Cifar Classification Project - AMAT BootCamp

## Overview

This project is a comprehensive exploration of image classification using CIFAR-10 and CIFAR-100 datasets. 
It encompasses the entire project lifecycle, from data preparation to constructing a custom CNN model and deploying it with monitoring capabilities. 
The project was undertaken within the framework of a boot camp hosted by Applied Materials.

### Dataset Preparation
- Downloaded the CIFAR-10 dataset.
- Integrated labels from CIFAR-100 and a custom captured dataset.
- Applied data augmentation techniques to balance the data.
- Resampled custom images to align with CIFAR data.
- Ensured data quality and suitability for neural networks.

### Data Analysis
- Conducted an in-depth analysis of the merged dataset.
- Split the data into training, validation, and test sets for model development and evaluation.

### Modeling with TensorFlow
- Developed a custom neural network from the ground up.
- Analyzed model training results, including convergence graphs, accuracy, and confusion matrices.


### Out Of Distribution Data Handling
- Utilized isolation forest to detect data outside trained classes for accurate classification.
- Demonstrated a deep understanding of the topic, experimenting with various methods to achieve the desired results.


### Inference and Evaluation
- Evaluated model inference on test data.
- Visualized correctly and incorrectly classified examples with probabilities.
- Presented a confusion matrix to evaluate model performance.

### GUI
- Implemented a user-friendly Dash interface.
- Enabled users to upload images from their devices or use their device cameras.
- Allow user provide feedback about prediction (for monitoring model perfomance).

### Serving & monitoring
- Exposed the image classification model through a server.
- Monitored incoming data to determine the need for retraining, using mlflow.
- Utilized Dashboards to analyze model performance.
- Stored incoming data with correct labels for potential retraining in case of concept drift or data drift.

### Development Environment
- project developed using PyCharm.
- Training model on Kaggle for powerfull GPU performance
- mlflow server



## Skills Gained

Throughout this project and bootcamp experience, I have gained several critical skills that are highly valuable in a professional setting:

- Deep Learning: Gained hands-on experience in designing and training neural networks for image classification.
- Machine Learning: Applied machine learning techniques to solve real-world problems.
- Python: Enhanced proficiency in Python, particularly for data manipulation and model development.
- Data Analysis: Conducted comprehensive data analysis to make informed decisions.
- TensorFlow: Developed models using TensorFlow, a popular deep learning framework.
- Project Management: Successfully managed project requirements, tasks, and timelines.
- Code Quality: Maintained clean, modular, and efficient code following best practices.
- Documentation: Documented code and assumptions to facilitate team collaboration.
- Problem Solving: Developed strong problem-solving skills through code debugging and online research.

