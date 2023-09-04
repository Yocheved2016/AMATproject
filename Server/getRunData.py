from mlflow.tracking import MlflowClient
import pandas as pd
import numpy as np
import mlflow

CLASS = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck",
         "fish", "flowers", "fruits and vegetables", "people", "trees"]
class_name_to_number = {class_name: idx for idx, class_name in enumerate(CLASS)}
def fetch_run_data():
    EXPERIMENT_NAME = "model_monitoring_cifar"
    # mlflow.set_tracking_uri("file:///./Server")
    client = MlflowClient()

    # Retrieve Experiment information
    EXPERIMENT_ID = client.get_experiment_by_name(EXPERIMENT_NAME).experiment_id

    # Retrieve Runs information (parameter 'depth', metric 'accuracy')
    ALL_RUNS_INFO = client.search_runs([EXPERIMENT_ID])
    ALL_RUNS_ID = [run.info.run_id for run in ALL_RUNS_INFO]
    CLASS = [client.get_run(run_id).data.tags["class"] for run_id in ALL_RUNS_ID]
    ENTROPY_METRIC = [client.get_run(run_id).data.metrics["Entropy"] for run_id in ALL_RUNS_ID]
    CONFIDENCE_METRIC = [client.get_run(run_id).data.metrics["Confidence"] for run_id in ALL_RUNS_ID]
    HIST_DIST_METRIC = [client.get_run(run_id).data.metrics["Histogram_distance"] for run_id in ALL_RUNS_ID]

    # View Runs information
    run_data = pd.DataFrame({"Run ID": ALL_RUNS_ID,
                             "Class": CLASS,
                             "Entropy": ENTROPY_METRIC,
                             "Confidence": CONFIDENCE_METRIC,
                             "Histogram_distance": HIST_DIST_METRIC})

    return run_data

def create_dict():
    run_data = fetch_run_data()
    result_dict = {}
    avg_entropy_per_class = extract_avg_entropy()
    print(avg_entropy_per_class)
    # Loop through each unique class in your DataFrame
    for class_name in run_data['Class'].unique():
        class_data = run_data[run_data['Class'] == class_name]  # Filter data for the current class
        histogram_measurements = list(class_data['Histogram_distance'])
        initial_entropy = avg_entropy_per_class[class_name_to_number[class_name]]
        # initial_entropy = 0
        entropy_measurements = list(class_data['Entropy'])
        confidence_measurements = list(class_data['Confidence'])

        # Create a dictionary for the current class
        class_dict = {
            "initial_histogram": 0,
            'histogram_measurements': histogram_measurements,
            'initial_entropy': initial_entropy,
            'entropy_measurements': entropy_measurements,
            "initial_confidence": 0,
            'confidence_measurements': confidence_measurements
        }

        # Add the class dictionary to the result dictionary
        result_dict[class_name] = class_dict
    return result_dict

def extract_avg_entropy():
    loaded_data = np.load('../model/entropy_all_classes.npz')
    return loaded_data['entropys']

print(create_dict())