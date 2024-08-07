import json
import sys
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import silhouette_score

# MUST RUN: pip install matplotlib

# Extend cluster-eval.py to produce (and maybe plot) the correlation between the
# accuracy and the confidence, i.e., the maximum distance to the cluster
# centroid.

def load_json(file_name):
    """
    Helper file that opens a file in read mode and returns the readable data.
    """    
    with open(file_name, 'r') as file:
        data = json.load(file)
    return data

def calculate_accuracy(data):
    """
    Calculate clustering membership accuracy with respect to the known label.
    Returns the accuracy as a float value. Requires data with the key names
    "content", "label", "cluster", and "distance".
    """
    # create dictionary to store most common label for each cluster
    cluster_labels = {}

    for item in data:
        cluster = item["cluster"]
        label = item["label"]
        
        if cluster not in cluster_labels:
            cluster_labels[cluster] = []
        
        # Append item's label to cluster 
        cluster_labels[cluster].append(label)
    
    # Determine most common label for each cluster
    correct_assignments = 0
    total_assignments = len(data)
    
    for cluster, labels in cluster_labels.items():
        most_common_label, _ = Counter(labels).most_common(1)[0]
        
        # Count number of correct assignments
        correct_assignments += sum(1 for label in labels if label == most_common_label)
    
    # calculate & return accuracy
    accuracy = correct_assignments / total_assignments
    return accuracy

def evaluate_clusters(data, max_distances):
    """
    Returns the clustering accuracy which, depending on the distance calculated
    to the nearest cluster, returns the associated accuracies. 
    """
    accuracies = []

    for max_distance in max_distances:
        filtered_data = [item for item in data if item["distance"] <= max_distance]
        
        if len(filtered_data) > 0:
            accuracy = calculate_accuracy(filtered_data)
            accuracies.append(accuracy)
        else:
            accuracies.append(0)
    
    return accuracies

def plot_correlation(max_distances, accuracies):
    """
    Plot the correlation between clustering accuracy and confidence. Uses
    matplotlib. 
    """
    plt.style.use('dark_background')
    plt.figure(figsize=(10, 6))
    plt.plot(max_distances, accuracies, marker='o')
    plt.xlabel('Maximum Distance to Centroid')
    plt.ylabel('Clustering Accuracy')
    plt.title('Correlation between Clustering Accuracy and Confidence')
    plt.grid(True)
    plt.show()

# main function
if __name__ == "__main__":
    # Incorrect usage case
    if len(sys.argv) != 2:
        print("Usage: python cluster-eval2.py <input_json_file>")
        sys.exit(1)
    
    input_json_file = sys.argv[1]

    data = load_json(input_json_file)

    max_distances = np.linspace(0, max(item["distance"] for item in data), num=50)
    accuracies = evaluate_clusters(data, max_distances)
    
    plot_correlation(max_distances, accuracies)

    correlation = np.corrcoef(max_distances, accuracies)[0, 1]
    print(f"Correlation between maximum distance and accuracy: {correlation:.4f}")


# python cluster-eval2.py k-means.json
#   Correlation between maximum distance and accuracy: -0.3054