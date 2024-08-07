import json
import sys
from collections import Counter

# Write a script, e.g., cluster-eval.py, which takes as its input the output of
# k-means.py and calculates the accuracy, i.e., the ratio of correctly clustered
# samples with respect to their labels.

def load_json(file_name):
    """
    Helper file that opens a file in read mode and returns the readable data
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
        
        # Append the item's label to cluster 
        cluster_labels[cluster].append(label)
    
    # Determine the most common label for each cluster
    correct_assignments = 0
    total_assignments = len(data)
    
    for cluster, labels in cluster_labels.items():
        most_common_label, _ = Counter(labels).most_common(1)[0]
        
        # Count the number of correct assignments
        correct_assignments += sum(1 for label in labels if label == most_common_label)
    
    # calculate & return accuracy
    accuracy = correct_assignments / total_assignments
    return accuracy

# Main function
if __name__ == "__main__":
    # Incorrect usage case
    if len(sys.argv) != 2:
        print("Usage: python cluster-eval.py <input_json_file>")
        sys.exit(1)
    
    # get command line argument
    input_json_file = sys.argv[1]

    data = load_json(input_json_file)
    
    # Calculate clustering accuracy
    accuracy = calculate_accuracy(data)

    print(f"Clustering Accuracy: {accuracy}")


# Ex run: python cluster-eval.py k-means2.json   # <-- from k-means.py
# Clustering Accuracy: 1.0

# python cluster-eval.py k-means.json
# Clustering Accuracy: 0.8