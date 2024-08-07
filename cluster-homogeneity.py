import json
import sys
import numpy as np
from collections import defaultdict

def load_json(file_name: str) -> list:
    """
    Helper function that opens a file in read mode and returns the readable data.
    """
    with open(file_name, 'r') as file:
        data = json.load(file)
    return data

def calculate_distance_homogeneity(data: list):
    """
    Helper function that calculates homogeneity based on distance/confidence.
    Requires a list of dictionaries with text, label, embedding, cluster, and
    distance (k-means.json or subset.json). Returns a dictionary with mean and
    standard deviation of distances.
    """    
    distances = [item["distance"] for item in data]
    mean_distance = np.mean(distances)
    std_distance = np.std(distances)
    # can calculate percentile using z-score formula (w/ mean & standard deviation)
    return mean_distance, {"mean_distance": mean_distance, "std_distance": std_distance}

def calculate_cluster_size_homogeneity(cluster_data: dict) -> dict:
    """
    Helper function that Calculates homogeneity based on cluster sizes. Requires
    a dictionary with cluster IDs as keys and lists of cluster data as values.
    Returns a dictionary with mean and standard deviation of cluster sizes.
    """    
    sizes = [len(cluster) for cluster in cluster_data.values()]
    mean_size = np.mean(sizes)
    std_size = np.std(sizes)
    # can calculate percentile using z-score formula (w/ mean & standard deviation)
    return mean_size, {"mean_size": mean_size, "std_size": std_size}

def calculate_individual_cluster_homogeneity(cluster_data: dict, overall_mean_distance, overall_mean_size) -> dict:
    """
    Calculate homogeneity for each individual cluster. Requires a dictionary
    with cluster IDs as keys and lists of cluster data as values. Returns a
    dictionary with homogeneity scores for each cluster.
    """
    individual_homogeneity = {}
    for cluster_id, data in cluster_data.items():
        mean_distance, distance_homogeneity = calculate_distance_homogeneity(data)
        # can calculate percentile using z-score formula (w/ mean & standard deviation)
        individual_homogeneity[cluster_id] = {
            "distance_homogeneity": distance_homogeneity,
            "distance_score": 1-(abs(overall_mean_distance-mean_distance) / overall_mean_distance), # similarity w/ overall mean distance
            "size": len(data),
            "size_score": 1-((abs(overall_mean_size-len(data))) / overall_mean_size), # similarity w/ overall mean size
            "data": data
        }
    return individual_homogeneity

def calculate_homogeneity(data: list) -> dict:
    """
    Calculate various homogeneity scores for the clusters. Requires a list of
    dictionaries with text, label, embedding, cluster, and distance. Returns a
    dictionary with overall and individual cluster homogeneity scores.
    """    
    clusters = defaultdict(list)
    for item in data:
        clusters[item["cluster"]].append(item)
    
    mean_distance, distance_homogeneity = calculate_distance_homogeneity(data)
    mean_size, size_homogeneity = calculate_cluster_size_homogeneity(clusters)
    number_of_clusters = len(clusters)
    individual_cluster_homogeneity = calculate_individual_cluster_homogeneity(clusters, mean_distance, mean_size)

    # can calculate percentile using z-score formula (w/ mean & standard deviation)
    return {
        "overall_distance_homogeneity": distance_homogeneity,
        "overall_size_homogeneity": size_homogeneity,
        "number_of_clusters": number_of_clusters,
        "individual_cluster_homogeneity": individual_cluster_homogeneity
    }

# main function
if __name__ == "__main__":
    # incorrect usage case
    if len(sys.argv) != 2:
        print("Usage: python cluster-homogeneity.py <input_json_file>")
        sys.exit(1)
    
    # get command line arguments
    input_json_file = sys.argv[1]

    data = load_json(input_json_file)
    homogeneity_scores = calculate_homogeneity(data)

    # output to json
    with open('homogeneity_scores.json', 'w') as outfile:
        json.dump(homogeneity_scores, outfile, indent=4)
    
    print("Cluster homogeneity scores have been written to homogeneity_scores.json")


    # for key, value in homogeneity_scores.items():
    #     print(f"{key}: {value}")