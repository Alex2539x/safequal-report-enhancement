import json
import sys
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

# MUST RUN: pip install scikit-learn

# Make k-means.py script, which takes the output of your splice.py and generates
# output in the form [{“content”: “content 1”, “label”: “label 1”, “cluster”: 2,
# “distance”: 0.123}, … ].

def load_json(file_name):
    """
    Helper function that opens a file in read mode and returns the readable data.
    """
    # open input file in read mode
    with open(file_name, 'r') as file:
        data = json.load(file)
    return data

def k_means_clustering(data, n_clusters=3):
    """
    Adds clustering information to the given data (in the form of a json file).
    The number of clusters is determined by the user, but defaults to the value
    3 if none is provided. Uses class KMeans from sklearn module.
    """
    embeddings = [item["embedding"] for item in data]
    kmeans = KMeans(n_clusters=n_clusters)
    
    # fit_predict() must be called first
    clusters = kmeans.fit_predict(embeddings)
    # cluster_centers_ : ndarray of shape (n_clusters, n_features) Coordinates of cluster centers
    centroids = kmeans.cluster_centers_
    distances = cdist(embeddings, centroids, 'euclidean')   # 'cityblock', 'cosine'

    for i, item in enumerate(data):
        item["cluster"] = int(clusters[i])
        item["distance"] = float(distances[i][clusters[i]])
    
    return data

# main function
if __name__ == "__main__":
    # Incorrect usage case
    if len(sys.argv) != 3:
        print("Usage: python k-means.py <input_json_file> <number_of_clusters>")
        sys.exit(1)
    
    # get command line arguments
    input_json_file = sys.argv[1]
    n_clusters = int(sys.argv[2])

    data = load_json(input_json_file)
    clustered_data = k_means_clustering(data, n_clusters)

    # print updated data w/ cluster info
    with open("k-means.json", "w") as outfile:
        json.dump(clustered_data, outfile, indent=4)

    print("Clustered statements written to \"k-means.json\"")

# # of clusters calculated using k-means_silhouette.py

# python k-means.py embeddings.json 11
# output: k-means.json