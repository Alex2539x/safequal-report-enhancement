import json
import sys
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# MUST RUN: pip install scikit-learn

# "Research how to determine the optimal number of clusters"

def load_json(file_name):
    """
    Helper function that opens a file in read mode and returns the readable data.
    """
    # open input file in read mode
    with open(file_name, 'r') as file:
        data = json.load(file)
    return data

def find_optimal_clusters(data, max_k):
    """
    Returns the optimal number (k) of clusters and the associated silhouette
    scores. Requires data (data from json input file) and a max number of
    clusters k.
    """
    embeddings = [item["embedding"] for item in data]
    silhouette_scores = []

    # Minimum 2 clusters --> max clusters limit computation time
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k)
        clusters = kmeans.fit_predict(embeddings)
        score = silhouette_score(embeddings, clusters)
        silhouette_scores.append((k, score))
    
    optimal_k = max(silhouette_scores, key=lambda x: x[1])[0]
    return optimal_k, silhouette_scores

# main function
if __name__ == "__main__":
    # Incorrect usage case
    if len(sys.argv) != 3:
        print("Usage: python k-means-silhouette.py <input_json_file> <max_k>")
        sys.exit(1)
    
    # get command line arguments
    input_json_file = sys.argv[1]
    max_k = int(sys.argv[2])

    data = load_json(input_json_file)
    optimal_k, silhouette_scores = find_optimal_clusters(data, max_k)

    # Print final results
    print("Silhouette scores for each k:")
    for k, score in silhouette_scores:
        print(f"Clusters: {k}, Silhouette Score: {score}")    
    print(f"Optimal number of clusters: {optimal_k}")

# python k-means-silhouette.py <json file w/ embeddings> 10

# Silhouette scores for each k:
#     Clusters: 2, Silhouette Score: 0.12326430883921445
#     Clusters: 3, Silhouette Score: 0.16527376237270555
#     Clusters: 4, Silhouette Score: 0.13792613193469216
#     Clusters: 5, Silhouette Score: 0.20808863240739817
#     Clusters: 6, Silhouette Score: 0.20917014351988594
#     Clusters: 7, Silhouette Score: 0.23298439169123433
#     Clusters: 8, Silhouette Score: 0.28228337734434894
#     Clusters: 9, Silhouette Score: 0.3237589900029845
#     Clusters: 10, Silhouette Score: 0.31638609531021245
#     Clusters: 11, Silhouette Score: 0.32505185097291045
#     Clusters: 12, Silhouette Score: 0.3149062968566358
#     Clusters: 13, Silhouette Score: 0.3008378975430539
#     Clusters: 14, Silhouette Score: 0.29668774786616425
#     Clusters: 15, Silhouette Score: 0.2770569668365871
# Optimal number of clusters: 11

