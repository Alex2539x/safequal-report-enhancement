import json
import sys
import os
import numpy as np
from openai import OpenAI
from scipy.spatial.distance import cosine, euclidean

client = OpenAI()

# Given a query, return best matches from a dataset with embeddings, e.g.,
# search.py “<query>” <set.json>. The query can be a question, in which case the
# matches should contain an answer, or a statement, in which case the matches
# contain similar or relevant information.

# This script uses mean and standard deviation of distance data to determine
# what is returned.

OpenAI.api_key = os.environ["OPENAI_API_KEY"]

def load_json(file_name: str) -> list:
    """
    Helper function that opens a file in read mode and returns the readable data.
    """
    with open(file_name, 'r') as file:
        data = json.load(file)
    return data

def get_embedding(text: str) -> list:
    """
    Get the embedding for the given text using OpenAI's API. Requires text (str)
    for embedding. Returns a list of the text's embeddings.
    """
    response = client.embeddings.create(
        input=[text],
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

# def calculate_cosine_similarity(embedding1: list, embedding2: list) -> float:
#     """
#     Calculate the Euclidean distance between two embeddings. Requires two
#     embeddings. Returns the Euclidean distance between the
#     two embeddings.
#     """
#     return 1 - cosine(embedding1, embedding2)

def calculate_euclidean_distance(embedding1: list, embedding2: list) -> float:
    """
    Calculate the cosine similarity between two embeddings. Requires two
    embeddings. Returns the cosine similarity score between the two embeddings.
    """
    return euclidean(embedding1, embedding2)


def find_best_matches_sd(query: str, dataset: list) -> list:
    """
    Find the best matches for the query in the dataset using standard deviation
    to determine the number of returned queries. Requires the query text and the
    list of dictionaries containing the dataset. Returns the list of best
    matching entries from the dataset.
    """
    query_embedding = get_embedding(query)
    
    distances = []
    for item in dataset:
        item_embedding = item['embedding']
        distance = calculate_euclidean_distance(query_embedding, item_embedding)
        distances.append((item, distance))
    
    mean_distance = np.mean([dist for _, dist in distances])
    std_distance = np.std([dist for _, dist in distances])
    threshold_distance = mean_distance - (1.5*std_distance)
    
    best_matches = [item for item, dist in distances if dist <= threshold_distance]

    # Sort by distance in ascending order
    best_matches.sort(key=lambda x: x['distance'])  
    return best_matches


# main function
if __name__ == "__main__":
    # incorrect usage case
    if len(sys.argv) != 3:
        print("python search.py <query> <set.json>")
        sys.exit(1)
    
    # get command line arguments
    query = sys.argv[1]
    dataset_file = sys.argv[2]

    dataset = load_json(dataset_file)

    best_matches = find_best_matches_sd(query, dataset)

    with open("search-sd.json", "w") as outfile:
        json.dump(best_matches, outfile, indent=4)

    print("Clustered statements written to \"search-sd.json\"")

#  python search_sd.py "medication issues" k-means.json
#  python search_sd.py "What are some examples of dietary problems?" k-means.json
#  python search_sd.py "A patient was spotted on the ground in pain." k-means.json