import json
import sys
import os
from openai import OpenAI
from scipy.spatial.distance import cosine, euclidean

client = OpenAI()

# Given a query, return best matches from a dataset with embeddings, e.g.,
# search.py “<query>” <set.json>. The query can be a question, in which case the
# matches should contain an answer, or a statement, in which case the matches
# contain similar or relevant information.

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

def find_best_matches(query: str, dataset: list, top_n: int = 5) -> list:
    """
    Find the best matches for the query in the dataset. The default response
    number is 5. Requires the query text, list of dictionaries containing the
    searchable data, and the number of top matches to return. Returns the list
    of best matching entries from the dataset.
    """
    query_embedding = get_embedding(query)
    
    # similarities = []     # higher values -> more related
    distances = []          # lower values -> more related
    for item in dataset:
        item_embedding = item['embedding']
        # similarity = calculate_cosine_similarity(query_embedding, item_embedding)
        # similarities.append((item, similarity))
        distance = calculate_euclidean_distance(query_embedding, item_embedding)
        distances.append((item, distance))
    
    # Sort similarity score in descending order
    # similarities.sort(key=lambda x: x[1], reverse=True)
    
    # Sort distance in ascending order
    distances.sort(key=lambda x: x[1])

    # return [item[0] for item in similarities[:top_n]]
    return [item[0] for item in distances[:top_n]]

# main function
if __name__ == "__main__":
    # incorrect usage case
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print("Usage 1: python search.py <query> <set.json>\
              \nUsage 2: python search.py <query> <set.json> <response_quantity>")
        sys.exit(1)
    
    # get command line arguments
    query = sys.argv[1]
    dataset_file = sys.argv[2]

    dataset = load_json(dataset_file)

    # determine whether to use response_quantity as argument
    if len(sys.argv) == 4:
        quantity = int(sys.argv[3])
        best_matches = find_best_matches(query, dataset, quantity)
    else:
        best_matches = find_best_matches(query, dataset)

    with open("search2.json", "w") as outfile:
        json.dump(best_matches, outfile, indent=4)

    print("Clustered statements written to \"search2.json\"")

#  python search.py "medication issues" k-means.json 5
#  python search.py "medication issues" k-means.json 3