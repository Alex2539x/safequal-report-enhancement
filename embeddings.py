from openai import OpenAI
# import array as arr
import json
import sys
import os
import numpy as np

client = OpenAI()

# Write another script, embeddings.py, which takes the output of the previous
# script as its input, generates an embedding for each string, and writes an
# output in for the format of a JSON object array, e.g.: [{“text”: “statement
# 1”, “label”: “physics”, “embedding”: […]}, …]

OpenAI.api_key = os.environ["OPENAI_API_KEY"]

def normalize_l2(x):
    """
    Helper function that normalizes an array of vectors. From embeddings
    tutorial on platform.openai.com.
    """
    x = np.array(x)
    if x.ndim == 1:
        norm = np.linalg.norm(x)
        if norm == 0:
            return x
        return x / norm
    else:
        norm = np.linalg.norm(x, 2, axis=1, keepdims=True)
        return np.where(norm == 0, x, x / norm)

# Both of our new embedding models were trained with a technique that allows
# developers to trade-off performance and cost of using embeddings.
# Specifically, developers can shorten embeddings (i.e. remove some numbers from
# the end of the sequence) without the embedding losing its concept-representing
# properties by passing in the dimensions API parameter. For example, on the
# MTEB benchmark, a text-embedding-3-large embedding can be shortened to a size
# of 256 while still outperforming an unshortened text-embedding-ada-002
# embedding with a size of 1536. You can read more about how changing the
# dimensions impacts performance in our embeddings v3 launch blog post.

# In general, using the dimensions parameter when creating the embedding is the
# suggested approach. In certain cases, you may need to change the embedding
# dimension after you generate it. When you change the dimension manually, you
# need to be sure to normalize the dimensions of the embedding as is shown
# below.

def generate_embeddings(statements):
    """
    Returns an updated list of statements with their associated embeddings.
    """
    for s in statements:
        response = client.embeddings.create(
            input = s["text"],
            model="text-embedding-3-small"
        )
        embedding = response.data[0].embedding
        # If reducing dimensionality is desired.
        cut_dim = response.data[0].embedding[:256]
        norm_dim = normalize_l2(cut_dim) 
        s["embedding"] = embedding #norm_dim    

    return statements

# main function
if __name__ == "__main__":
    # Incorrect usage case
    if len(sys.argv) != 2:
        # file w/ json info generated from generate.py
        print("Usage (file in json format): python embeddings.py <input_file>")
        sys.exit(1)

    # get command line arguments
    input_file = sys.argv[1]

    # open the json input file in read mode
    with open(input_file, "r") as file:
        statements = json.load(file)

    statements_embeddings = generate_embeddings(statements)
    
    # write statements to a file named generate.json
    with open("embeddings.json", "w") as outfile:
        json.dump(statements_embeddings, outfile, indent=4)

    print("Generated statements written to \"embeddings.json\"")

# input file: generate.json
# output file: embeddings.json

# python embeddings.py generate.json
