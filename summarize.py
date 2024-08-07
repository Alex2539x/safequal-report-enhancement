import json
import sys
import os
from openai import OpenAI
from collections import defaultdict

client = OpenAI()

OpenAI.api_key = os.environ["OPENAI_API_KEY"]

# Summarize.py, given the compacted clustered input above, generates for each
# cluster: [{“label”: “label1”, “cluster”: 1, “name”: “short descriptive cluster
# name”, “summary”: “cluster description”}, …]. The interesting qualitative
# correlation is between name/summary, and the original topic when generated

def load_json(file_name: str) -> dict:
    """
    Helper file that opens a file in read mode and returns the readable data.
    """
    with open(file_name, 'r') as file:
        data = json.load(file)
    return data

def generate_name(combined_text):
    """
    Generate a short descriptive name for the given combined text. Requires the
    combined text of the cluster. Returns a short descriptive name for the
    cluster.
    """
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "You are a helpful assistant. Provide healthcare incident descriptions for the following texts that belong to a single category, please give this category a short and descriptive name."},
                 {"role": "user", "content": f"\n\n{combined_text}\n\nName:"}],
        max_tokens=50
    )
    
    name = response.choices[0].message.content.strip()
    return name

def generate_summary(combined_text):
    """
    Generate a summary for the given combined text. Requires the total combined
    text of the cluster. Returns a summary for the cluster.
    """
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "You are a helpful assistant. Provide a cohesive healthcare incident summary for the following texts."}, 
                  {"role": "user", "content": f"\n\n{combined_text}\n\nSummary:"}],
        max_tokens=200
    )
    
    summary = response.choices[0].message.content.strip()
    return summary


def summarize_cluster(cluster_data: list) -> dict:
    """
    Generate a summary for the given cluster data. Requires a list of
    dictionaries with text, label, embedding, cluster, and distance. Returns a
    dictionary with label, cluster number, name, and summary.
    """
    texts = [item["text"] for item in cluster_data]
    combined_text = " ".join(texts)

    name = generate_name(combined_text)
    summary = generate_summary(combined_text)

    return {
        "label": cluster_data[0]["label"],
        "cluster": cluster_data[0]["cluster"],
        "name": name,
        "summary": summary
    }


def generate_summaries(data: dict) -> list:
    """
    Generate summaries for each cluster in the data. The input data is in the form
    created by subset.py, which is a dictionary with cluster numbers as keys and 
    lists of cluster data as values. Returns a list of dictionaries with cluster 
    summaries.
    """
    clusters = defaultdict(list)
    for item in data:
        clusters[item["cluster"]].append(item)
    
    summaries = []
    for cluster_data in clusters.values():
        summary = summarize_cluster(cluster_data)
        summaries.append(summary)
    return summaries


# main function
if __name__ == "__main__":
    # incorrect usage case
    if len(sys.argv) != 2:
        print("Usage: python summarize.py <input_json_file>")
        sys.exit(1)
    
    # get command line arguments
    input_json_file = sys.argv[1]

    data = load_json(input_json_file)
    summaries = generate_summaries(data)

    with open('summarize.json', 'w') as outfile:
        json.dump(summaries, outfile, indent=4)
    
    print("Cluster summaries have been written to summarize.json")


# 