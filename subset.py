import json
import sys

def load_json(file_name):
    """
    Helper file that opens a file in read mode and returns the readable data.
    """
    with open(file_name, 'r') as file:
        data = json.load(file)
    return data

def create_subsets(data, threshold):
    """
    Returns subsets of the clusters
    """
    subsets = []

    for item in data:
        cluster = item["cluster"]
        distance = item["distance"]
        
        if distance <= threshold:
            # if cluster not in subsets:
            #     subsets[cluster] = []
            # subsets[cluster].append(item)
            subsets.append(item)
    
    return subsets

# main function
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python subset.py <input_json_file> <distance_threshold>")
        sys.exit(1)
    
    # get command line arguments
    input_json_file = sys.argv[1]
    distance_threshold = float(sys.argv[2])

    data = load_json(input_json_file)
    subsets = create_subsets(data, distance_threshold)
    
    # write statements to a file named generate.json
    with open("subset.json", "w") as outfile:
        json.dump(subsets, outfile, indent=4)

    print("Generated statements written to \"subset.json\"")


# python subset.py k-means.json 0.2

# input: k-means.json (threshhold: .2)
# output: subset.json

# input: k-means.json (theshhold: .4)
# output: subset2.json