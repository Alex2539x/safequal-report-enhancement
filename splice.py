import json
import sys

# Write a simple Python script that splices multiple files with JSON arrays,
# e.g., splice.py

def splice_to_output_file(input_files, output_file):
    """
    Retrieves information from the input_files and write new information to 
    output_file
    """
    combined_data = []

    for file in input_files:
        # open input file in read mode
        with open(file, "r") as file:
            data = json.load(file)
            combined_data.extend(data)
    
    # open output file in write mode
    with open(output_file, "w") as output:
        json.dump(combined_data, output, indent=4)

# main function
if __name__ == "__main__":
    # incorrect usage case; must be at least two input files
    if len(sys.argv) < 3:
        print("Usage: python splice.py <output_file> <input_file1> ... <input_fileN>")
        sys.exit(1)
    
    # get command line arguments
    output_file = sys.argv[1]
    input_files = sys.argv[2:]

    splice_to_output_file(input_files, output_file)
