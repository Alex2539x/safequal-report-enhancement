from openai import OpenAI
import json
import sys
import os

client = OpenAI()

# Write a Python script generate.py, which takes a topic description, its label,
# and a number as arguments and generates that many text statements in the
# output in the format of a JSON object array, e.g.: [{“text”: “statement 1”,
# “label”: “physics”}, …].

OpenAI.api_key = os.environ["OPENAI_API_KEY"]

# openai.BadRequestError: Error code: 400 - {'error': {'message': "'messages'
# must contain the word 'json' in some form, to use 'response_format' of type
# 'json_object'.", 'type': 'invalid_request_error', 'param': 'messages', 'code':
# None}}    <-- "JSON" description required in content request

# function to add to JSON
def write_json(new_data, filename='generate.json'):
    with open(filename,'r+') as file:
          # First we load existing data into a dict.
        file_data = json.load(file)
        # Join new_data with file_data inside emp_details
        file_data.extend(new_data)
        # Sets file's current position at offset.
        file.seek(0)
        # convert back to json.
        json.dump(file_data, file, indent = 4)

def generate_statements(description, label, number):
    """
    Returns a list of statements given a topic description, its label, and the
    number of statements that should be returned.
    """
    statements = []

    # output JSON might not be best decision for generating embeddings
    # name of output file in command line as possible future argument
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": f"You are a nurse at a hospital. \
             You witnessed a healthcare related incident involving a patient. \
             You are reporting the incident. Uniquely and specifically narrate \
             the incident in a single paragraph. Only report facts, avoid any \
             introductions. Exclude dates, times, and any room or bed numbers. \
             Make the report about a(n) {label} incident. You may not need a user prompt."},
            {"role": "user", "content": description}
        ],
        n=number,
        temperature=0.8
    )

    statements = [{"text": choices.message.content.strip(), "label": label} 
                  for choices in response.choices] 

    return statements

# main function
if __name__ == "__main__":
    # Incorrect usage case
    if len(sys.argv) != 4:
        # if specific topic_description not wanted, use placeholder: "No topic description provided"
        print("Usage: python generate.py <topic_description> <label> <num_statements>")
        sys.exit(1)

    # get command line arguments
    topic_description = sys.argv[1]
    label = sys.argv[2]
    num_statements = int(sys.argv[3])

    statements = generate_statements(topic_description, label, num_statements)

    # write statements to a file named generate.json
    # with open("generate.json", "w") as outfile:
    #     json.dump(statements, outfile, indent=4)

    # append statements to existing file (generate.json)
    write_json(statements, 'generate.json')

    print("Generated statements written to \"generate.json\"")

    # print(json.dumps(statements, indent=4))

# This time, each is generated w/ n parameter of 5; temperature was increased to 0.8
#  python generate.py "No topic description provided" "medication" 5
#  python generate.py "No topic description provided" "patient fall" 5
#  python generate.py "No topic description provided" "pressure injury" 5
#  python generate.py "No topic description provided" "behavioral" 5
#  python generate.py "No topic description provided" "healthcare-associated infection" 5
#  python generate.py "No topic description provided" "dietary" 5
#  python generate.py "No topic description provided" "misdiagnosis" 5
#  python generate.py "No topic description provided" "sepsis" 5
#  python generate.py "No topic description provided" "unsafe transfusion" 5
#  python generate.py "No topic description provided" "unsafe injection" 5
