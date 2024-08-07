from openai import OpenAI
import json
import sys
import os

client = OpenAI()

OpenAI.api_key = os.environ["OPENAI_API_KEY"]

# A proposed prototype will use a chain of semantic search and inference to
# enrich and augment the initial prompt for further analysis, e.g., get similar
# “good” incident reports, infer questions to make the given report better, get
# answer from users, use them to enrich/augment the report, repeat.
# Conceptually, this is imagined amounting to nudging/converging the given
# report embedding towards one or more of the good ones.

# It is also proposed that the OpenAI function calling capability is used to
# implement the chain.


def load_json(file_name: str) -> list:
    """
    Helper function that opens a file in read mode and returns the readable data.
    """    
    with open(file_name, 'r') as file:
        data = json.load(file)
    return data

def generate_questions(matches: list) -> list:
    """
    Generate questions based on the provided matches. Requires a list of
    dictionaries ("matches") representing the matched reports. Returns a list of
    questions generated from the matched reports.    
    """
    combined_texts = " ".join([match['text'] for match in matches])
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Based on \
             the following incident reports, generate questions to gather more \
             context. Separate the questions into separate lines and include \
             the question number in front of each question if applicable. Make \
             sure to include a question mark at the end of each question. Give your \
             responses in the form \"1. <question_1>\", \"2. <question_2> \" ... \
             \"n.\" <question_n>"},
            {"role": "user", "content": f"\n\n{combined_texts}\n\nQuestions:"}
            ],
        max_tokens=150
    )
    
    questions = response.choices[0].message.content.strip().split('\n')
    return questions

# main function
if __name__ == "__main__":
    # incorrect usage case
    if len(sys.argv) != 2:
        print("Usage: python infer-questions.py <matches.json>")
        sys.exit(1)
    
    # get command line arguments
    matches_file = sys.argv[1]

    matches = load_json(matches_file)
    questions = generate_questions(matches)

    # write statements to a file named questions.json
    with open("questions.json", "w") as outfile:
        json.dump(questions, outfile, indent=4)

    print("Generated statements written to \"questions.json\"")

    # print(json.dumps(questions, indent=4))


# python infer_questions.py search-sd3.json