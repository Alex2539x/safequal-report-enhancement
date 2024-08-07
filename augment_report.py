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

def augment_report(report: str, answers: dict) -> str:
    """
    Augment the given report with additional information from answers. Requires
    the original report text ("report") and a dictionary containing questions
    and their respective answers ("answers"). Returns the augmented report text.
    """
    prompt = f"Original Report:\n{report}\n\nAdditional Information:\n"
    for question, answer in answers.items():
        prompt += f"Q: {question}\nA: {answer}\n"
    # prompt += "\nAugmented Report:"

    # experiment w/: including essential info 
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "You are a helpful assistant. \
                   Given the original report labeled Original Report, \
                   augment, extend, and concisely display information from the \
                   original report with the additional information labeled \
                   Additional Information by an additional 500 or less characters. \
                   Write in paragraph format and in complete sentences. \
                   Do not write in bulletpoint format. Include only essential \
                   information. "},
                  {"role": "user", "content": f"{prompt}"}],
        max_tokens=1000,
    )
    
    augmented_report = response.choices[0].message.content.strip()
    return augmented_report

# main function
if __name__ == "__main__":
    # incorrect usage case; report is the original report used in search-sd.py
    if len(sys.argv) != 3:
        print("Usage: python augment-report.py \"<report>\" <answers.json>")
        sys.exit(1)
    
    # get command line arguments
    report = sys.argv[1]
    answers_file = sys.argv[2]

    answers = load_json(answers_file)
    augmented_report = augment_report(report, answers)

    # write statements to a file named questions.json
    with open("augmented-report.json", "w") as file:
        json.dump(augmented_report, file, indent=4)

    print("Generated statements written to \"augmented-report.json\"")

    # print(augmented_report)

# python search_sd.py "A patient was spotted on the ground in pain." k-means.json
#  -->
# python augment_report.py "A patient was spotted on the ground in pain." answers.json