from openai import OpenAI
import json
import os
import sys
from collections import defaultdict


client = OpenAI()

OpenAI.api_key = os.environ["OPENAI_API_KEY"]

def load_json(file_name: str) -> list:
    """
    Helper function that opens a file in read mode and returns the readable data.
    """
    with open(file_name, 'r') as file:
        data = json.load(file)
    return data

def generate_answer(question: str) -> str:
    """
    Generate a plausible answer for the given question using OpenAI's API. 
    Requires the question text. Returns generated answer text.
    """
    # Medical context prompt

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": f"You are a helpful assistant. \
                   Provide a plausible medical response to the following \
                   question by inventing new information: "},
                  {"role": "user", "content": f"{question}\n\nAnswer:"}],
        max_tokens=100,
        n=1,
        temperature=0.8
    )
    
    answer = response.choices[0].message.content.strip()
    return answer

# main function
def main():
    # incorrect usage case
    if len(sys.argv) != 2:
        print("Usage: python answer_questions.py <questions.json>")
        sys.exit(1)

    # get command line arguments
    questions_file = sys.argv[1]
    questions = load_json(questions_file)

    answers = defaultdict(list)
    for question in questions:
        answer = generate_answer(question)
        answers[str(question)] = answer

    answers_file = 'answers.json'
    with open(answers_file, 'w') as file:
        json.dump(answers, file, indent=4)

    print(f"Generated answers have been written to {answers_file}")

# run main function
if __name__ == "__main__":
    main()

# python answer_questions.py questions.json