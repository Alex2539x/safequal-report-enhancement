from openai import OpenAI
import json
import os
import numpy as np
from collections import defaultdict
from typing_extensions import override
from openai import AssistantEventHandler

from search_sd import load_json, get_embedding, calculate_euclidean_distance

client = OpenAI()

OpenAI.api_key = os.environ["OPENAI_API_KEY"]

# IMPORTANT: This implementation currently simulates user answers (by answering
# the questions created from the most recent search results).

# Functions are defined for agents
functions = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search for similar good incident reports and returns matches.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The report query used to search for similar reports."
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_questions",
            "description": "Infer questions to gather more context based on the matches from search results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "matches": {
                        "type": "array",
                        "items": {
                            "type": "object"
                        },                        
                        "description": "The search results to base the questions on, passed from the output of the search function."
                    }
                },
                "required": ["matches"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_answers",
            "description": "Invent responses to answer context collecting questions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "questions": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },                        
                        "description": "The list questions to respond to, passed from the output of the generate_questions function."
                    }
                },
                "required": ["questions"]
            }
        }
    },    
    {
        "type": "function",
        "function": {
            "name": "augment_report",
            "description": "Augment the user provided report with user provided answers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The original user provided report."
                    }
                },
                "required": ["query"]
            }
        }
    }
]

# Initialize the assistant with the provided functions
assistant = client.beta.assistants.create(
    instructions="You are an assistant that helps improve incident reports. \
        Use the provided functions to search for similar reports, \
            generate questions, and augment reports.",
    model="gpt-4o-mini",
    tools=functions
)

# Define the event handler class
class EventHandler(AssistantEventHandler):
    @override
    def on_event(self, event):
        # Retrieve events that are denoted with 'requires_action'
        if event.event == 'thread.run.requires_action':
            run_id = event.data.id  # Retrieve the run ID from the event data
            self.handle_requires_action(event.data, run_id)

    def handle_requires_action(self, data, run_id):
        tool_outputs = []

        for tool in data.required_action.submit_tool_outputs.tool_calls:
            if tool.function.name == "search":
                query = json.loads(tool.function.arguments)["query"]
                search_results = search(query)
                tool_outputs.append({"tool_call_id": tool.id, "output": json.dumps(search_results)})

            elif tool.function.name == "generate_questions":
                # matches = json.loads(tool.function.arguments)["matches"]
                matches = load_json("search-sd-fc.json")
                questions = generate_questions(matches)
                tool_outputs.append({"tool_call_id": tool.id, "output": json.dumps(questions)})

            elif tool.function.name == "generate_answers":
                questions = json.loads(tool.function.arguments)["questions"]
                answers = generate_answers(questions)
                tool_outputs.append({"tool_call_id": tool.id, "output": json.dumps(answers)})

            elif tool.function.name == "augment_report":
                query = json.loads(tool.function.arguments)["query"]
                # answers = json.loads(tool.function.arguments)["answers"]
                augmented_report = augment_report(query)
                tool_outputs.append({"tool_call_id": tool.id, "output": json.dumps(augmented_report)})

        # Submit all tool_outputs at the same time
        self.submit_tool_outputs(tool_outputs, run_id)

    def submit_tool_outputs(self, tool_outputs, run_id):
        # Use the submit_tool_outputs_stream helper
        with client.beta.threads.runs.submit_tool_outputs_stream(
            thread_id=self.current_run.thread_id,
            run_id=self.current_run.id,
            tool_outputs=tool_outputs,
            event_handler=EventHandler(),
        ) as stream:
            for text in stream.text_deltas:
                print(text, end="", flush=True)
            print()

def main():
    """
    Main function. Handles the initialization and execution of the script. 
    """
    # Run augmented report process in constant loop
    while True:
        # Prompt user for query
        query = input("Enter your query (or type 'quit' to exit): ")
        if query.lower() == 'quit':
            exit    

        initial_query_embedding = get_embedding(query)
        prev_query_embedding = initial_query_embedding
        sse_list = []

        # Max num of iterations set to 10 (arbitrary)
        for _ in range(10):  
            # Initial user message
            user_message = f"Here is an initial incident report that needs improvement:\n\n{query}"

            # Start new thread
            thread = client.beta.threads.create()   

            # Format initial user message
            message = client.beta.threads.messages.create(
                thread_id=thread.id,
                role="user",
                content=user_message
            )

            # Stream events and handle them with the event handler
            with client.beta.threads.runs.stream(
                thread_id=thread.id,
                assistant_id=assistant.id,
                event_handler=EventHandler()
            ) as stream:
                stream.until_done()

            # ===================================================
            # vvv Assess drift (from previous search results) vvv
            # ===================================================
            with open("search-sd-fc.json", 'r') as file:
                matches = json.load(file)   

            match_embeddings = [match['embedding'] for match in matches]
            centroid = find_centroid(match_embeddings)
            sse = calculate_sse(match_embeddings, centroid)
            sse_list.append(sse)

            if len(sse_list) > 1:
                drift = np.sum((np.array(centroid) - np.array(prev_query_embedding)) ** 2)
                # Convergence criteria: 100 times less than the original query's distance
                original_distance = calculate_euclidean_distance(initial_query_embedding, centroid)
                if drift < original_distance / 1000 and abs(sse_list[-1] - sse_list[-2]) < 0.01:
                    print("Convergence criteria met.")
                    break

            prev_query_embedding = centroid

            augmented_report_file = "augmented-report.json"
            augmented_report = load_json(augmented_report_file)
            query = augmented_report

def search(query):
    """
    Find the best matches for the query in the dataset using standard deviation
    to determine the number of returned queries. Requires the query text and the
    list of dictionaries containing the dataset. Returns the list of best
    matching entries from the dataset.
    """    
    print("\nRunning search function")
    dataset_file = "k-means.json"
    dataset = load_json(dataset_file)

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

    # for-loop (remove embeddings; uncomment to shorten search-sd-fc result file)
    # key_embeddings = "embedding"
    # for item in best_matches:
    #     if key_embeddings in item:
    #         item.pop(key_embeddings)

    # Sort by distance in ascending order
    best_matches.sort(key=lambda x: x['distance'])  

    with open("search-sd-fc.json", "w") as outfile:
        json.dump(best_matches, outfile, indent=4)

    print("Completed search function execution")

    return best_matches

def generate_questions(matches):
    """
    Generate questions based on the provided matches. Requires a list of
    dictionaries ("matches") representing the matched reports. Returns a list of
    questions generated from the matched reports.    
    """
    # matches_file = "search-sd-fc.json"
    # matches = load_json(matches_file)
    print("Running generate_questions function")

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
             \"n.\" <question_n> \n\n The total number of characters must be less \
             than 250 characters."},
            {"role": "user", "content": f"\n\n{combined_texts}\n\nQuestions:"}
            ],
        max_tokens=250
    )
    
    questions = response.choices[0].message.content.strip().split('\n')

    with open("questions-fc.json", "w") as outfile:
        json.dump(questions, outfile, indent=4)

    print("Completed generate_questions function execution")

    return questions

def generate_answers(questions):
    """
    Generate a plausible answer for the given question using OpenAI's API. 
    Requires the question text. Returns generated answer text.
    """
    # questions_file = sys.argv[1]
    # questions = load_json(questions_file)
    print("Running generate_answers function")

    answers = defaultdict(list)
    for question in questions:
        
        answer = generate_answer(question)
        answers[str(question)] = answer

    answers_file = 'answers.json'
    with open(answers_file, 'w') as file:
        json.dump(answers, file, indent=4)

    print("Completed generate_answers function execution")

    return answers

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

def augment_report(query):
    """
    Augment the given report with additional information from answers. Requires
    the original report text ("report") and a dictionary containing questions
    and their respective answers ("answers"). Returns the augmented report text.
    """    
    print("Running augment_report function")
    
    answers_file = "answers.json"
    answers = load_json(answers_file)

    prompt = f"Original Report:\n{query}\n\nAdditional Information:\n"
    for question, answer in answers.items():
        prompt += f"Q: {question}\nA: {answer}\n"

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
    
    with open("augmented-report.json", "w") as file:
        json.dump(augmented_report, file, indent=4)    
    
    print("Completed augment_report function execution\n")

    print(augmented_report)

    return augmented_report

# ======================= HELPER FUNCTIONS =============================== #

def calculate_sse(embeddings: list, centroid: list) -> float:
    """
    Calculate the Sum of Squared Errors (SSE) for a set of embeddings and a
    centroid. Requires a list of embeddings and the centroid's embedding.
    Returns the SSE (error sum of squares) value.
    """
    sse = sum(np.sum((np.array(embedding) - np.array(centroid)) ** 2) for embedding in embeddings)
    return sse

def find_centroid(embeddings: list) -> list:
    """
    Find the centroid of a set of embeddings. Requires a list of embeddings.
    Returns the centroid embedding.
    """
    centroid = np.mean(embeddings, axis=0).tolist()
    return centroid


if __name__ == "__main__":
    main()
