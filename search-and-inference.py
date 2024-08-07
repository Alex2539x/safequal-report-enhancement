import json
import subprocess
import numpy as np

def run_search(query: str, dataset_file: str) -> str:
    """
    Run the search-sd.py script with the given query and dataset file. Requires
    the query text ("query") and the path to the JSON file containing the
    dataset. Returns the path to the output JSON file with search results.
    """
    search_output_file = 'search-sd.json'
    subprocess.run(['python', 'search_sd.py', query, dataset_file])
    return search_output_file

def run_infer_questions(matches_file: str) -> str:
    """
    Run the infer-questions.py script with the given matches file. Requires the
    path to the JSON file with search results ("matches_file"). Returns the path
    to the output JSON file with generated questions.
    """
    questions_output_file = 'questions.json'
    subprocess.run(['python', 'infer_questions.py', matches_file])
    return questions_output_file


def run_answer_questions(questions_file: str) -> str:
    """
    Run the answer_questions.py script with the given matches file. Requires the
    path to the JSON file with search results ("matches_file"). Returns the path
    to the output JSON file with generated questions.
    """
    answers_output_file = 'answers.json'
    subprocess.run(['python', 'answer_questions.py', questions_file])
    return answers_output_file

def run_augment_report(report: str, answers_file: str) -> str:
    """
    Run the augment-report.py script with the given report and answers file.
    Requires the Original report text ("report") and the path to the JSON
    file with answers ("answers_file"). Returns the augmented report text.
    """
    augment_output_file = 'augmented-report.json'
    subprocess.run(['python', 'augment_report.py', report, answers_file])
    with open(augment_output_file, 'r') as file:
        augmented_report = file.read()
    return augmented_report

def calculate_sse(embeddings: list, centroid: list) -> float:
    """
    Calculate the Sum of Squared Errors (SSE) for a set of embeddings and a
    centroid. Requires a list of embeddings and the centroid's embedding.
    Returns the SSE (error sum of squares) value.
    """
    sse = sum(
        np.sum((np.array(embedding) - np.array(centroid)) ** 2) 
        for embedding in embeddings
        )
    return sse

def find_centroid(embeddings: list) -> list:
    """
    Find the centroid of a set of embeddings. Requires a list of embeddings.
    Returns the centroid embedding.
    """
    centroid = np.mean(embeddings, axis=0).tolist()
    return centroid

# main function; run python search-and-inference.py first
def main():    
    # dataset_file = input("Enter the path to the dataset JSON file: ")
    dataset_file = "k-means.json"

    # run augmented report process in constant loop
    while True:
        query = input("Enter your query (or type 'quit' to exit): ")
        if query.lower() == 'quit':
            break
        
        prev_query_embedding = None
        query_embedding = None
        sse_list = []

        # max num of iterations set to 10 (arbitrary)
        for _ in range(10):  
            # Run the search script
            matches_file = run_search(query, dataset_file)

            with open(matches_file, 'r') as file:
                matches = json.load(file)
            
            match_embeddings = [match['embedding'] for match in matches]
            centroid = find_centroid(match_embeddings)
            sse = calculate_sse(match_embeddings, centroid)
            sse_list.append(sse)

            if prev_query_embedding is not None:
                drift = np.sum((np.array(query_embedding) - np.array(prev_query_embedding)) ** 2)
                # convergence criteria (0.01 is arbitary)
                if drift < 0.01 and abs(sse_list[-1] - sse_list[-2]) < 0.01:  # should be differential (should decrease by 100, or expenentially to some number)
                    print("Convergence criteria met.")
                    break

            # run inference script
            questions_file = run_infer_questions(matches_file)

            # simulate user answers (demo - use script below for manual user input)
            answers_file = run_answer_questions(questions_file)

            # run augment script
            augmented_report = run_augment_report(query, answers_file)

            print("Augmented Report:")
            print(augmented_report)
            print()

            augmented_report_json_file = 'si-augmented_report.json'
            with open(augmented_report_json_file, 'w') as file:
                json.dump({"augmented_report": augmented_report}, file, indent=4)

            print(f"Augmented report has been written to {augmented_report_json_file}\n")

            prev_query_embedding = query_embedding
            query_embedding = find_centroid([match['embedding'] for match in matches])
            query = augmented_report

# run main function
if __name__ == "__main__":
    main()

# python search-and-inference.py
# Enter your query (or type 'quit' to exit): "A patient had a fever."
# Enter the path to the dataset JSON file: k-means.json

# EXAMPLE RUN: search-and-inference.txt

# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

# # User manual input for questions:
# # (Simulate user answers)

# with open(questions_file, 'r') as file:
#     questions = json.load(file)

# answers = {}
# for question in questions:
#     answer = input(f"{question} ")
#     answers[question] = answer

# answers_file = 'answers.json'
# with open(answers_file, 'w') as file:
#     json.dump(answers, file, indent=4)