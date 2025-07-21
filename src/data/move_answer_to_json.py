"""
given a txt with answers, move it to the input json
"""

import json
import sys

if __name__ == "__main__":
    # filename = sys.argv[1]

    filename = "llama_original"
    with open("experiments/input_data/original_input.json", "r") as f:
        questions = json.load(f)

    with open(f"experiments/output_data/{filename}.txt", "r") as f:
        answers = f.read().splitlines()

    assert len(questions) == len(answers), "sizes don't match"

    output_json = []

    i = 0
    for q, a in zip(questions, answers):
        # print(q["prompt"], a)
        model_answer = q.copy()
        model_answer["model_output"] = a
        # print(model_answer["prompt"])
        output_json.append(model_answer)

    with open(f"experiments/output_data/{filename}.json", "w") as f:
        json.dump(output_json, f)
