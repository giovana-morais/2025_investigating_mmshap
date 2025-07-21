"""
calculate result for llama experiments

IMPORTANT!!!!
this currently does not work for qwenaudio (i'm still implementing the parsing)
"""

import json
import sys


def letter_to_index(model_output):
    """
    convert answer letter (a, b, c, d) into index
    """
    letter_options = ["A", "B", "C", "D"]

    return letter_options.index(model_output)


def compare_answers(model_output, answer_order):
    ground_truth = answer_order.index(0)
    prediction = letter_to_index(model_output)

    return ground_truth == prediction


if __name__ == "__main__":
    # filename = "llama_original"
    # filename = "llama_randomized"
    filename = sys.argv[1]

    with open(f"output_data/{filename}.json", "r") as f:
        responses = json.load(f)

    total_questions = 0
    total_correct = 0

    for r in responses:
        total_questions += 1
        if compare_answers(r["model_output"], r["answer_orders"]):
            total_correct += 1

    print(f"Total questions {total_questions}")
    print(f"Total correct {total_correct} ({total_correct / total_questions:.2%})")
