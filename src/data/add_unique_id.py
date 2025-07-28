"""
Create a new field in the muchomusic original .csv to differentiate between
different questions with the same track id
"""
import argparse
import json


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Create unique id column"
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        default=None,
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        default=None,
    )

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()

    with open(args.input_path, "r") as f:
        questions = json.load(f)

    question_counter = 0
    for q in questions:
        q["question_id"] = question_counter
        question_counter += 1

    print(f"Save file as {args.output_path}")
    with open(args.output_path, "w") as f:
        json.dump(questions, f)
