"""
create questions variations for experiments and save them on a different file.

example usage:
    python create_experiments_json.py               \
            --experiment=few_shot                   \
            --base_file=muchomusic_original.json   \
            --output_folder=input_data/

this will output the file as muchomusic_original_fs.json
"""

import argparse
import json
import os


def fs_fn(question):
    return question


def qo_fn(question):
    question = question.split("Question:")[-1].split("\n")[0]
    return question


def zs_fn(question):
    question = "Question:" + question.split("Question:")[-1]
    return question


def desc_fn(question):
    question = "Please describe this song."
    return question


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Create appropriate json files for each experiment"
    )

    parser.add_argument(
        "--experiment",
        type=str,
        choices=["few_shot", "zero_shot", "question_only", "description"],
        required=True,
        default=None,
        help="Name of the experiment",
    )

    parser.add_argument(
        "--base_file",
        type=str,
        required=True,
        default=None,
        help="Path to the file we will use as base to change the questions/prompt",
    )

    parser.add_argument(
        "--output_folder",
        type=str,
        required=True,
        default=None,
        help="Path to the folder where we will save the experiment file",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    with open(args.base_file, "r") as f:
        questions = json.load(f)

    experiment_info = {
        "few_shot": {"fn": fs_fn, "suffix": "fs"},
        "zero_shot": {"fn": zs_fn, "suffix": "zs"},
        "question_only": {"fn": qo_fn, "suffix": "qo"},
        "description": {"fn": desc_fn, "suffix": "desc"},
    }

    exp_fn = experiment_info[args.experiment]["fn"]
    exp_suffix = experiment_info[args.experiment]["suffix"]
    # get filename and add experiment suffix to it
    filename = os.path.splitext(os.path.basename(args.base_file))[0]
    filename += f"_{exp_suffix}.json"

    new_questions = []
    for q in questions:
        q["prompt"] = exp_fn(q["prompt"])
        new_questions.append(q)

    output_file = os.path.join(args.output_folder, filename)
    print(f"Saving json file as {output_file}")
    with open(output_file, "w") as f:
        json.dump(new_questions, f)
