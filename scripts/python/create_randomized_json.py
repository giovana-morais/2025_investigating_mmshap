"""
the original muchomusic sample question answer is always A. this script
creates a new version that randomizes the answer.
"""

import json
import random
from collections import Counter


def fs_answer_distribution(questions):
    answers = []
    for q in questions:
        answers.append(q["prompt"].split("\n")[2][-1])

    return Counter(answers)


def randomize_answer(question):
    options = ["A", "B", "C", "D"]
    new_option = random.choice(options)
    new_question = question.replace(
        " The correct answer is: A", f" The correct answer is: {new_option}"
    )

    return new_question


if __name__ == "__main__":
    with open("muchomusic_eval/example_file.json", "r") as f:
        questions = json.load(f)

    fs_answer_dist = fs_answer_distribution(questions)
    print(fs_answer_dist)

    all_options = []
    new_questions = []
    for q in questions:
        q["prompt"] = randomize_answer(q["prompt"])
        new_questions.append(q)

    random_fs_answer_dist = fs_answer_distribution(new_questions)
    print(random_fs_answer_dist)

    with open("muchomusic_eval/randomized_input.json", "w") as f:
        json.dump(new_questions, f)
