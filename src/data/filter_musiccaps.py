"""
Create filtered version of MuChoMusic with MusicCaps questions only
"""

import argparse
import json

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Create filtered version of MuChoMusic with MusicCaps questions only"
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

    filtered_output = []

    for q in questions:
        if q["dataset"] == "musiccaps":
            filtered_output.append(q)

    with open(args.output_path, "w") as f:
        json.dump(filtered_output, f)

    print(f"done! total remaning questions: {len(filtered_output)}")
