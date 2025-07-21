"""
Llama3 inference on MuChoMusic
"""
import argparse
import json
import multiprocessing
import random
from collections import Counter

import transformers
import torch


def model_inference(pipeline, tokenizer, question):
    messages =  [
        {"role": "system", "content": "You will receive multiple choice questions and your answer should be only the letter of the correct choice."},
        {"role": "user", "content": question["prompt"]},
    ]

    output = pipeline(
            messages,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            max_new_tokens=1,
    )
    answer = output[0]["generated_text"][-1]["content"]
    return answer


def create_pipeline(model_id, tokenizer):
    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto"
    )

    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )
    return pipeline

def parse_arguments():
    # Create the parser
    parser = argparse.ArgumentParser(description="Parse experiment settings.")

    # Define the arguments
    parser.add_argument(
        '--input',
        type=str,
        choices=['original', 'random'],
        required=True,
        help="Choose the type of experiment: 'original' or 'random'."
    )
    parser.add_argument(
        '--environment',
        type=str,
        choices=['local', 'hpc'],
        default="hpc",
        required=False,
        help="Choose the environment: 'local' or 'hpc'."
    )

    # Parse the arguments
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_arguments()

    if args.input == "original":
        questions_path = "experiments/input_data/muchomusic_original.json"
        output_file = "experiments/output_data/llama_original.json"
    elif args.input == "random":
        questions_path = "experiments/input_data/muchomusic_random.json"
        output_file = "experiments/output_data/llama_random.json"

    with open(questions_path, "r") as f:
        questions = json.load(f)

    model_id = "/vast/work/public/ml-datasets/llama-3/Meta-Llama-3-8B-Instruct-hf"
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)

    pipeline = create_pipeline(model_id, tokenizer)
    answers = []
    for q in questions:
        answer = q.copy()
        response = model_inference(pipeline, tokenizer, q)
        answer["model_output"] = response
        answers.append(answer)

    # with multiprocessing.Pool(4) as p:
    #     answers = p.map(model_inference, (pipeline, questions))

    # TODO: parametrize experiment name to save this file
    # with open("answers_randomized.txt", "w") as f:
    #     f.write("\n".join(str(i) for i in answers))

    with open(output_file, "w") as f:
        json.dump(answers, f)
