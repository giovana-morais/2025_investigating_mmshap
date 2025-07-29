"""
run MM-SHAP for MU-LLaMA
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch.cuda

import llama
from util.misc import *
from data.utils import load_and_transform_audio_data
import shap

SAMPLE_RATE=24000

def compute_mm_score(audio_length, shap_values):
    """
    Compute Multimodality Score.
    """
    # print("shap_values", shap_values.shape)
    # print("contrib shape", shap_values.values[0, 0, :audio_length, :].shape)
    # input("")
    audio_contrib = np.abs(shap_values.values[0, 0, :audio_length, :]).sum()
    text_contrib = np.abs(shap_values.values[0, 0, audio_length:, :]).sum()
    text_score = text_contrib / (text_contrib + audio_contrib)
    return text_score

def multimodal_generate(
        audio_path,
        audio_weight,
        prompt,
        cache_size,
        cache_t,
        cache_weight,
        max_gen_len,
        gen_t, top_p
):
    def token_masker(mask, x):
        """
        receives only valid tokens to mask. we don't do anything about audio
        tokens yet.
        """
        masked_X = x.clone().detach()
        mask = torch.tensor(mask).unsqueeze(0)

        audio_mask = mask.clone().detach()
        audio_mask[:, -input_ids.shape[1]:] = True # do not mask text tokens yet

        # apply mask to audio tokens
        masked_X[~audio_mask] = 0  # ~mask !!! to zero

        text_mask = mask.clone().detach()
        text_mask[:, :-input_ids.shape[1]] = True # do not do anything to audio tokens anymore

        masked_X[~text_mask] = 0
        return masked_X.to("cpu")

    def get_prediction(x):
        nonlocal output_ids
        nonlocal input_ids
        nonlocal audio

        # text mask itself. shape (n, n_text_token)
        # print("audio and input shaoe", audio_ids.shape, input_ids.shape)
        # print("x.shape", x.shape)
        masked_text_tokens = torch.tensor(x[:, -input_ids.shape[1]:])
        # print("masked_text_tokens.shape", masked_text_tokens.shape)

        # ids that we need to mask from audio (n, n_audio_tokens)
        masked_audio_token_ids = torch.tensor(x[:, :-input_ids.shape[1]])

        # results is the (number of permutations, number of output_ids)
        result = np.zeros((masked_text_tokens.shape[0], output_ids.shape[1]))
        masked_inputs = {}

        for i in range(masked_text_tokens.shape[0]):
            masked_prompt = masked_text_tokens[i].unsqueeze(0).clone().detach().to("cuda:0")
            masked_audio = audio.clone().detach()
            # zero the audio patches (audio is already resampled)
            to_mask = torch.where(masked_audio_token_ids[i] == 0)[0]
            for k in to_mask:
                masked_audio[k*SAMPLE_RATE:(k+1)*SAMPLE_RATE] = 0

            masked_inputs["Audio"] = [masked_audio, 1.]
            # generate answer with masked audio and masked input
            results, tokens, output_logits = model.generate(
                masked_inputs,
                masked_prompt,
                max_gen_len=max_gen_len,
                temperature=gen_t,
                top_p=top_p,
                cache_size=cache_size,
                cache_t=cache_t,
                cache_weight=cache_weight
            )

            logits = output_logits.detach().cpu().numpy()
            output_ids = output_ids.to("cpu")

            result[i] = logits[0, output_ids]
        return result

    inputs = {}
    # audio is an array. we mask it directly.
    audio = load_and_transform_audio_data([audio_path])
    inputs['Audio'] = [audio, audio_weight]
    response = None

    # i don't really understand why they put this insize a list, but for now i'm
    # just keeping whatever the authors of the repo have as an example
    prompts = [llama.format_prompt(prompt)]
    prompts = [model.tokenizer.encode(x, bos=True, eos=False) for x in prompts]

    input_ids = torch.tensor(prompts)
    with torch.amp.autocast("cuda"):
        results, tokens, output_logits = model.generate(
            inputs,
            prompts,
            max_gen_len=max_gen_len,
            temperature=gen_t,
            top_p=top_p,
                cache_size=cache_size,
            cache_t=cache_t,
            cache_weight=cache_weight
        )
    response = results[0].strip()

    # removing whatever padding tokens we have after the end of sequence
    # to reduce the computation of masking + shapley values
    eos_position = torch.where(tokens == model.tokenizer.eos_id)[1][0]
    output_ids = tokens[:, input_ids.shape[1]:eos_position]

    logits = output_logits[:, output_ids]

    # define how many patches we need to cover the audio based on text length
    n_patches = math.ceil(math.sqrt(input_ids.shape[1]))
    audio_ids = torch.tensor(range(-1, -n_patches**2-1, -1)).unsqueeze(0)
    input_ids = input_ids.to("cuda:0")
    audio_ids = audio_ids.to("cuda:0")
    # print("input_ids.device", input_ids.device)
    # print("audio_ids.device", audio_ids.device)
    # print("input_ids.shape", input_ids.shape)
    # print("audio_ids.shape", audio_ids.shape)

    X = torch.cat((audio_ids, input_ids), 1).unsqueeze(1)
    X.to("cpu")
    input_ids = input_ids.to("cpu")

    explainer = shap.Explainer(get_prediction, token_masker, silent=True, max_evals=800)
    shap_values = explainer(X)
    np.save(f"/scratch/gv2167/ismir2025/ismir2025_exploration/extreme_case_analysis/mullama_{args.prompt_type}_base_values.npy", shap_values.base_values)
    np.save(f"/scratch/gv2167/ismir2025/ismir2025_exploration/extreme_case_analysis/mullama_{args.prompt_type}_shapley_values.npy", shap_values.values)
    np.save(f"/scratch/gv2167/ismir2025/ismir2025_exploration/extreme_case_analysis/mullama_{args.prompt_type}_tokens.npy", X.cpu().numpy())
    mm_score = compute_mm_score(audio_ids.shape[1], shap_values)
    return response, mm_score

def parse_arguments():
    # Create the parser
    parser = argparse.ArgumentParser(description="Parse experiment settings.")

    # Define the arguments
    parser.add_argument(
        "--input",
        type=str,
        choices=["original", "random", "qual_analysis"],
        required=False,
        help="Choose the type of experiment: 'original' or 'random'."
    )
    parser.add_argument(
        "--environment",
        type=str,
        choices=["local", "hpc"],
        required=False,
        default="hpc",
        help="Choose the environment: 'local' or 'hpc'."
    )
    parser.add_argument(
        "--prompt_type",
        type=str,
        # choices=["zero_shot", "few_shot", "description", "description_zero_shot"],
        default="few_shot",
        required=False,
        help="Use the example question or not."
    )

    parser.add_argument(
        "--system_instruction",
        type=str,
        choices=["default", "detailed"],
        required=False,
        default="default",
        help="Use QwenAudio default system instruction or not"
    )

    parser.add_argument(
        "--range",
        type=int,
        required=False,
        default=None
    )

    parser.add_argument(
        "--index",
        type=int,
        required=False,
        default=None
    )

    parser.add_argument(
        "--output_path",
        type=str,
        required=False,
        default=None
    )

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_arguments()

    if args.environment == "hpc":
        data_path = "/scratch/gv2167/datasets"
    else:
        data_path = "/media/gigibs/DD02EEEC68459F17/datasets"

    if args.input == "original":
        questions_path = "/scratch/gv2167/ismir2025/ismir2025_exploration/experiments/input_data/muchomusic_original.json"
    elif args.input == "random":
        questions_path = "/scratch/gv2167/ismir2025/ismir2025_exploration/experiments/input_data/muchomusic_random.json"
    elif args.input == "qual_analysis":
        questions_path = "/scratch/gv2167/ismir2025/ismir2025_exploration/experiments/input_data/extreme_example.json"

    # # FIXME: horrible overwrite. remove later.
    # if args.prompt_type == "description":
    #     questions_path = "/scratch/gv2167/ismir2025/ismir2025_exploration/experiments/input_data/muchomusic_musiccaps_desc.json"
    # elif args.prompt_type == "question_only":
    #     questions_path = "/scratch/gv2167/ismir2025/ismir2025_exploration/experiments/input_data/muchomusic_musiccaps_qo.json"
    # elif args.prompt_type == "few_shot":
    #     questions_path = "/scratch/gv2167/ismir2025/ismir2025_exploration/experiments/input_data/muchomusic_musiccaps_fs.json"


    with open(questions_path, "r") as f:
        questions = json.load(f)

    questions = questions[args.range*args.index:args.range*(args.index+1)]
    print(f"processing {len(questions)} questions from {args.range*args.index}:{args.range*args.index+1}")

    if args.output_path is not None:
        output_path = args.output_path

    print(f"output_path = {output_path}")

    # load model
    mullama_dir = "/scratch/gv2167/ismir2025/ismir2025_exploration/models/mullama/ckpts/checkpoint.pth"
    llama_dir = "/scratch/gv2167/ismir2025/ismir2025_exploration/models/mullama/ckpts/LLaMA"
    llama_type = "7B"
    knn_dir = "/scratch/gv2167/ismir2025/ismir2025_exploration/models/mullama/ckpts"

    model = llama.load(mullama_dir, llama_dir, knn=True, knn_dir=knn_dir, llama_type=llama_type)
    model.eval()

    # start processing
    start = time.time()
    mm_scores = []
    for q in questions:
        question = q["prompt"]

        if args.prompt_type == "zero_shot":
            question = "Question:" + q["prompt"].split("Question:")[-1]
        elif args.prompt_type == "description":
            question = "Please describe the song."
        elif args.prompt_type == "description_zero_shot":
            preamble = "Please describe the following audio and then answer the question. "
            question = preamble + "Question:" + q["prompt"].split("Question:")[-1]
        elif args.prompt_type == "single_question":
            question = "What is the capital of Egypt?"
        elif args.prompt_type == "question_only":
            question = q["prompt"].split("Question:")[-1].split("\n")[0]
        elif args.prompt_type == "single_question_mc":
            question = "What is the capital of Egypt?\n(A) Montevideo\n(B) Tokyo\n(C) Cairo\n(D) London"

        try:
            audio_url = os.path.join(data_path, "/".join(q["audio_path"].split("/")[1:]))
            response, question_mm_score = multimodal_generate(audio_url, 1, question, 100, 20.0, 0.0, 256, 0.6, 0.8)
            mm_scores.append(question_mm_score)
            q["model_output"] = response
            q["mmshap_text"] = question_mm_score
        except Exception as e:
            print(f"could not process song {q['audio_path']}. reason: {e}")

    end = time.time()
    print(f"execution for {len(questions)}: {(end - start)/60} minutes")
    mm_scores = np.asarray(mm_scores)
    print(f"avg text score for the dataset: {mm_scores.mean()}")
    print(f"avg audio score for the dataset: {1-mm_scores.mean()}")

    with open(output_path, "w") as f:
        json.dump(questions, f)
