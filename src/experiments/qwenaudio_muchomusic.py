"""
runs MM-SHAP for QwenAudio

example usage
    python experiments/qwenaudiochat_mmshap.py --input=original --environment=hpc --prompt_type=zero_shot --system_instruction=default
"""

import argparse
import json
import math
import os
import tempfile
import time

import numpy as np
import torch
import torchaudio
import torchaudio.functional as F
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
# torch.manual_seed(1234)

import shap
from models.custom_qwen_tokenizer import CustomQwenTokenizer
import models.Qwen_Audio.qwen_generation_utils as qwen_gen_utils
from models.Qwen_Audio.audio import *
from qwenaudiochat_utils import *
from mmshap import compute_mm_score

SAMPLE_RATE = 16000  # from QwenAudio/audio.py

def compute_tokens(
    model,
    tokenizer,
    query,
    history,
    system="You are a helpful assistant",
    append_history=None,
    stop_words_ids=None,
    **kwargs,
):
    """
    based on `chat` function of the Qwen-Audio repo. adapted so i can get the
    tokens.
    """
    generation_config = model.generation_config

    if history is None:
        history = []
    else:
        # make a copy of the user's input such that is is left untouched
        history = copy.deepcopy(history)

    if stop_words_ids is None:
        stop_words_ids = []

    max_window_size = kwargs.get("max_window_size", None)
    if max_window_size is None:
        max_window_size = 6144  # stolen from chat_generation_config.json

    raw_text, context_tokens, audio_info = qwen_gen_utils.make_context(
        tokenizer,
        query,
        history=history,
        system=system,
        max_window_size=max_window_size,
        chat_format=generation_config.chat_format,
    )

    stop_words_ids.extend([[tokenizer.im_end_id], [tokenizer.im_start_id]])

    input_ids = torch.tensor([context_tokens]).to(model.device)

    return input_ids, stop_words_ids, audio_info, raw_text, context_tokens


def explain_ALM(question, audio_url, model, tokenizer, args):
    ### internal functions
    def token_masker(mask, x):
        """
        receives only valid tokens to mask. we don't do anything about audio
        tokens yet.
        """
        masked_X = x.clone().detach()
        mask = torch.tensor(mask).unsqueeze(0)

        audio_mask = mask.clone().detach()
        audio_mask[:, -n_text_tokens:] = True  # do not mask text tokens yet

        # apply mask to audio tokens
        masked_X[~audio_mask] = 0  # ~mask !!! to zero

        text_mask = mask.clone().detach()
        text_mask[:, :-n_text_tokens] = (
            True  # do not do anything to audio tokens anymore
        )

        masked_X[~text_mask] = 0

        return masked_X.to("cpu")

    def get_prediction(x):
        nonlocal input_ids
        nonlocal output_ids
        nonlocal interval  # where the correct text tokens are
        nonlocal n_text_tokens
        nonlocal audio_info

        # text mask itself. shape (n, n_text_token)
        masked_text_tokens = torch.tensor(x[:, -n_text_tokens:])

        # ids that we need to mask from audio (n, n_audio_tokens)
        masked_audio_token_ids = torch.tensor(x[:, :-n_text_tokens])

        # clone original input_ids for inference
        masked_input_ids = input_ids.clone().detach()

        # results is the (number of permutations, number of output_ids)
        result = np.zeros((masked_text_tokens.shape[0], output_ids.shape[1]))

        for i in range(masked_text_tokens.shape[0]):
            iteration_input_id = masked_input_ids.clone().detach().to("cuda:0")
            # replace the text tokens for the masked ones, keep everything else
            iteration_input_id[:, interval[0] : interval[1]] = masked_text_tokens[i, :]

            masked_audio = audio.clone().detach()
            # zero the audio patches (audio is already resampled)
            to_mask = torch.where(masked_audio_token_ids[i] == 0)[0]
            for k in to_mask:
                masked_audio[k * SAMPLE_RATE : (k + 1) * SAMPLE_RATE] = 0

            masked_audio_info = tokenizer.process_audio_no_url(masked_audio, audio_url)

            kwargs["audio_info"] = masked_audio_info

            # generate answer with masked audio and masked input
            outputs = model.generate(
                iteration_input_id,
                stop_words_ids=stop_words_ids,
                return_dict_in_generate=True,
                output_scores=True,
                output_logits=True,
                generation_config=model.generation_config,
                **kwargs,
            )

            logits = outputs.logits[0].detach().cpu().numpy()
            output_ids = output_ids.to("cpu")

            result[i] = logits[0, output_ids]
            break

        return result

    ### main logic
    query = tokenizer.from_list_format(
        [
            {"audio": audio_url},
            {"text": question},
        ]
    )

    system_instruction = "You are a helpful assistant."

    # get input ids
    input_ids, stop_words_ids, audio_info, raw_text, context_tokens = compute_tokens(
        model, tokenizer, query=query, system=system_instruction, history=None
    )

    # print("raw_text", raw_text)
    kwargs["audio_info"] = audio_info

    # generate
    outputs = model.generate(
        input_ids,
        stop_words_ids=stop_words_ids,
        generation_config=model.generation_config,
        **kwargs,
    )

    response = qwen_gen_utils.decode_tokens(
        outputs[0],
        tokenizer,
        raw_text_len=len(raw_text),
        context_length=len(context_tokens),
        chat_format=model.generation_config.chat_format,
        verbose=False,
        errors="replace",
        audio_info=audio_info,
    )

    output_ids = outputs[:, input_ids.shape[1] :]
    input_ids.to("cpu")
    output_ids.to("cpu")

    text_tokens, n_text_tokens, interval = tokenizer.get_number_of_question_tokens(
        input_ids, special_tokens
    )
    text_tokens = text_tokens.unsqueeze(0)
    audio = load_audio(audio_url, sr=SAMPLE_RATE)
    audio = torch.from_numpy(audio)

    # give the audio chunks negative token ids to distinguish them
    # from text tokens
    audio_token_ids = torch.tensor(range(-1, -(n_text_tokens+1), -1)).unsqueeze(0)
    audio_token_ids = audio_token_ids.to("cuda:0")
    print(f"number of text tokens: {n_text_tokens}")
    print(f"number of audio tokens: {audio_token_ids.shape}")

    # concatenate everything
    X = torch.cat((audio_token_ids, text_tokens), 1).unsqueeze(1)
    X.to("cpu")

    explainer = shap.Explainer(get_prediction, token_masker, silent=True, max_evals=600)
    shap_values = explainer(X)
    print("shap_values.shape", shap_values.shape)
    print("shap_values.values.shape", shap_values.values.shape)
    print("shap_values", shap_values.values)
    print("audio_length", audio_token_ids.shape[1])
    # print("audio shap values", shap_values[0,0, :audio_length, :].sum())

    # TODO: change this afterwards to receive the track id
    np.save(f"{args.prompt_type}_shapley_values.npy", shap_values.values)
    np.save(f"{args.prompt_type}_base_values.npy", shap_values.base_values)
    np.save(f"{args.prompt_type}_tokens", X.cpu().numpy())
    audio_score, text_score = compute_mm_score(audio_token_ids.shape[1], shap_values, verbose=True)

    mm_score = audio_score

    return response, mm_score


def parse_arguments():
    # Create the parser
    parser = argparse.ArgumentParser(description="Parse experiment settings.")

    # Define the arguments
    parser.add_argument(
        "--input",
        type=str,
        choices=["original", "random", "qual_analysis"],
        default="original",
        required=False,
        help="Choose the type of experiment: 'original' or 'random'.",
    )
    parser.add_argument(
        "--environment",
        type=str,
        choices=["local", "hpc"],
        required=False,
        default="hpc",
        help="Choose the environment: 'local' or 'hpc'.",
    )
    parser.add_argument(
        "--prompt_type",
        type=str,
        # choices=["zero_shot", "few_shot", "description", "description_zero_shot"],
        default="few_shot",
        required=False,
        help="Use the example question or not.",
    )

    parser.add_argument(
        "--system_instruction",
        type=str,
        choices=["default", "detailed"],
        required=False,
        default="default",
        help="Use QwenAudio default system instruction or not",
    )

    parser.add_argument("--range", type=int, required=False, default=None)

    parser.add_argument("--index", type=int, required=False, default=None)

    parser.add_argument("--output_path", type=str, required=False, default=None)

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

    with open(questions_path, "r") as f:
        questions = json.load(f)

    if args.range is not None:
        questions = questions[args.range * args.index : args.range * (args.index + 1)]
        print(
            f"processing {len(questions)} questions from {args.range * args.index}:{args.range * (args.index + 1)}"
        )
    else:
        print(f"processing {len(questions)} questions")

    if args.output_path is not None:
        output_path = args.output_path

    # use cuda device
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen-Audio-Chat",
        device_map="cuda",
        trust_remote_code=True,
    ).eval()

    with open("config.yml", "r") as f:
        config = yaml.safe_load(f)

    vocab_file = config["qwenaudio"]["vocab_file"]
    tokenizer = CustomQwenTokenizer(vocab_file).from_pretrained(
        "Qwen/Qwen-Audio-Chat", trust_remote_code=True
    )

    tokenizer.padding_side = "left"
    tokenizer.pad_token_id = tokenizer.eod_id

    special_tokens = {
        "<audio>": tokenizer.special_tokens["<audio>"],
        "</audio>": tokenizer.special_tokens["</audio>"],
        "<|choice|>": tokenizer.special_tokens["<|choice|>"],
        "<|answer|>": tokenizer.special_tokens["<|answer|>"],
        "<|startofanalysis|>": tokenizer.special_tokens["<|startofanalysis|>"],
        "<|en|>": tokenizer.special_tokens["<|en|>"],
        "<|question|>": tokenizer.special_tokens["<|question|>"],
        "<|endoftext|>": tokenizer.special_tokens["<|endoftext|>"],
        "<|im_start|>": tokenizer.special_tokens["<|im_start|>"],
        "<|im_end|>": tokenizer.special_tokens["<|im_end|>"],
        "<audio_padding>": 151851,
    }

    start = time.time()
    mm_scores = []

    for q in questions:
        kwargs = {}

        audio_url = os.path.join(data_path, "/".join(q["audio_path"].split("/")[1:]))
        question = q["prompt"]

        if args.prompt_type == "zero_shot":
            question = "Question:" + question.split("Question:")[-1]
        elif args.prompt_type == "description":
            question = "Please describe the song."
        elif args.prompt_type == "description_zero_shot":
            preamble = (
                "Please describe the following audio and then answer the question. "
            )
            question = preamble + "Question:" + question.split("Question:")[-1]
        elif args.prompt_type == "single_question":
            question = "What is the capital of Egypt?"
        elif args.prompt_type == "single_question_mc":
            question = "What is the capital of Egypt?\n(A) Montevideo\n(B) Tokyo\n(C) Cairo\n(D) London"
        elif args.prompt_type == "question_only":
            question = question.split("Question:")[-1].split("\n")[0]

        # try:
        response, question_mm_score = explain_ALM(
            question, audio_url, model, tokenizer, args
        )
        mm_scores.append(question_mm_score)
        q["model_output"] = response
        q["mmshap_text"] = question_mm_score
        # except Exception as e:
        #     print(f"could not process song {q['audio_path']}. reason: {e}")

    end = time.time()
    print(f"execution for {len(questions)}: {(end - start) / 60} minutes")
    mm_scores = np.asarray(mm_scores)
    print(f"avg text score for the dataset: {mm_scores.mean()}")

    with open(output_path, "w") as f:
        json.dump(questions, f)
