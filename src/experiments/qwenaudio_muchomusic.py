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
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
# torch.manual_seed(1234)

import shap
import models.qwenaudio.qwen_generation_utils as qwen_gen_utils
# from qwenaudiochat_utils import *

SAMPLE_RATE = 16000 # from QwenAudio/audio.py

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
        audio_mask[:, -n_text_tokens:] = True # do not mask text tokens yet

        # apply mask to audio tokens
        masked_X[~audio_mask] = 0  # ~mask !!! to zero

        text_mask = mask.clone().detach()
        text_mask[:, :-n_text_tokens] = True # do not do anything to audio tokens anymore

        masked_X[~text_mask] = 0

        # interesting way of visualizing what happens
        # print("text")
        # print("original", x[~text_mask])
        # print("masked", masked_X)
        # input("")

        # another print that helps understanding
        # for i in range(text_mask.shape[0]):
        #     print("text", torch.count_nonzero(text_mask[i]), len(text_mask[i]))
        #     print("audio", torch.count_nonzero(audio_mask[i]), len(audio_mask[i]))

        return masked_X.to("cpu")

    def get_prediction(x):
        nonlocal input_ids
        nonlocal output_ids
        nonlocal interval # where the correct text tokens are
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
            iteration_input_id[:, interval[0]:interval[1]] = masked_text_tokens[i, :]

            masked_audio = audio.clone().detach()
            # zero the audio patches (audio is already resampled)
            to_mask = torch.where(masked_audio_token_ids[i] == 0)[0]
            for k in to_mask:
                masked_audio[k*SAMPLE_RATE:(k+1)*SAMPLE_RATE] = 0

            # save masked audio as a temporary file and provide the model
            # with the masked audio path.
            # it is absolutely crazy that i didn't find yet a better way of
            # doing this
            with tempfile.NamedTemporaryFile(suffix=".wav") as tmpfile:
                torchaudio.save(tmpfile.name, masked_audio, SAMPLE_RATE, format="wav")
                # add tags before providing the url, otherwise the model
                # screams at me
                masked_audio_info = tokenizer.process_audio(f"<audio>{tmpfile.name}</audio>")

            kwargs["audio_info"] = masked_audio_info

            # generate answer with masked audio and masked input
            outputs = model.generate(
                        iteration_input_id,
                        stop_words_ids=stop_words_ids,
                        return_dict_in_generate=True,
                        output_scores = True,
                        output_logits = True,
                        generation_config=model.generation_config,
                        **kwargs,
                    )

            logits =  outputs.logits[0].detach().cpu().numpy()
            output_ids = output_ids.to("cpu")

            result[i] = logits[0, output_ids]

            # decode tokens back to see how masked question looks like
            # translate_back = tokenizer.decode(iteration_input_id.squeeze(0),audio_info=audio_info)
            # print("\nmasked question", translate_back)

            # response_masked = decode_tokens(
            #     outputs.sequences[0],
            #     tokenizer,
            #     raw_text_len=len(raw_text),
            #     context_length=len(context_tokens),
            #     chat_format=model.generation_config.chat_format,
            #     verbose=True,
            #     errors='replace',
            #     audio_info=masked_audio_info
            # )
            # print(f"response with masked input '{response_masked}'")

        return result

    ### main logic
    query = tokenizer.from_list_format([
        {"audio": audio_url},
        {"text": question},
    ])

    # does muchomusic provide system instructions?
    # if args.system_instruction == "detailed":
    #     system_instruction = "You are a helpful assistant. You will receive multiple choice questions and your answer should be only the letter of the correct choice (e.g., A, B, C, D). Do *not* include any additional text, explanations, or reasoning in your response. "
    # else:
    system_instruction = "You are a helpful assistant."

    # get input ids
    input_ids, stop_words_ids, audio_info, raw_text, context_tokens = compute_tokens(
            model,
            tokenizer,
            query=query,
            system=system_instruction,
            history=None)

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
        audio_info=audio_info
    )

    output_ids = outputs[:, input_ids.shape[1]:]
    input_ids.to("cpu")
    output_ids.to("cpu")

    text_tokens, n_text_tokens, interval = get_number_of_text_tokens(input_ids, special_tokens)
    text_tokens = text_tokens.unsqueeze(0)
    n_patches = int(math.ceil(np.sqrt(n_text_tokens)))
    audio, fs = torchaudio.load(audio_url)
    audio = F.resample(audio, fs, SAMPLE_RATE)

    # give the audio chunks negative token ids to distinguish them
    # from text tokens
    audio_token_ids = torch.tensor(range(-1, -n_patches**2-1, -1)).unsqueeze(0)
    audio_token_ids = audio_token_ids.to("cuda:0")
    print(f"number of text tokens: {n_text_tokens}")
    print(f"number of audio tokens: {len(audio_token_ids)}")

    # concatenate everything
    X = torch.cat((audio_token_ids, text_tokens), 1).unsqueeze(1)
    X.to("cpu")

    explainer = shap.Explainer(get_prediction, token_masker, silent=True, max_evals=600)
    shap_values = explainer(X)

    # TODO: change this afterwards to receive the track id
    np.save(f"{args.prompt_type}_inspect_example.npy", shap_values)
    np.save(f"{args.prompt_type}_tokens", X)
    mm_score = compute_mm_score(audio_token_ids.shape[1], shap_values)

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

    with open(questions_path, "r") as f:
        questions = json.load(f)

    if args.range is not None:
        questions = questions[args.range*args.index:args.range*(args.index+1)]
        print(f"processing {len(questions)} questions from {args.range*args.index}:{args.range*(args.index+1)}")
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


    tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen-Audio-Chat",
            trust_remote_code=True)

    tokenizer.padding_side = "left"
    tokenizer.pad_token_id = tokenizer.eod_id

    special_tokens = {
            "<audio>": tokenizer.special_tokens["<audio>"],
            "</audio>": tokenizer.special_tokens["</audio>"], "<|choice|>": tokenizer.special_tokens["<|choice|>"],
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
            preamble = "Please describe the following audio and then answer the question. "
            question = preamble + "Question:" + question.split("Question:")[-1]
        elif args.prompt_type == "single_question":
            question = "What is the capital of Egypt?"
        elif args.prompt_type == "single_question_mc":
            question = "What is the capital of Egypt?\n(A) Montevideo\n(B) Tokyo\n(C) Cairo\n(D) London"
        elif args.prompt_type == "question_only":
            question = question.split("Question:")[-1].split("\n")[0]

        try:
            response, question_mm_score = explain_ALM(question, audio_url, model, tokenizer, args)
            mm_scores.append(question_mm_score)
            q["model_output"] = response
            q["mmshap_text"] = question_mm_score
        except Exception as e:
            print(f"could not process song {q['audio_path']}. reason: {e}")

    end = time.time()
    print(f"execution for {len(questions)}: {(end - start)/60} minutes")
    mm_scores = np.asarray(mm_scores)
    print(f"avg text score for the dataset: {mm_scores.mean()}")

    with open(output_path, "w") as f:
        json.dump(questions, f)
