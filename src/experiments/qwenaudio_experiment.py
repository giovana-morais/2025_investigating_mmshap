"""
runs MM-SHAP for QwenAudio

example usage
    # run for all
    python experiments/qwenaudiochat_mmshap.py \
            --input_path=data/input_data/muchomusic_original_fs.json \
            --output_path=data/output_data/qwenaudio_original_fs.json

    # run for first 10 q&a pairs
    python experiments/qwenaudiochat_mmshap.py \
            --input_path=data/input_data/muchomusic_original_fs.json \
            --output_path=data/output_data/qwenaudio_original_fs.json \
            --index=0 \
            --range=50
"""

import json
import os
import time

import numpy as np
import shap
import torch
import torchaudio
import torchaudio.functional as F
import yaml
from transformers import AutoModelForCausalLM
# torch.manual_seed(1234)

from mmshap import compute_mm_score
from models.custom_qwen_tokenizer import CustomQwenTokenizer
from models.Qwen_Audio.audio import load_audio, SAMPLE_RATE
from utils import *
import models.Qwen_Audio.qwen_generation_utils as qwen_gen_utils


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
    Based on `QWenLMHeadModel.chat` function of the Qwen-Audio repo. Tokenizes
    the question and returns the necessary tokens for the output generation.
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
    """
    Parameters
    ---
        question : string
        audio_url : string
        model : model
        tokenizer : tokenizer
        args : parsed args

    Returns
    ---
    """
    def token_masker(mask, x):
        """
        Receives only valid tokens to mask. We don't do anything about audio
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
            # replace the question tokens for the masked ones, keep everything else
            iteration_input_id = masked_input_ids.clone().detach().to("cuda:0")
            iteration_input_id[:, interval[0] : interval[1]] = masked_text_tokens[i, :]

            # zero the audio patches
            masked_audio = audio.clone().detach()
            to_mask = torch.where(masked_audio_token_ids[i] == 0)[0]
            for k in to_mask:
                masked_audio[k * SAMPLE_RATE : (k + 1) * SAMPLE_RATE] = 0

            masked_audio_info = tokenizer.process_audio_no_url(masked_audio, audio_url)
            kwargs["audio_info"] = masked_audio_info

            # generate answer with masked inputs
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

        return result

    ### ==== Calculate baseline (outputs without any masking) ====
    query = tokenizer.from_list_format(
        [
            {"audio": audio_url},
            {"text": question},
        ]
    )

    system_instruction = "You are a helpful assistant."

    # get input_ids
    input_ids, stop_words_ids, audio_info, raw_text, context_tokens = compute_tokens(
        model, tokenizer, query=query, system=system_instruction, history=None
    )
    kwargs["audio_info"] = audio_info

    # generate output_tokens
    outputs = model.generate(
        input_ids,
        stop_words_ids=stop_words_ids,
        generation_config=model.generation_config,
        **kwargs,
    )

    # decode tokens and generate string response
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

    # audio windows have negative token_ids to distinguish them from text tokens
    audio_token_ids = torch.tensor(range(-1, -(n_text_tokens+1), -1)).unsqueeze(0)
    audio_token_ids = audio_token_ids.to("cuda:0")
    print(f"number of text tokens: {n_text_tokens}")
    print(f"number of audio tokens: {audio_token_ids.shape}")

    # concatenate text and audio tokens
    X = torch.cat((audio_token_ids, text_tokens), 1).unsqueeze(1)
    X.to("cpu")

    explainer = shap.Explainer(get_prediction, token_masker, silent=True, max_evals=600)
    shap_values = explainer(X)
    print("shap_values.shape", shap_values.shape)

    # TODO: change this afterwards to receive the track id
    # np.save(f"{args.prompt_type}_shapley_values.npy", shap_values.values)
    # np.save(f"{args.prompt_type}_base_values.npy", shap_values.base_values)
    # np.save(f"{args.prompt_type}_tokens", X)
    mm_score = compute_mm_score(audio_token_ids.shape[1], shap_values,
            verbose=True)

    return response, mm_score


if __name__ == "__main__":
    args = utils.parse_arguments()

    if args.environment == "hpc":
        data_path = "/scratch/gv2167/datasets"
    else:
        data_path = "/media/gigibs/DD02EEEC68459F17/datasets"

    with open(args.input_path, "r") as f:
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
        "</audio>": tokenizer.convert_tokens_to_ids("</audio>"),
        "<|im_end|>": tokenizer.convert_tokens_to_ids("<|im_end|>"),
        "<audio_padding>": tokenizer.convert_tokens_to_ids("<audio_padding>"),
    }

    start = time.time()
    mm_scores = []

    for q in questions:
        kwargs = {}

        audio_url = os.path.join(data_path, "/".join(q["audio_path"].split("/")[1:]))
        question = q["prompt"]

        try:
            response, question_mm_score = explain_ALM(
                question, audio_url, model, tokenizer, args
            )
            mm_scores.append(question_mm_score)
            q["model_output"] = response
            q["mmshap_text"] = question_mm_score
        except Exception as e:
            print(f"could not process song {q['audio_path']}. reason: {e}")

    end = time.time()
    print(f"execution for {len(questions)}: {(end - start) / 60} minutes")
    mm_scores = np.asarray(mm_scores)
    print(f"avg text score for the dataset: {mm_scores.mean()}")

    with open(output_path, "w") as f:
        json.dump(questions, f)
