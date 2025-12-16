"""
runs MM-SHAP for QwenAudio

example usage
    python experiments/qwenaudiochat_bloomberg.py \
            --input_path=data/bloomberg_data/question.json
"""

import json
import os
import time

import numpy as np
import shap
import torch
import yaml
from transformers import AutoModelForCausalLM

import models.Qwen_Audio.qwen_generation_utils as qwen_gen_utils
import parsing
from models.custom_qwen_tokenizer import CustomQwenTokenizer
from models.Qwen_Audio.audio import load_audio, SAMPLE_RATE


def compute_tokens(
    model,
    tokenizer,
    query,
    history,
    system="You are a helpful assistant.",
    append_history=None,
    stop_words_ids=None,
    **kwargs,
):
    """
    Based on `QWenLMHeadModel.chat` function of the Qwen-Audio repo. Tokenizes
    the question and returns the necessary tokens for the output generation.
    """
    generation_config = model.generation_config

    history = []

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


def explain_ALM(entry, audio_url, model, tokenizer, args, **kwargs):
    """
    Parameters
    ---
        entry : dict
        audio_url : string
        model : model
        tokenizer : tokenizer
        args : parsed args

    Returns
    ---
    """

    def token_masker(mask, x):
        """
        Mask audio tokens (which will be later converted to audio segments)
        """
        masked_X = x.clone().detach()
        mask = torch.tensor(mask).unsqueeze(0)

        # apply mask to audio tokens
        masked_X[~mask] = 0

        return masked_X.to("cpu")

    def get_prediction(x):
        nonlocal input_ids
        nonlocal output_ids
        nonlocal audio_info
        nonlocal n_audio_tokens

        # tokens to mask audio. (n, n_audio_tokens)
        masked_audio_token_ids = torch.tensor(x[:, :-n_audio_tokens])

        # results.shape is (number of permutations, number of output_ids)
        result = np.zeros((masked_audio_token_ids.shape[0], output_ids.shape[1]))

        # get the size (in samples) of the windows we're masking
        audio_segment_size = audio.shape[0] // n_audio_tokens

        for i in range(masked_audio_token_ids.shape[0]):
            # replace the question tokens for the masked ones, keep everything else
            iteration_input_id = input_ids.clone().detach().to("cuda:0")

            # zero the audio segments
            masked_audio = audio.clone().detach()
            to_mask = torch.where(masked_audio_token_ids[i] == 0)[0]

            for k in to_mask:
                start = k * audio_segment_size
                end = min(
                    (k + 1) * audio_segment_size, masked_audio.shape[0]
                )  # ensure we don't go past the end
                masked_audio[start:end] = 0

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
            {"text": entry["prompt"]},
        ]
    )

    system_instruction = "You are a helpful assistant. Give short answers to the questions."

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
    # we filter here the last two special tokens (<im_end> and <end of text>)
    output_ids = output_ids[:, :-2]

    input_ids.to("cpu")
    output_ids.to("cpu")

    audio = load_audio(audio_url, sr=SAMPLE_RATE)
    audio = torch.from_numpy(audio)

    # audio windows have negative token_ids to distinguish them from text tokens
    n_audio_tokens = int(audio.shape[0] // (SAMPLE_RATE * 0.1))
    # we are NOT relying on text features anymore
    audio_token_ids = torch.tensor(range(-1, -(n_audio_tokens + 1), -1))
    audio_token_ids = audio_token_ids.unsqueeze(0).unsqueeze(1)
    audio_token_ids = audio_token_ids.to("cuda:0")

    entry["n_audio_tokens"] = audio_token_ids.shape[-1]
    print(f"number of audio tokens: {audio_token_ids.shape}")

    X = audio_token_ids
    X.to("cpu")

    explainer = shap.Explainer(get_prediction, token_masker, silent=True, max_evals=600)
    shap_values = explainer(X)
    print("shap_values.shape", shap_values.shape)

    outfile = os.path.join(entry["output_folder"], f"{entry['question_id']}_info.npz")

    np.savez(
        outfile,
        shapley_values=shap_values.values,
        base_values=shap_values.base_values,
        input_ids=X.cpu().numpy(),
        output_tokens_str=[
            i.decode("utf-8")
            for i in tokenizer.convert_ids_to_tokens(output_ids.squeeze(0))
        ],
    )

    return response


if __name__ == "__main__":
    args = parsing.parse_arguments()

    if args.environment == "hpc":
        dataset_path = "/scratch/gv2167/datasets"
    else:
        dataset_path = "/media/gigibs/DD02EEEC68459F17/datasets"

    with open(args.input_path, "r") as f:
        questions = json.load(f)

    print(f"processing {len(questions)} questions")

    with open("config.yml", "r") as f:
        config = yaml.safe_load(f)

    # setup tokenizer
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

    # setup model
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen-Audio-Chat",
        device_map="cuda",
        trust_remote_code=True,
    ).eval()

    start = time.time()

    experiment_type = (
        f"{args.model}_{os.path.basename(args.input_path).replace('.json', '')}_audio_only"
    )
    print("experiment type", experiment_type)

    for entry in questions:
        kwargs = {}

        # in this, we have the data inside /data/bloomberg_data
        audio_url = entry["audio_path"]

        output_folder = os.path.join("data/output_data", experiment_type)
        entry["output_folder"] = output_folder
        os.makedirs(output_folder, exist_ok=True)

        # try:
        response = explain_ALM(entry, audio_url, model, tokenizer, args)
        entry["model_output"] = response

        with open(
            os.path.join(output_folder, f"{entry['question_id']}.json"), "w"
        ) as f:
            json.dump(entry, f)
        # except Exception as e:
        #     print(f"ERROR: Could not process song {entry['audio_path']}. Reason: {e}")

    end = time.time()
    print(f"execution for {len(questions)}: {(end - start) / 60} minutes")
