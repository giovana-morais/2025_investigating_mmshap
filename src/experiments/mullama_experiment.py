"""
run MM-SHAP for MU-LLaMA
"""
import json
import os
import time

import numpy as np
import shap
import torch.cuda

import models.MULLaMA.MU_LLaMA.llama as llama
import utils
from models.MULLaMA.MU_LLaMA.util.misc import *
from models.MULLaMA.MU_LLaMA.data.utils import load_and_transform_audio_data

SAMPLE_RATE=24000

def explain_ALM(
        entry,
        audio_path,
        model,
        args,
        audio_weight=1,
        cache_size=100,
        cache_t=20.0,
        cache_weight=0.0,
        max_gen_len=256,
        gen_t=0.6, top_p=0.8
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

        masked_question_tokens = torch.tensor(x[:, -input_ids.shape[1]:])

        # ids that we need to mask from audio (n, n_audio_tokens)
        masked_audio_token_ids = torch.tensor(x[:, :-input_ids.shape[1]])

        # results is the (number of permutations, number of output_ids)
        result = np.zeros((masked_question_tokens.shape[0], output_ids.shape[1]))

        # get the size (in samples) of the windows we're masking
        audio_segment_size = audio.shape[0] // masked_audio_token_ids.shape[1]

        # initialize dictionary to add masked audio
        masked_inputs = {}

        for i in range(masked_question_tokens.shape[0]):
            masked_prompt = masked_question_tokens[i].unsqueeze(0).clone().detach().to("cuda:0")
            masked_audio = audio.clone().detach()

            # zero the audio patches (audio is already resampled)
            to_mask = torch.where(masked_audio_token_ids[i] == 0)[0]
            for k in to_mask:
                start = k * audio_segment_size
                end = min((k + 1) * audio_segment_size, masked_audio.shape[0])  # ensure we don't go past the end
                masked_audio[start:end] = 0

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

    # audio is an array. we mask it directly.
    audio = load_and_transform_audio_data([audio_path])
    inputs = {}
    inputs['Audio'] = [audio, audio_weight]
    response = None

    # i don't really understand why they put this inside a list, but for now i'm
    # just keeping whatever the authors of the repo have as an example
    prompts = [llama.format_prompt(entry["prompt"])]
    prompts = [model.tokenizer.encode(x, bos=True, eos=False) for x in prompts]

    question_tokens = model.tokenizer.encode(entry["prompt"], bos=True, eos=False,
        out_type=str)

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

    # define how many patches we need to cover the audio based on text length
    n_question_tokens = input_ids.shape[1]
    audio_ids = torch.tensor(range(-1, -(n_question_tokens + 1), -1)).unsqueeze(0)
    entry["n_question_tokens"] = n_question_tokens
    entry["n_audio_tokens"] = audio_ids.shape[-1]
    input_ids = input_ids.to("cuda:0")
    audio_ids = audio_ids.to("cuda:0")

    X = torch.cat((audio_ids, input_ids), 1).unsqueeze(1)
    X.to("cpu")
    input_ids = input_ids.to("cpu")

    explainer = shap.Explainer(get_prediction, token_masker, silent=True, max_evals=800)
    shap_values = explainer(X)

    outfile = os.path.join(entry["output_folder"],
            f"{entry['question_id']}_info.npz")

    np.savez(outfile,
            shapley_values=shap_values.values,
            base_values=shap_values.base_values,
            input_ids=X.cpu().numpy(),
            input_tokens_str=question_tokens,
            output_tokens_str=model.tokenizer.decode_ids(output_ids.squeeze(0))
    )

    return response


if __name__ == "__main__":
    args = utils.parse_arguments()

    if args.environment == "hpc":
        dataset_path = "/scratch/gv2167/datasets"
    else:
        dataset_path = "/media/gigibs/DD02EEEC68459F17/datasets"

    with open(args.input_path, "r") as f:
        questions = json.load(f)

    if args.range is not None:
        questions = questions[args.range * args.index : args.range * (args.index + 1)]
        print(
            f"processing {len(questions)} questions from {args.range * args.index}:{args.range * (args.index + 1)}"
        )
    else:
        print(f"processing {len(questions)} questions")

    # load model
    base_path = "/scratch/gv2167/2025_investigating_mmshap/src/models/MULLaMA/MU_LLaMA"
    mullama_dir = os.path.join(base_path, "ckpts/checkpoint.pth")
    llama_dir = os.path.join(base_path, "ckpts/LLaMA")
    llama_type = "7B"
    mert_path = "m-a-p/MERT-v1-330M"
    knn_dir = os.path.join(base_path, "ckpts")

    model = llama.load(mullama_dir, llama_dir, mert_path=mert_path, knn=True, knn_dir=knn_dir, llama_type=llama_type)
    model.eval()

    # start processing
    start = time.time()

    experiment_type = (
        f"{args.model}_{os.path.basename(args.input_path).replace('.json', '')}"
    )
    print("experiment type", experiment_type)

    for entry in questions:
        kwargs = {}

        audio_url = os.path.join(
            dataset_path, "/".join(entry["audio_path"].split("/")[1:])
        )
        output_folder = os.path.join(
            "data/output_data", experiment_type
        )
        entry["output_folder"] = output_folder
        os.makedirs(output_folder, exist_ok=True)

        try:
            response = explain_ALM(entry=entry, audio_path=audio_url,
                    model=model, args=args)
            entry["model_output"] = response

            with open(output_folder + f"{entry['question_id']}.json", "w") as f:
                json.dump(entry, f)

        except Exception as e:
            print(f"ERROR: Could not process song {entry['audio_path']}. Reason: {e}")

    end = time.time()
    print(f"execution for {len(questions)}: {(end - start) / 60} minutes")
