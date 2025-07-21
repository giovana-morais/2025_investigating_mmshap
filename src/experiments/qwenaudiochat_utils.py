import json
import os

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig


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

    raw_text, context_tokens, audio_info = qgu.make_context(
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
