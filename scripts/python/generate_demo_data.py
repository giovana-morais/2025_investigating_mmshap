"""
For the selected samples and their respectives .npy files, this script converts
everything to .json so we can use it in the demo page.
"""

import os
import json

import pandas as pd

if __name__ == "__main__":
    src_folder = "/home/gigibs/Documents/research/2025_investigating_mmshap/data/"
    output_folder = "/home/gigibs/Documents/research/2025_investigating_mmshap_demo/data"
    models = ["qwen", "mu"]
    exps = ["fs", "zs"]

    # 5 examples with single sounding event
    sse_examples = [
        427, # laugh example
        719, # bell example
        396, # clock example
        425, # phone ring
        655, # crinckling paper
        869, # telephone sound effect
    ]

    # 5 other random exmaples so no one says that i'm cherry-picking my results.
    random_examples = [
            833, # birds and crickets
    ]

    # for model in models:
    #     for exp in exps:
    #         df = pd.read_json(os.path.join(src_folder, "output_data", f"{model}_{exp}.json"))
    #         print(df.columns)
    #         break
    #     break

    # setting random_state for reproducibiliy
    # random_examples = df.sample(..., random_state=42)

# data example
# {
#         "title": "Example 1: Classical Melody",
#         "text_tokens": ["which", "test", "is", "playing", "the", "melody", "at", "the", "beginning    ", "of", "the", "piece"],
#         "text_shapley_values": [0.01, 0.82, 0.03, 0.75, 0.05, 0.91, 0.07, 0.08, 0.61, 0.1, 0.11, 0    .12],
#         "sample_rate": 16000,
#         "total_duration": 10,
#         "num_audio_samples": 160000,
#         "num_shapley_samples": 200,
#         "audio_signal": [0.0, 0.1, 0.2, "...etc"],
#         "audio_shapley_values": [0.0, 0.0, "...etc"],
#         "gt_start": 2.8,
#         "gt_end": 4.1,
#         "intensity_threshold": 0.8
#         }
