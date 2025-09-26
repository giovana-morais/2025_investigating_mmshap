#!/usr/bin/env python
# coding: utf-8
import json

import numpy as np

experiments = ["zs", "fs"]
models = ["qwen", "mu"]

for exp in experiments:
    for m in models:
        print(f"\nEXP: {exp} MODEL {m}")
        input_file_path = f"../data/input_data/muchomusic_musiccaps_{exp}.json"
        output_file_path = f"../data/output_data/{m}_{exp}.json"

        with open(input_file_path, "r") as f:
            input_exp = json.load(f)

        with open(output_file_path, "r") as f:
            output_exp = json.load(f)

        input_qids = np.array([i["question_id"] for i in input_exp])
        output_qids = np.array([i["question_id"] for i in output_exp])
        mask = np.isin(input_qids, output_qids)

        print(f"input: {len(input_qids)}, output: {len(output_qids)}")
        print(f"missing idx: {list(np.argwhere(~mask).T.squeeze(0))}")
        print(f"missing question_ids: {input_qids[~mask]}")
