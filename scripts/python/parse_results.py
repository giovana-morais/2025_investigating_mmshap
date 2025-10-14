"""
This script read individual .json files and combine it into a single json for
every experiment we run.
Available experiments are:
    * qwen_fs (qwen-audio, mcpi)
    * qwen_zs (qwen-audio, mcnpi)
    * mu_fs   (mullama, mcpi)
    * mu_zs   (mullama, mcnpi)
"""
#!/usr/bin/env python
# coding: utf-8

import glob
import os
import json


def combine_output(json_list):
    """
    receive a list of output .json files. load then and combine them into a single one.
    """
    data = []
    for jf in json_list:
        with open(jf, "r") as f:
            data.append(json.load(f))
    return data


if __name__ == "__main__":
    mullama_fs_list = glob.glob(
        "../data/output_data/mullama_muchomusic_musiccaps_fs/*.json"
    )
    mullama_zs_list = glob.glob(
        "../data/output_data/mullama_muchomusic_musiccaps_zs/*.json"
    )
    qwen_fs_list = glob.glob(
        "../data/output_data/qwenaudio_muchomusic_musiccaps_fs/*.json"
    )
    qwen_zs_list = glob.glob(
        "../data/output_data/qwenaudio_muchomusic_musiccaps_zs/*.json"
    )
    qwen_toy_list = glob.glob(
        "../data/output_data/qwenaudio_toy_example/*.json"
    )

    qwen_fs = combine_output(qwen_fs_list)
    qwen_zs = combine_output(qwen_zs_list)
    mu_fs = combine_output(mullama_fs_list)
    mu_zs = combine_output(mullama_zs_list)
    qwen_toy = combine_output(qwen_toy_list)

    exps_info = {
        "mu_fs": {
            "folder_name": "mullama_muchomusic_musiccaps_fs",
        },
        "mu_zs": {
            "folder_name": "mullama_muchomusic_musiccaps_zs",
        },
        "qwen_fs": {
            "folder_name": "qwenaudio_muchomusic_musiccaps_fs",
        },
        "qwen_zs": {
            "folder_name": "qwenaudio_muchomusic_musiccaps_zs",
        },
        "qwen_toy": {
            "folder_name": "qwenaudio_toy_example",
        }
    }

    for kexp, vexp in exps_info.items():
        print(kexp, vexp)
        output_json = combine_output(
            glob.glob(f"../../data/output_data/{vexp['folder_name']}/*.json")
        )

        with open(f"../../data/output_data/{kexp}.json", "w") as f:
            json.dump(output_json, f)
