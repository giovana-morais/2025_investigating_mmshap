import os
import pandas as pd
import numpy as np

from mmshap import compute_mm_score


# FIXME: find a better name for this function
def parse_df(df, experiment_name):
    df["extracted_response"] = df[["model_output", "answers"]].apply(lambda x:
            extract_answer_pandas(x.model_output, x.answers), axis=1)
    df["final_answer"] = df[["extracted_response",
        "answer_orders"]].apply(lambda x: compare_answers(x.extracted_response,
            x.answer_orders), axis=1)
    df["question"] = df[["prompt"]].apply(lambda x:
            x.prompt.split("Question: ")[-1], axis=1)
    df["audio_path"] = df[["audio_path"]].apply(lambda x:
            x.audio_path.replace("data/", ""), axis=1)
    df[["a_shap", "t_shap", "input_tokens", "output_tokens", "input_ids"]] = df.apply(compute_mmshap_row, axis=1)

    df["experiment"] = experiment_name
    df["len_output"] = df[["model_output"]].apply(lambda x: len(x.model_output), axis=1)
    df["n_output_tokens"] = df[["output_tokens"]].apply(lambda x:
            len(x.output_tokens), axis=1)

    df.set_index("question_id", inplace=True)

    return df



def extract_answer_pandas(
    model_output,
    answer_options,
    prefix="The correct answer is:",
    letter_options=["A", "B", "C", "D"]):
    """
    adaptation of muchomusic function but applies to our pandas dataframe
    """

    output = model_output.split(prefix)[-1].strip()
    response = list(set(letter_options).intersection(output))
    if len(response) == 1:
        final_response = letter_options.index(response[0])
    else:
        normalized_output = output.lower().strip()
        normalized_answers = [j.lower().strip() for j in answer_options]

        for j, answer in enumerate(normalized_answers):
            if answer in normalized_output:
                final_response = j
                break
            else:
                final_response = -1
    return final_response


def compare_answers(response, answer_orders):
    """
    return correct/incorrect/unanswered
    """
    answer = 0
    if response == answer_orders.index(0):
        answer = 1
    elif response == -1:
        answer = -1

    return answer


def accuracy(df):
    """
    compute accuracy for answers that followed the instructions
    """
    return df[df["final_answer"] == 1]["final_answer"].count()/df["final_answer"].count()



def compute_mmshap_row(row, agg_method="sum"):
    """
    Compute MM-SHAP value of a row
    """
    # FIXME: change the ".." for the actual repo path
    base_folder = ".."
    file_path = os.path.join(base_folder, row.output_folder, f"{row.question_id}_info.npz")

    output = np.load(file_path)
    shapley_values = output["shapley_values"]
    input_ids = output["input_ids"].squeeze(0).squeeze(0)
    input_tokens = output["input_tokens_str"]
    output_tokens = output["output_tokens_str"]

    audio_length = row.n_audio_tokens
    audio_score, text_score = compute_mm_score(shap_values=shapley_values,
        audio_length=audio_length, method=agg_method)

    return pd.Series({
        "a-shap": audio_score,
        "t-shap": text_score,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "input_ids": input_ids})



def load_shapley_values(input):
    if isinstance(input, pd.core.series.Series):
        return _load_shapley_values_sample(input)
    else: # isinstance(input, pd.core.frame.DataFrame):
        try:
            return _load_shapley_values_row(input)
        except Exception:
            print("format not supported:", type(input), e)
    # else:
    #     print("format not supported:", type(input))


def _load_shapley_values_row(row):
    question_id =  row.Index

    data = f"../{row.output_folder}/{question_id}_info.npz"
    data = np.load(data)
    tokens = row.input_ids
    audio_tokens = np.where(tokens < 0)[-1]
    question_tokens = np.where(tokens >= 0)[-1]

    all_shapley_values = data["shapley_values"].squeeze(0).squeeze(0)
    audio_shapley_values = all_shapley_values[audio_tokens]
    question_shapley_values = all_shapley_values[question_tokens]
    return all_shapley_values, audio_shapley_values, question_shapley_values


def _load_shapley_values_sample(sample):
    question_id =  sample.name

    data = f"../{sample['output_folder']}/{question_id}_info.npz"
    data = np.load(data)
    tokens = sample["input_ids"]
    audio_tokens = np.where(tokens < 0)[-1]
    question_tokens = np.where(tokens >= 0)[-1]

    all_shapley_values = data["shapley_values"].squeeze(0).squeeze(0)
    audio_shapley_values = all_shapley_values[audio_tokens]
    question_shapley_values = all_shapley_values[question_tokens]

    return all_shapley_values, audio_shapley_values, question_shapley_values
