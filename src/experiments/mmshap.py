import numpy as np

def compute_mm_score(audio_length, shap_values, method="sum", verbose=False):
    """
    Compute Multimodality Score. Assumes that audio tokens are concatenated
    *before* text tokens.

    Parameters
    ---
        audio_length :  int
            number of audio tokens
    """
    if method == "sum":
        audio_contrib = np.abs(shap_values[0, 0, :audio_length, :]).sum()
        text_contrib = np.abs(shap_values[0, 0, audio_length:, :]).sum()

        text_score = text_contrib / (text_contrib + audio_contrib)
        audio_score = audio_contrib / (text_contrib + audio_contrib)
    elif method == "avg":
        audio_contrib = np.abs(shap_values[0, 0, :audio_length, :]).mean()
        text_contrib = np.abs(shap_values[0, 0, audio_length:, :]).mean()

        text_score = text_contrib / (text_contrib + audio_contrib)
        audio_score = audio_contrib / (text_contrib + audio_contrib)

    if verbose:
        print("compute_mm_score")
        print("audio contribution", audio_contrib)
        print("text contribution", text_contrib)
        print("text score", text_score)
        print("audio score", audio_score)
    return audio_score, text_score
