import os
import sys

import numpy as np
import pytest
import torch
import yaml
from transformers import AutoTokenizer

from src.models.custom_qwen_tokenizer import CustomQwenTokenizer
from src.models.Qwen_Audio.audio import *


def test_empty_audio(tokenizer):
    """Test case A: Empty audio should return None"""
    audio_url = "<audio>tests/audio.wav</audio>"
    empty_audio = np.array([])
    result = tokenizer.process_audio_no_url(empty_audio, audio_url)
    assert result is None


def test_mono_audio(tokenizer):
    """Test case B: Mono audio (1 channel)"""
    # Create mono audio (shape: [samples])
    audio_url = "<audio>tests/audio.wav</audio>"
    mono_audio = torch.rand(16000)  # 1 second of audio at 16kHz
    result = tokenizer.process_audio_no_url(mono_audio, audio_url)

    assert isinstance(result, dict)
    assert "input_audios" in result
    assert "input_audio_lengths" in result
    assert "audio_span_tokens" in result
    assert "audio_urls" in result

    # Check tensor shapes
    # assert result["input_audios"].shape == (1, 80, 100)  # (batch, n_mels, time)
    assert result["input_audio_lengths"].shape == (1, 2)  # (batch, [audio_len, token_num])


def test_stereo_audio(tokenizer):
    """Test case C: Stereo audio (2 channels)"""
    # Create stereo audio (shape: [2, samples])
    audio_url = "<audio>tests/audio.wav</audio>"
    stereo_audio = torch.rand(2, 48000)  # 2 channels, 3 seconds at 16kHz
    result = tokenizer.process_audio_no_url(stereo_audio, audio_url)

    assert isinstance(result, dict)
    # Should flatten stereo to mono, so same checks as mono apply
    # assert result["input_audios"].shape == (1, 80, 300)  # Longer audio
    assert result["input_audio_lengths"].shape == (1, 2)


def test_audio_too_long(tokenizer):
    """Test audio longer than 30s gets truncated"""
    long_audio = torch.rand(480001)  # 30s + 1 sample
    audio_url = "<audio>tests/audio.wav</audio>"
    result = tokenizer.process_audio_no_url(long_audio, audio_url)

    # Should be truncated to 480000 samples
    assert result["input_audios"].shape[2] == 3000  # 480000 / 160 = 3000


def test_basic_question_extraction(tokenizer):
    """Test extraction of a simple question after audio tag"""
    audio_url = "<audio>tests/audio.wav</audio>"

    text = f"""<|im_start|>system
    You are a helpful assistant.<|im_end|>
    <|im_start|>user
    Audio 1:{audio_url}
    What is the capital of France?<|im_end|>
    <|im_start|>assistant"""

    # Tokenize the input
    audio_info = tokenizer.process_audio(audio_url)
    input_ids = tokenizer(text, return_tensors="pt", audio_info=audio_info).input_ids

    # Get special tokens
    special_tokens = {
        "</audio>": tokenizer.convert_tokens_to_ids("</audio>"),
        "<|im_end|>": tokenizer.convert_tokens_to_ids("<|im_end|>"),
    }

    # Call the function
    question_tokens, n_tokens, (start_idx, end_idx) = (
        tokenizer.get_number_of_question_tokens(input_ids, special_tokens)
    )

    # Verify results
    assert n_tokens > 0
    assert start_idx < end_idx
    decoded_question = tokenizer.decode(question_tokens)
    assert "What is the capital of France?" in decoded_question
    assert "</audio>" not in decoded_question
    assert "<|im_end|>" not in decoded_question


def test_multiple_im_end_tags(tokenizer):
    """Test that it correctly uses the last <|im_end|> tag"""
    # Input with multiple im_end tags (system message has one)
    audio_url = "<audio>tests/audio.wav</audio>"

    text = f"""<|im_start|>system
You are helpful.<|im_end|>
<|im_start|>user
{audio_url}
Question with multiple parts?
Second part?<|im_end|>
<|im_start|>assistant"""

    audio_info = tokenizer.process_audio(audio_url)
    input_ids = tokenizer(text, return_tensors="pt", audio_info=audio_info).input_ids

    special_tokens = {
        "</audio>": tokenizer.convert_tokens_to_ids("</audio>"),
        "<|im_end|>": tokenizer.convert_tokens_to_ids("<|im_end|>"),
    }

    question_tokens, n_tokens, _ = tokenizer.get_number_of_question_tokens(
        input_ids, special_tokens
    )
    decoded_question = tokenizer.decode(question_tokens)

    assert "Question with multiple parts?" in decoded_question
    assert "Second part?" in decoded_question
    assert "<|im_end|>" not in decoded_question


def test_empty_question(tokenizer):
    """Test case where there are no tokens between </audio> and <|im_end|>"""

    audio_url = "<audio>tests/audio.wav</audio>"
    text = f"""<|im_start|>system Instructions<|im_end|>
<|im_start|>user
{audio_url}
<|im_start|>assistant"""

    audio_info = tokenizer.process_audio(audio_url)
    input_ids = tokenizer(text, return_tensors="pt", audio_info=audio_info).input_ids

    special_tokens = {
        "</audio>": tokenizer.convert_tokens_to_ids("</audio>"),
        "<|im_end|>": tokenizer.convert_tokens_to_ids("<|im_end|>"),
    }

    question_tokens, n_tokens, _ = tokenizer.get_number_of_question_tokens(
        input_ids, special_tokens
    )

    assert n_tokens == 0
    assert len(question_tokens) == 0


def test_question_with_options(tokenizer):
    """Test extraction of a question with multiple choice options"""
    audio_url = "<audio>tests/audio.wav</audio>"
    text = f"""<|im_start|>System
	Instructions<|im_end|>
	<|im_start|>user
    {audio_url}
	Question: What is 2+2?
	Options: (A) 1 (B) 2 (C) 3 (D) 4<|im_end|>
	<|im_start|>assistant"""

    audio_info = tokenizer.process_audio(audio_url)
    input_ids = tokenizer(text, return_tensors="pt", audio_info=audio_info).input_ids
    special_tokens = {
        "</audio>": tokenizer.convert_tokens_to_ids("</audio>"),
        "<|im_end|>": tokenizer.convert_tokens_to_ids("<|im_end|>"),
    }

    question_tokens, n_tokens, _ = tokenizer.get_number_of_question_tokens(
        input_ids, special_tokens
    )
    decoded_question = tokenizer.decode(question_tokens)

    assert "Question: What is 2+2?" in decoded_question
    assert "Options: (A) 1 (B) 2 (C) 3 (D) 4" in decoded_question
    assert n_tokens > 10  # Should have several tokens
