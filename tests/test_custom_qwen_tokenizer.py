import pytest
import numpy as np
import torch

from custom_qwen_tokenizer import CustomQwenTokenizer
from src.models.qwenaudio.audio import *

# Mock the audio processing functions
def pad_or_trim(audio):
    """Mock function for pad_or_trim"""
    return audio[:480000]  # Just trim to max length

def log_mel_spectrogram(audio):
    """Mock function for log_mel_spectrogram"""
    # Return a dummy mel spectrogram of shape (n_mels, time)
    return torch.randn(80, len(audio) // 160)  # Assuming 80 mel bands

def get_T_after_cnn(mel_len):
    """Mock function for get_T_after_cnn"""
    return mel_len - (mel_len % 4)  # Simple downsampling mock

# Fixture for the tokenizer
@pytest.fixture
def tokenizer():
    # Since QwenTokenizer requires initialization, we'll mock the parent class
    class MockQwenTokenizer:
        pass

    CustomQwenTokenizer.__bases__ = (MockQwenTokenizer,)
    return CustomQwenTokenizer()

# Test cases
def test_empty_audio(tokenizer):
    """Test case A: Empty audio should return None"""
    empty_audio = np.array([])
    result = tokenizer.process_audio_no_url(empty_audio)
    assert result is None

def test_mono_audio(tokenizer):
    """Test case B: Mono audio (1 channel)"""
    # Create mono audio (shape: [samples])
    mono_audio = np.random.rand(16000)  # 1 second of audio at 16kHz
    result = tokenizer.process_audio_no_url(mono_audio)

    assert isinstance(result, dict)
    assert "input_audios" in result
    assert "input_audio_lengths" in result
    assert "audio_span_tokens" in result
    assert "audio_urls" in result

    # Check tensor shapes
    assert result["input_audios"].shape == (1, 80, 100)  # (batch, n_mels, time)
    assert result["input_audio_lengths"].shape == (1, 2)  # (batch, [audio_len, token_num])

def test_stereo_audio(tokenizer):
    """Test case C: Stereo audio (2 channels)"""
    # Create stereo audio (shape: [2, samples])
    stereo_audio = np.random.rand(2, 48000)  # 2 channels, 3 seconds at 16kHz
    result = tokenizer.process_audio_no_url(stereo_audio)

    assert isinstance(result, dict)
    # Should flatten stereo to mono, so same checks as mono apply
    assert result["input_audios"].shape == (1, 80, 300)  # Longer audio
    assert result["input_audio_lengths"].shape == (1, 2)

def test_audio_too_long(tokenizer):
    """Test audio longer than 30s gets truncated"""
    long_audio = np.random.rand(480001)  # 30s + 1 sample
    result = tokenizer.process_audio_no_url(long_audio)

    # Should be truncated to 480000 samples
    assert result["input_audios"].shape[2] == 3000  # 480000 / 160 = 3000

def test_get_question_tokens(tokenizer):
    """Test text processing extracts correct question tokens and interval"""
    # Example input from docstring
    example_text = """<|im_start|>system
You are a helpful assistant...<|im_end|>
<|im_start|>user
Audio 1:<audio>/scratch/audio.mp3</audio>
Question: What rhythm pattern do the digital drums primarily follow?
Options: (A) Four on the floor. (B) Off-beat syncopation.
The correct answer is: B
Question: Which instrument initiates the piece?
Options: (A) Rueful tune (B) Synthesizer
The correct answer is: <|im_end|>
<|im_start|>assistant"""

	tokenizer = QwenTokenizer.from_pretrained("Qwen/Qwen-audio")
	input_ids = real_tokenizer(example_text, return_tensors="pt").input_ids
	special_tokens = {k: real_tokenizer.convert_tokens_to_ids(v)
					 for k, v in real_tokenizer.special_tokens_map.items()}

    # Run the method
    question_tokens, n_tokens, (start_idx, end_idx) = tokenizer.get_number_of_question_tokens(
        input_ids, special_tokens
    )

    # Verify outputs
    assert n_tokens == 14  # 7 tokens per question
    assert start_idx == 9   # After </audio> (position 8) + 1
    assert end_idx == 23    # Last <|im_end|> position
    assert torch.all(question_tokens == torch.tensor([
        103, 104, 105, 106, 107, 108, 109,  # Question 1
        110, 111, 112, 113, 114, 115, 116   # Question 2
    ]))

def test_no_question_tokens(tokenizer):
    """Test case where no question exists between tags"""
    special_tokens = {"</audio>": 102, "<|im_end|>": 101}
    input_ids = torch.tensor([100, 102, 101]).unsqueeze(0)

    question_tokens, n_tokens, _ = tokenizer.get_number_of_question_tokens(
        input_ids, special_tokens
    )

    assert n_tokens == 0
    assert len(question_tokens) == 0
