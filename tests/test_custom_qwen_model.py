import pytest
import torch
import numpy as np

from src.models.custom_qwen_model import CustomQwenModel

# --- Tests for CustomQwenModel ---
def test_chat_method_with_decoded_response(model, tokenizer):
    """
    Tests the CustomQwenModel.chat method when `decode_response` is True.
    """
    # --- Act ---
    print("calling chat")

    audio_url = "<audio>tests/audio.wav</audio>"
    query = f"""<|im_start|>System
	Instructions<|im_end|>
	<|im_start|>user
    {audio_url}
	Question: What is 2+2?
	Options: (A) 1 (B) 2 (C) 3 (D) 4<|im_end|>
	<|im_start|>assistant"""

    outputs, response, history = CustomQwenModel.chat(
        model,
        tokenizer,
        query=query,
        history=[],
        decode_response=True
    )

        # --- Assert ---
        # mock_model.generate.assert_called_once()
        # assert torch.equal(outputs, torch.tensor([[2, 2, 3, 6, 7]]))
        # assert response == "This is a decoded response."
        # assert history == [("Hello", "This is a decoded response.")]


# def test_chat_method_without_decoded_response(mock_model):
#     """
#     Tests the CustomQwenModel.chat method when `decode_response` is False.
#     """
#     # --- Arrange ---
#     with patch('custom_qwen.make_context') as mock_make_context, \
#          patch('custom_qwen.get_stop_words_ids') as mock_get_stop_words:

#         mock_make_context.return_value = ("raw_text", [1, 2, 3], "audio_info")
#         mock_get_stop_words.return_value = [[4], [5]]
#         mock_model.generate.return_value = torch.tensor([[1, 2, 3, 6, 7]])
#         mock_tokenizer = MagicMock()

#         # --- Act ---
#         outputs, response, history = CustomQwenModel.chat(
#             mock_model,
#             mock_tokenizer,
#             query="Hello",
#             history=[],
#             decode_response=False
#         )

#         # --- Assert ---
#         mock_model.generate.assert_called_once()
#         assert torch.equal(outputs, torch.tensor([[1, 2, 3, 6, 7]]))
#         assert response is None
#         assert history == []
