import pytest
import torch
import numpy as np

from src.models.custom_qwen_model import CustomQwenModel

# --- Tests for CustomQwen ---
def test_chat_method_with_decoded_response(model, tokenizer):
    """
    Tests the CustomQwen.chat method when `decode_response` is True.
    """
    # --- Arrange ---
    # We use `patch` as a context manager, which is common in pytest
    # with patch('custom_qwen.make_context') as mock_make_context, \
    #      patch('custom_qwen.get_stop_words_ids') as mock_get_stop_words, \
    #      patch('custom_qwen.decode_tokens') as mock_decode:

    #     # Configure mocks
    #     mock_make_context.return_value = ("raw_text", [1, 2, 3], "audio_info")
    #     mock_get_stop_words.return_value = [[4], [5]]
    #     mock_model.generate.return_value = torch.tensor([[1, 2, 3, 6, 7]])
    #     mock_decode.return_value = "This is a decoded response."
    #     mock_tokenizer = MagicMock() # Mock tokenizer for this specific test

    # --- Act ---
    print("calling chat")
    outputs, response, history = CustomQwen.chat(
        model,
        tokenizer,
        query="Hello",
        history=[],
        decode_response=True
    )

    print(outputs)
    print(response)

        # --- Assert ---
        # mock_model.generate.assert_called_once()
        # assert torch.equal(outputs, torch.tensor([[2, 2, 3, 6, 7]]))
        # assert response == "This is a decoded response."
        # assert history == [("Hello", "This is a decoded response.")]


# def test_chat_method_without_decoded_response(mock_model):
#     """
#     Tests the CustomQwen.chat method when `decode_response` is False.
#     """
#     # --- Arrange ---
#     with patch('custom_qwen.make_context') as mock_make_context, \
#          patch('custom_qwen.get_stop_words_ids') as mock_get_stop_words:

#         mock_make_context.return_value = ("raw_text", [1, 2, 3], "audio_info")
#         mock_get_stop_words.return_value = [[4], [5]]
#         mock_model.generate.return_value = torch.tensor([[1, 2, 3, 6, 7]])
#         mock_tokenizer = MagicMock()

#         # --- Act ---
#         outputs, response, history = CustomQwen.chat(
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
