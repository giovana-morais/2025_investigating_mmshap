import pytest
import yaml

from src.models.custom_qwen_model import CustomQwenModel
from src.models.custom_qwen_tokenizer import CustomQwenTokenizer

@pytest.fixture
def tokenizer():
    with open("config.yml", "r") as f:
        config = yaml.safe_load(f)

    vocab_file = config["qwenaudio"]["vocab_file"]
    tokenizer = CustomQwenTokenizer(vocab_file).from_pretrained(
        "Qwen/Qwen-Audio-Chat", trust_remote_code=True
    )
    return tokenizer


@pytest.fixture
def model():
    with open("config.yml", "r") as f:
        config = yaml.safe_load(f)

    config_file = config["qwenaudio"]["chat_config"]

    model = CustomQwenModel.from_pretrained(
            "Qwen/Qwen-Audio-Chat",
            trust_remote_code=True
            ).eval()

    model.device = "cpu"

    return model
