import pytest
import yaml

from src.models.custom_qwen_model import CustomQwenModel
from src.models.custom_qwen_tokenizer import CustomQwenTokenizer
from src.models.Qwen_Audio.configuration_qwen import QWenConfig

@pytest.fixture
def tokenizer():
    with open("config.yml", "r") as f:
        config = yaml.safe_load(f)

    vocab_file = config["qwenaudio"]["vocab_file"]
    custom_tokenizer = CustomQwenTokenizer(vocab_file).from_pretrained(
        "Qwen/Qwen-Audio-Chat", trust_remote_code=True
    )

    return custom_tokenizer

def model():
    with open("config.yml", "r") as f:
        config = yaml.safe_load(f)

    qwen_config = QWenConfig.from_pretrained(config["qwenaudio"]["model_config"])
    model = CustomQwenModel(qwen_config).from_pretrained(
            "Qwen/Qwen-Audio-Chat",
            trust_remote_code=True
            ).eval()

    return model
