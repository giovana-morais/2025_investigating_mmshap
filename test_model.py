import yaml
from transformers import AutoTokenizer, AutoModelForCausalLM, PretrainedConfig

from src.models.custom_qwen_model import CustomQwenModel
from src.models.custom_qwen_tokenizer import CustomQwenTokenizer
from src.models.Qwen_Audio.configuration_qwen import QWenConfig


with open("config.yml", "r") as f:
    config = yaml.safe_load(f)

vocab_file = config["qwenaudio"]["vocab_file"]
custom_tokenizer = CustomQwenTokenizer(vocab_file).from_pretrained(
    "Qwen/Qwen-Audio-Chat", trust_remote_code=True
)

qwen_config = QWenConfig.from_pretrained(config["qwenaudio"]["model_config"])

custom_model = CustomQwenModel(qwen_config).from_pretrained(
        "Qwen/Qwen-Audio-Chat",
        map_device="cuda",
        trust_remote_code=True
        ).eval()

audio_url = "<audio>tests/audio.wav</audio>"

text = f"""<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Audio 1:{audio_url}
What is the capital of France?<|im_end|>
<|im_start|>assistant"""

outputs = custom_model.chat(tokenizer, text, history=None, decode_response=True)
print(outputs)
