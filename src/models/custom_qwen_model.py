import torch

from models.Qwen_Audio.modeling_qwen import *

class CustomQwenModel(QWenLMHeadModel):
    def __init__(self, config):
        super().__init__(config)
        print("SEARCH ORDER:\n",self.__mro__)

    # overwriting the chat method to return `outputs`
    def chat(
        self,
        tokenizer: PreTrainedTokenizer,
        query: str,
        history: Optional[HistoryType] = None,
        system: str = "You are a helpful assistant.",
        append_history: bool = True,
        stop_words_ids: Optional[List[List[int]]] = None,
        generation_config: Optional[GenerationConfig] = None,
        decode_response: Optional[str] = False,
        **kwargs,
    ) -> Tuple[str, HistoryType]:
        generation_config = generation_config if generation_config is not None else self.generation_config

        assert generation_config.chat_format == 'chatml', _ERROR_BAD_CHAT_FORMAT
        if history is None:
            history = []
        else:
            # make a copy of the user's input such that is is left untouched
            history = copy.deepcopy(history)

        if stop_words_ids is None:
            stop_words_ids = []

        max_window_size = kwargs.get('max_window_size', None)
        if max_window_size is None:
            max_window_size = generation_config.max_window_size

        raw_text, context_tokens, audio_info = make_context(
            tokenizer,
            query,
            history=history,
            system=system,
            max_window_size=max_window_size,
            chat_format=generation_config.chat_format,
        )

        stop_words_ids.extend(get_stop_words_ids(
            generation_config.chat_format, tokenizer
        ))
        input_ids = torch.tensor([context_tokens]).to(self.device)
        kwargs['audio_info'] = audio_info

        print("CustomQwen super().generate")
        outputs = super().generate(
                    input_ids,
                    stop_words_ids=stop_words_ids,
                    return_dict_in_generate=True,
                    output_scores=True,
                    output_logits=True,
                    generation_config=generation_config,
                    **kwargs,
                )

        # we just want to decode the response for debugging purposes
        if decode_response:
            response = decode_tokens(
                outputs[0],
                tokenizer,
                raw_text_len=len(raw_text),
                context_length=len(context_tokens),
                chat_format=generation_config.chat_format,
                verbose=False,
                errors='replace',
                audio_info=audio_info
            )
            # as history is a copy of the user inputs,
            # we can always return the new turn to the user.
            # separating input history and output history also enables the user
            # to implement more complex history management
            history.append((query, response))
        else:
            response = None

        return outputs, response, history
