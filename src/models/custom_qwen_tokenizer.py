import torch

class CustomQwenTokenizer(QwenTokenizer):
    def process_audio_no_url(self, audio):
        """
        Extension of `process_audio`. process one audio at a time and the
        input is the actual waveform instead of the audio url.

        """
        if np.prod(audio.shape) > 0:
            audios, audio_lens, audio_span_tokens = [], [], []

            L = (audio.shape[0] if audio.shape[0] <= 480000 else 480000)  # max_length < 30s
            mel_len = L // 160
            audio = pad_or_trim(audio.flatten())
            mel = log_mel_spectrogram(audio)
            audio_len_after_cnn = get_T_after_cnn(mel_len)
            audio_token_num = (audio_len_after_cnn - 2) // 2 + 1
            audio_len = [audio_len_after_cnn, audio_token_num]
            audios.append(mel)
            audio_lens.append(audio_len)
            audio_span_tokens.append(audio_token_num + 2)  # add audio bos eos
            input_audio_lengths = torch.IntTensor(audio_lens)
            input_audios = torch.stack(audios, dim=0)

            return {"input_audios": input_audios,
                    "input_audio_lengths": input_audio_lengths,
                    "audio_span_tokens": audio_span_tokens,
                    # FIXME: now we leave this url as none, but we might want to
                    # receive this as an argument afterwards
                    "audio_urls": None}
        else:
            return None

    def get_number_of_question_tokens(self, input_ids, special_tokens):
        """
        Given the model input with all its special tokens, this function returns
        only the tokens that are from the question we wish to mask.
        We consider a question token everything after </audio> and before
        the last <|im_end|> (system instructions has its own
        <|im_start|><|im_end|> tags.

        IMPORTANT: this assumes that we have no "history". otherwise we might have
        others <|im_start|> and <|im_end|> tags

        Parameters
        ---
        input_ids : torch.Tensor
        special_tokens :  dict

        Return
        ---
        question_tokens : numpy.array
            text tokens from the questions
        n_text_tokens: int
            number of question tokens
        (end_audio+1, im_end) : tuple
            interval in which the question is localized

        Input Example
        ---

        <|im_start|>system
        You are a helpful assistant. You will receive multiple choice questions and
        your answer should be only the letter of the correct choice (e.g., A, B, C,
        D). Do *not* include any additional te xt, explanations, or reasoning in your response. <|im_end|>
        <|im_start|>user
        Audio 1:<audio>/scratch/gv2167/datasets/sdd/audio/90/43990.2min.mp3</audio>
        Question: What rhythm pattern do the digital drums primarily follow in this
        pop music piece?
        Options: (A) Four on the floor. (B) Off-beat syncopation. (C) Scat singing.
        (D) E-guitar playing a simple melody.
        The correct answer is: B
        Question: Which instrument initiates the piece?
        Options: (A) Rueful tune (B) Synthesizer (C) Acoustic guitar (D) Vocals
        The correct answer is: <|im_end|>
        <|im_start|>assistant
        """

        tokens = input_ids.clone().detach().squeeze(0)
        end_audio = torch.where(tokens == special_tokens["</audio>"])[0][0]
        im_end = torch.where(tokens == special_tokens["<|im_end|>"])[0][-1]
        # here we use end_audio+1 to avoid including the tag itself
        # [end_audio+1:im_end] should not include the last tag by default
        text_tokens = tokens[end_audio+1:im_end]
        n_text_tokens = len(text_tokens)

        return text_tokens, n_text_tokens, (end_audio+1, im_end)
