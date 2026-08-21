"""
The chat prompt format and the streaming text decoder, in one place so the
the one-shot sampler (sample.py) and the web server
(server.py) all speak to the model in exactly the format sft_prepare.py
trained it on. Before this module existed each of them had its own copy,
and they had drifted apart.

The format, which must match sft_prepare.py exactly:

    <|user|>{message}<|assistant|>{reply}<|endoftext|><|user|>...

Note there is no newline or space between the special tokens and the text --
sft_prepare.py calls tokenizer.encode(text.strip()), so any whitespace added
here would be whitespace the model never saw during training.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


class ChatSpecialTokens:
    """The special token ids the chat format needs, looked up once.

    Raises a clear error naming the missing token rather than silently
    passing None into the prompt builder, which is what used to happen when
    a tokenizer trained without chat tokens was pointed at the server."""

    REQUIRED = ("<|user|>", "<|assistant|>", "<|endoftext|>")
    OPTIONAL = ("<|system|>", "<|image|>", "<|pad|>")

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        missing = [name for name in self.REQUIRED if tokenizer.token_to_id(name) is None]
        if missing:
            raise ValueError(
                f"tokenizer is missing required chat special tokens: {', '.join(missing)}. "
                f"Retrain it with tokenizer_train.py, which registers them."
            )
        self.user = tokenizer.token_to_id("<|user|>")
        self.assistant = tokenizer.token_to_id("<|assistant|>")
        self.end_of_text = tokenizer.token_to_id("<|endoftext|>")
        self.system = tokenizer.token_to_id("<|system|>")

        # every special token id, used to keep the repetition penalty away from
        # them -- penalizing <|endoftext|> makes the model progressively unable to stop talking
        self.all_special_ids = set()
        for name in self.REQUIRED + self.OPTIONAL:
            token_id = tokenizer.token_to_id(name)
            if token_id is not None:
                self.all_special_ids.add(token_id)

    @property
    def stop_ids(self):
        """Token ids that end an assistant turn. <|user|> is included
        because an undertrained model sometimes starts writing the user's
        next message instead of stopping -- cutting there is much better
        than showing the hallucinated dialogue."""
        return {self.end_of_text, self.user}
