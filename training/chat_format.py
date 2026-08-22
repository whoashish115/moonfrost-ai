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


def build_prompt_token_ids(tokenizer, conversation_history, new_message, special_tokens,
                           system_prompt=None, max_prompt_tokens=None):
    """Turns a conversation into the exact token-id sequence the model
    expects, ending with <|assistant|> so the model's next token starts the
    reply.

    conversation_history is a list of (role, text) turns, oldest first.

    If max_prompt_tokens is given and the conversation is longer than that,
    the OLDEST turns are dropped first -- the same policy sft_prepare.py
    used when packing training rows, so the model sees a shape it was
    trained on. The newest user message and the system prompt are always
    kept, even if that alone exceeds the budget (in which case the message
    itself is truncated from the left as a last resort).
    """
    def encode(text):
        return tokenizer.encode(text.strip()).ids

    system_token_ids = []
    if system_prompt and special_tokens.system is not None:
        system_token_ids = [special_tokens.system] + encode(system_prompt)

    # encode each history turn as a self-contained block so whole turns can be dropped
    history_blocks = []
    for role, text in conversation_history:
        if role == "user":
            history_blocks.append([special_tokens.user] + encode(text) + [special_tokens.assistant])
        else:
            history_blocks.append(encode(text) + [special_tokens.end_of_text])

    current_turn = [special_tokens.user] + encode(new_message) + [special_tokens.assistant]

    if max_prompt_tokens is not None:
        budget_for_history = max_prompt_tokens - len(system_token_ids) - len(current_turn)
        while history_blocks and sum(len(block) for block in history_blocks) > budget_for_history:
            history_blocks.pop(0)
        if budget_for_history < 0:
            # even the newest message alone doesn't fit: keep its tail, plus the closing
            # <|assistant|> so the model still knows it's being asked to reply
            room = max_prompt_tokens - len(system_token_ids) - 2
            if room > 0:
                current_turn = [special_tokens.user] + current_turn[1:-1][-room:] + [special_tokens.assistant]
            history_blocks = []

    token_ids = list(system_token_ids)
    for block in history_blocks:
        token_ids += block
    token_ids += current_turn
    return token_ids


REPLACEMENT_CHARACTER = "�"
