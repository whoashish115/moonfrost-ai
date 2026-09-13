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


class IncrementalTextDecoder:
    """Turns a stream of token ids into a stream of APPEND-ONLY text deltas.

    Decoding tokens one at a time and concatenating the pieces is wrong for
    a byte-level BPE tokenizer: a single character can span several tokens.
    An emoji, for instance, is four UTF-8 bytes that the tokenizer may split
    across two or three tokens, and decoding any prefix of them yields the
    replacement character U+FFFD rather than part of the emoji. So the whole
    sequence is decoded each step and only the part that grew is emitted.

    That alone is still not enough. When the final byte of a character
    arrives, the full decode does not merely get longer -- the U+FFFD that
    was standing in for the incomplete character is REPLACED by the real
    character. A naive "emit whatever is new" would already have sent the
    U+FFFD, leaving a permanent black diamond in the middle of the reply.

    The fix is to hold back a trailing run of replacement characters until
    it resolves. Everything this class emits is therefore a true suffix
    extension: callers can append deltas without ever needing to rewrite.
    Call flush() at the end to release anything still held back.

    Re-decoding the whole sequence each step is O(n^2) in principle, but n
    is at most a few thousand tokens and the tokenizer decodes in
    microseconds -- the total across an entire reply stays far below the
    cost of a single model forward pass.
    """

    def __init__(self, tokenizer, skip_special_tokens=True):
        self.tokenizer = tokenizer
        self.skip_special_tokens = skip_special_tokens
        self.token_ids = []
        self.text = ""      # everything decoded so far, including any unresolved characters
        self.emitted = ""   # the prefix of self.text already returned by push()

    def push(self, token_id):
        """Appends one token and returns the new text to display, which may
        be "" while a multi-token character is still incomplete."""
        self.token_ids.append(int(token_id))
        self.text = self.tokenizer.decode(self.token_ids, skip_special_tokens=self.skip_special_tokens)

        stable = self.text
        while stable.endswith(REPLACEMENT_CHARACTER):
            stable = stable[:-len(REPLACEMENT_CHARACTER)]

        if not stable.startswith(self.emitted):
            # should not happen now that unresolved characters are held back, but never
            # emit a correction that would duplicate text the caller already appended
            return ""
        delta = stable[len(self.emitted):]
        self.emitted = stable
        return delta

    def flush(self):
        """Releases any text still held back (an unterminated multi-token
        character at the very end of a reply). Returns "" if there is none."""
        delta = self.text[len(self.emitted):] if self.text.startswith(self.emitted) else ""
        self.emitted = self.text
        return delta

    def push_all(self, token_ids):
        pieces = [self.push(token_id) for token_id in token_ids]
        pieces.append(self.flush())
        return "".join(pieces)


# Measured on this project's own sft_last.pt, on the prompt "Name two colours.":
#   repetition_penalty 1.10 -> "A font with a font and a font with a font..." (locked in a loop)
#   repetition_penalty 1.25 -> varied, on-topic, no loop
#   temperature 0.0 + 1.15  -> "Two colours are colour, blue and yellow." and stopped after 9 tokens
# The old default was 1.3 applied to the ENTIRE sequence including the prompt and the special
# tokens, which is far too aggressive -- it blocked reuse of the question's own words and made
# <|endoftext|> progressively harder to emit, so replies rambled. 1.15 over a 256-token window
# of only the model's own output is enough to break loops on an undertrained model without
# either of those side effects. Lower it towards 1.05 once the base model is better trained.
DEFAULT_SAMPLING = {
    "temperature": 0.7,
    "top_k": 40,
    "top_p": 0.92,
    "min_p": 0.05,
    "repetition_penalty": 1.15,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
    "repetition_window": 256,
    "max_new_tokens": 400,
}
