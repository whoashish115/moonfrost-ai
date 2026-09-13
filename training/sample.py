"""
Load a checkpoint and generate text from a single prompt, from the command
line -- useful for quick one-off checks without starting the web interface
(server.py) for a full conversation.

Two modes:
  - completion (default): raw text continuation, for base/pretrained checkpoints
  - --chat: wraps the prompt in the <|user|>/<|assistant|> template and
    stops at <|endoftext|>, for checkpoints produced by sft_train.py

Usage:
    python sample.py --ckpt ../checkpoints/small_best.pt \
        --prompt "The history of the internet" --max-new-tokens 200

    python sample.py --ckpt ../checkpoints/sft_last.pt --chat \
        --prompt "What's a good way to learn a new language?"

    # side-by-side comparison of several settings on the same prompt
    python sample.py --ckpt ../checkpoints/sft_last.pt --chat --prompt "Hi" --sweep
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # avoid an MKL/libiomp5 conflict seen on some Anaconda+CUDA installs

import argparse
import time

import torch
from tokenizers import Tokenizer

import checkpoint_utils
from chat_format import DEFAULT_SAMPLING, ChatSpecialTokens, IncrementalTextDecoder, build_prompt_token_ids
from config import ModelConfig, count_total_parameters
from model import GPT

# settings compared by --sweep, chosen to show the two failure modes of a small model:
# too-low temperature loops, too-high temperature turns to noise
SWEEP_SETTINGS = [
    ("greedy",       dict(temperature=0.0, top_k=0, top_p=1.0, min_p=0.0, repetition_penalty=1.0)),
    ("conservative", dict(temperature=0.5, top_k=40, top_p=0.9, min_p=0.05, repetition_penalty=1.1)),
    ("balanced",     dict(temperature=0.8, top_k=40, top_p=0.92, min_p=0.05, repetition_penalty=1.1)),
    ("creative",     dict(temperature=1.1, top_k=80, top_p=0.95, min_p=0.02, repetition_penalty=1.05)),
    ("no penalty",   dict(temperature=0.8, top_k=40, top_p=0.92, min_p=0.05, repetition_penalty=1.0)),
]


def stream_once(model, tokenizer, special_tokens, prompt_token_ids, device, settings,
                max_new_tokens, seed, echo=True):
    """Generates one continuation and returns (text, token_count, seconds).
    Only newly generated tokens reach the decoder, so the prompt is never
    echoed back."""
    prompt_tensor = torch.tensor([prompt_token_ids], dtype=torch.long, device=device)
    decoder = IncrementalTextDecoder(tokenizer, skip_special_tokens=True)
    token_count, started = 0, time.time()

    for token_id in model.generate_stream(
        prompt_tensor, max_new_tokens=max_new_tokens,
        temperature=settings["temperature"],
        top_k=settings["top_k"] or None,
        top_p=settings["top_p"] if settings["top_p"] < 1.0 else None,
        min_p=settings["min_p"] or None,
        stop_token_ids=special_tokens.stop_ids if special_tokens else (),
        repetition_penalty=settings["repetition_penalty"],
        protected_token_ids=special_tokens.all_special_ids if special_tokens else (),
        seed=seed,
    ):
        if special_tokens and token_id in special_tokens.stop_ids:
            break
        token_count += 1
        delta = decoder.push(token_id)
        if delta and echo:
            print(delta, end="", flush=True)
    trailing = decoder.flush()  # release a character still mid-encoding at the cut-off
    if trailing and echo:
        print(trailing, end="", flush=True)
    return decoder.text, token_count, max(time.time() - started, 1e-6)


def main():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--ckpt", type=str, required=True)
    argument_parser.add_argument("--tokenizer-dir", type=str, default="../tokenizer")
    argument_parser.add_argument("--prompt", type=str, default="")
    argument_parser.add_argument("--chat", action="store_true",
                                  help="use the <|user|>/<|assistant|> chat template (SFT checkpoints)")
    argument_parser.add_argument("--system", type=str, default=None)
    argument_parser.add_argument("--max-new-tokens", type=int, default=200)
    argument_parser.add_argument("--temperature", type=float, default=DEFAULT_SAMPLING["temperature"])
    argument_parser.add_argument("--top-k", type=int, default=DEFAULT_SAMPLING["top_k"])
    argument_parser.add_argument("--top-p", type=float, default=DEFAULT_SAMPLING["top_p"])
    argument_parser.add_argument("--min-p", type=float, default=DEFAULT_SAMPLING["min_p"])
    argument_parser.add_argument("--repetition-penalty", type=float, default=DEFAULT_SAMPLING["repetition_penalty"])
    argument_parser.add_argument("--num-samples", type=int, default=1)
    argument_parser.add_argument("--seed", type=int, default=None)
    argument_parser.add_argument("--sweep", action="store_true",
                                  help="generate the same prompt under several sampling settings and print them together")
    argument_parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = argument_parser.parse_args()

    tokenizer = Tokenizer.from_file(os.path.join(args.tokenizer_dir, "tokenizer.json"))
    special_tokens = ChatSpecialTokens(tokenizer)

    model, model_config, info = checkpoint_utils.load_for_inference(args.ckpt, args.device, ModelConfig, GPT)
    model.enable_inference_absorption()
    print(f"{os.path.basename(args.ckpt)}: {count_total_parameters(model_config)/1e6:.1f}M parameters, "
          f"step {info['step']}, best val loss {info['best_val']:.4f}, context {model_config.max_sequence_length}")

    if args.chat:
        prompt_token_ids = build_prompt_token_ids(
            tokenizer, [], args.prompt, special_tokens, system_prompt=args.system,
            max_prompt_tokens=max(16, model_config.max_sequence_length - args.max_new_tokens),
        )
        stop_tokens = special_tokens
    else:
        prompt_token_ids = tokenizer.encode(args.prompt).ids if args.prompt else [special_tokens.end_of_text]
        stop_tokens = None
        print(f"prompt: {args.prompt!r}")

    if args.sweep:
        for name, settings in SWEEP_SETTINGS:
            print(f"\n--- {name}: " + ", ".join(f"{k}={v}" for k, v in settings.items()) + " ---")
            text, token_count, seconds = stream_once(
                model, tokenizer, stop_tokens, prompt_token_ids, args.device,
                settings, args.max_new_tokens, args.seed, echo=False,
            )
            print(text.strip() or "(empty)")
            print(f"[{token_count} tokens, {token_count/seconds:.1f} tok/s]")
        return

    settings = {"temperature": args.temperature, "top_k": args.top_k, "top_p": args.top_p,
                "min_p": args.min_p, "repetition_penalty": args.repetition_penalty}

    for sample_index in range(args.num_samples):
        print(f"\n--- sample {sample_index + 1} ---")
        if not args.chat and args.prompt:
            print(args.prompt, end="", flush=True)
        _, token_count, seconds = stream_once(
            model, tokenizer, stop_tokens, prompt_token_ids, args.device, settings,
            args.max_new_tokens, None if args.seed is None else args.seed + sample_index,
        )
        print(f"\n[{token_count} tokens, {token_count/seconds:.1f} tok/s]")


if __name__ == "__main__":
    main()
