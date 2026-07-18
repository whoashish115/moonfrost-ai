"""
Turns a trained checkpoint into a real Hugging Face repository: safetensors
weights, a config, the architecture code, the tokenizer with a chat
template, and an honest model card.

    # build the repo folder locally and check it loads back
    python export_to_huggingface.py --ckpt ../checkpoints/chat_final.pt --out ../hf_repo --validate

    # then upload (you must have run `hf auth login` yourself first)
    python export_to_huggingface.py --ckpt ../checkpoints/chat_final.pt --out ../hf_repo \
        --repo-id yourname/moonfrost-777m-chat --push

This model's architecture is not one transformers ships, so the repo carries
its own code and is loaded with trust_remote_code=True:

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("yourname/moonfrost-777m-chat")
    model = AutoModelForCausalLM.from_pretrained("yourname/moonfrost-777m-chat", trust_remote_code=True)

What goes in the repo:
    model.safetensors            weights (bfloat16 by default: half the download)
    config.json                  architecture, with auto_map pointing at the code below
    configuration_moonfrost.py   the PretrainedConfig subclass
    modeling_moonfrost.py        the PreTrainedModel wrapper
    model_arch.py, arch_config.py   this project's actual architecture, copied verbatim
    tokenizer.json               your own byte-level BPE
    tokenizer_config.json        including a chat template matching how the model was trained
    README.md                    the model card

Note on generation: the wrapper exposes plain (uncached) forward passes, so
transformers' .generate() works but recomputes the prompt each step. For fast
cached generation use this project's own server.py and sample.py, which drive
the KV cache directly. The model card says so plainly.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import dataclasses
import json
import math
import shutil

import torch

import checkpoint_utils
from config import ModelConfig, count_active_parameters_per_token, count_total_parameters

HERE = os.path.dirname(os.path.abspath(__file__))

CHAT_TEMPLATE = (
    "{% if messages[0]['role'] == 'system' %}"
    "{{ '<|system|>' + messages[0]['content'] | trim }}"
    "{% set loop_messages = messages[1:] %}"
    "{% else %}{% set loop_messages = messages %}{% endif %}"
    "{% for message in loop_messages %}"
    "{% if message['role'] == 'user' %}"
    "{{ '<|user|>' + message['content'] | trim + '<|assistant|>' }}"
    "{% else %}"
    "{{ message['content'] | trim + '<|endoftext|>' }}"
    "{% endif %}{% endfor %}"
)
