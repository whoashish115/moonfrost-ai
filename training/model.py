"""
DeepSeek-V2/V3's actual architecture, implemented from scratch in plain
PyTorch: Multi-head Latent Attention (MLA) + DeepSeekMoE feed-forward layers,
RMSNorm, RoPE, weight tying, and a KV cache for fast autoregressive
generation. No pretrained weights, tokenizers, or HuggingFace `AutoModel`
classes are used anywhere in this file -- every matrix here starts from
random initialization and is trained by train.py from zero.

Variable naming in this file deliberately spells things out in full
(query_key_content_head_dim rather than qk_nope_head_dim, hidden_states
rather than x) so it can be read and understood without memorizing a table
of abbreviations first. The one place short names remain is inside tensor
shape comments like "(batch_size, num_heads, seq_len, head_dim)", which is
a standard, widely-recognized way to describe PyTorch tensor shapes.

--- Multi-head Latent Attention (MLA) ---
Standard multi-head attention caches a full-size key and value tensor per
head, for every past token, so they don't have to be recomputed during
generation. MLA instead down-projects the input into a small shared
"latent" vector (kv_compressed_latent_dim numbers) plus a small shared
decoupled-RoPE key (query_key_rotary_head_dim numbers), and only expands
back out to full per-head size when attention is actually computed. What
gets cached during generation is just that small latent + rotary key --
NOT full per-head keys/values -- which is the whole point: much less memory
used per cached token, at the cost of one extra small matrix multiply per
step.

Simplification, stated plainly: DeepSeek's production implementation avoids
even that per-step expansion by algebraically "absorbing" the expansion
matrices into the query/output projection matrices (since, for a query q
and a compressed latent c, q . (W_expand @ c) is mathematically the same as
(q @ W_expand) . c -- so you can pre-multiply W_expand into the query
projection instead of expanding the latent every step). That absorption is
a real speed optimization but easy to get subtly wrong, and skipping it
does not change the model's outputs at all -- it's a pure speed trick. This
implementation re-expands the cached latent every step instead, which is
simpler to verify correct and produces IDENTICAL results (proven by a
smoke test that compares cached vs. non-cached generation token-for-token),
just with a bit more compute per generation step. It's a documented future
optimization, not a correctness gap.

--- DeepSeekMoE (Mixture of Experts) ---
Most layers (all but num_initial_dense_layers) replace the single dense
feed-forward network with a bank of small "routed" expert feed-forward
networks plus one or more "shared" expert feed-forward networks that always
run. A per-token router (a linear layer + softmax) picks the top-k routed
experts for each token; shared experts run for every token regardless. An
auxiliary load-balancing loss (in the Switch-Transformer / DeepSeekMoE
style) is added to the language-modeling loss so the router learns to
spread tokens roughly evenly across experts instead of collapsing onto a
favorite few.

--- Multimodal note ---
This is a text-only backbone by design. The standard, proven way to add
vision later (LLaVA / Qwen-VL / DeepSeek-VL all do this) is NOT to redesign
this transformer -- it's to keep this text model as-is, add a separate
vision encoder + a small projection network that maps image patch
embeddings into this model's embedding_dimension, and splice those
projected embeddings into the input sequence at the reserved <|image|>
token positions (see tokenizer_train.py) before the first transformer
block. See MULTIMODAL.md for the full explanation.
"""
import math
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # avoid an MKL/libiomp5 conflict seen on some Anaconda+CUDA installs

from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig


class RMSNorm(nn.Module):
    """Root-Mean-Square Layer Normalization. Cheaper than standard
    LayerNorm (no mean-subtraction step) and equally stable for
    transformers at this scale and larger. Reference: Zhang & Sennrich,
    'Root Mean Square Layer Normalization', 2019 (arXiv:1910.07467)."""

    def __init__(self, dimension: int, epsilon: float = 1e-5):
        super().__init__()
        self.epsilon = epsilon
        self.weight = nn.Parameter(torch.ones(dimension))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        original_dtype = hidden_states.dtype
        # normalize in float32 for numerical stability, even if the surrounding
        # computation is running in bfloat16 under autocast
        hidden_states = hidden_states.float()
        root_mean_square = torch.rsqrt(hidden_states.pow(2).mean(dim=-1, keepdim=True) + self.epsilon)
        normalized = hidden_states * root_mean_square
        return normalized.to(original_dtype) * self.weight


def precompute_rotary_embeddings(rotary_dimension: int, max_sequence_length: int, theta_base: float,
                                  device, dtype=torch.float32):
    """Precomputes the cosine/sine tables used by Rotary Position Embedding
    (RoPE) for every position up to max_sequence_length, so they can just be
    sliced and reused instead of recomputed on every forward pass.
    Reference: Su et al., 'RoFormer: Enhanced Transformer with Rotary
    Position Embedding', 2021 (arXiv:2104.09864)."""
    # one frequency per pair of dimensions; lower frequencies rotate slowly (encode
    # coarse/long-range position), higher frequencies rotate quickly (fine-grained position)
    inverse_frequencies = 1.0 / (theta_base ** (torch.arange(0, rotary_dimension, 2, device=device).float() / rotary_dimension))
    position_indices = torch.arange(max_sequence_length, device=device).float()
    angles = torch.outer(position_indices, inverse_frequencies)  # shape: (max_sequence_length, rotary_dimension / 2)
    cosine_values = torch.cat([angles.cos(), angles.cos()], dim=-1)  # shape: (max_sequence_length, rotary_dimension)
    sine_values = torch.cat([angles.sin(), angles.sin()], dim=-1)
    return cosine_values.to(dtype), sine_values.to(dtype)


def rotate_half(input_tensor: torch.Tensor) -> torch.Tensor:
    """Splits the last dimension in half and swaps the two halves (with a
    sign flip on one), which is the specific rotation RoPE's formula needs.
    This is a helper used only by apply_rotary_embeddings() below."""
    first_half, second_half = input_tensor.chunk(2, dim=-1)
    return torch.cat([-second_half, first_half], dim=-1)
