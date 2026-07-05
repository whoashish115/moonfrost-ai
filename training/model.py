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


def apply_rotary_embeddings(input_tensor: torch.Tensor, cosine_values: torch.Tensor,
                             sine_values: torch.Tensor) -> torch.Tensor:
    """Rotates input_tensor's vectors by an angle proportional to their
    position, using the precomputed tables from precompute_rotary_embeddings().
    input_tensor shape: (batch_size, num_heads_or_1, seq_len, rotary_dimension);
    cosine_values/sine_values shape: (seq_len, rotary_dimension).

    The rotation is computed in float32 for precision and then cast back to
    whatever dtype came in. Returning the input's dtype matters: the tables
    are float32, so a bfloat16 query would otherwise be promoted to float32
    here while the values stay bfloat16, and scaled_dot_product_attention
    rejects mismatched dtypes. Under autocast that promotion was invisible
    (autocast re-cast everything anyway), but it made the model unusable in
    pure bfloat16 -- which is exactly how it runs on a small GPU and how it
    is published to Hugging Face."""
    original_dtype = input_tensor.dtype
    cosine_values = cosine_values[None, None, :, :].float()  # add batch and head dimensions for broadcasting
    sine_values = sine_values[None, None, :, :].float()
    input_in_float32 = input_tensor.float()
    rotated = (input_in_float32 * cosine_values) + (rotate_half(input_in_float32) * sine_values)
    return rotated.to(original_dtype)


class MultiHeadLatentAttention(nn.Module):
    """DeepSeek-V2/V3's Multi-head Latent Attention. See the module-level
    docstring at the top of this file for the full explanation."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.content_head_dim = config.query_key_content_head_dim   # per-head "what" part of query/key (not positional)
        self.rotary_head_dim = config.query_key_rotary_head_dim     # per-head "where" part of query/key (positional, via RoPE)
        self.value_head_dim = config.value_head_dim
        self.query_key_head_dim = config.query_key_content_head_dim + config.query_key_rotary_head_dim  # combined size used for the attention dot product
        self.kv_compressed_latent_dim = config.kv_compressed_latent_dim
        self.query_compressed_latent_dim = config.query_compressed_latent_dim
        self.dropout_probability = config.dropout_probability

        if self.query_compressed_latent_dim > 0:
            # The query path is compressed as well. Queries are never cached, so this buys
            # no inference saving the way the KV path does; it is here because it cuts
            # activation memory during training, which is what DeepSeek uses it for.
            self.query_down_projection = nn.Linear(config.embedding_dimension, self.query_compressed_latent_dim, bias=False)
            self.query_down_projection_norm = RMSNorm(self.query_compressed_latent_dim, config.normalization_epsilon)
            self.query_up_projection = nn.Linear(self.query_compressed_latent_dim, self.num_heads * self.query_key_head_dim, bias=False)
        else:
            self.query_projection = nn.Linear(config.embedding_dimension, self.num_heads * self.query_key_head_dim, bias=False)

        # one matrix produces BOTH the compressed key/value latent AND the shared decoupled
        # rotary key in a single matrix multiply, then they're split apart afterward
        self.kv_down_projection = nn.Linear(
            config.embedding_dimension, self.kv_compressed_latent_dim + self.rotary_head_dim, bias=False
        )
        self.kv_down_projection_norm = RMSNorm(self.kv_compressed_latent_dim, config.normalization_epsilon)
        self.kv_up_projection = nn.Linear(
            self.kv_compressed_latent_dim, self.num_heads * (self.content_head_dim + self.value_head_dim), bias=False
        )

        self.output_projection = nn.Linear(self.num_heads * self.value_head_dim, config.embedding_dimension, bias=False)

        # inference-only derived tensors, filled in by absorb_matrices() below. Declared here
        # so every code path can test self.is_absorbed instead of getattr(..., False).
        self.is_absorbed = False
        self.W_UK = None
        self.W_O_absorbed = None
        self.scale_correction = 1.0

    def absorb_matrices(self):
        """Absorbs the latent up-projection matrices into the query and output
        projections. This is the crucial DeepSeek inference optimization that
        allows caching only the small latent vector instead of expanding it.

        The absorbed tensors are derived from (not replacements for) the
        module's real parameters, so they must be thrown away by
        release_absorbed_matrices() before any further training -- otherwise
        the model would keep attending through a stale copy of weights the
        optimizer has since updated. Calling .train() does that
        automatically; see TransformerBlock/GPT below."""
        if self.is_absorbed:
            return

        # Split the up-projection weight into Keys (UK) and Values (UV)
        W_U = self.kv_up_projection.weight.view(self.num_heads, self.content_head_dim + self.value_head_dim, self.kv_compressed_latent_dim)
        self.W_UK = W_U[:, :self.content_head_dim, :].clone()  # (H, C, L)
        W_UV = W_U[:, self.content_head_dim:, :]               # (H, V, L)

        # Absorb W_UV into the output projection
        E = self.output_projection.weight.size(0)
        W_O = self.output_projection.weight.view(E, self.num_heads, self.value_head_dim)
        self.W_O_absorbed = torch.einsum('ehv,hvl->hle', W_O, W_UV).clone()

        # SDPA scales by 1/sqrt(query width), and in absorbed mode the query is L + R wide
        # while the attention should be scaled for C + R. Folding the ratio into the query
        # once here is cheaper than rescaling the scores on every step.
        self.scale_correction = math.sqrt(self.kv_compressed_latent_dim + self.rotary_head_dim) / math.sqrt(self.content_head_dim + self.rotary_head_dim)
        self.is_absorbed = True

    def release_absorbed_matrices(self):
        """Drops the cached absorbed tensors and returns to the standard
        (trainable) attention path, freeing their memory."""
        if not self.is_absorbed:
            return
        self.W_UK = None
        self.W_O_absorbed = None
        self.is_absorbed = False

    def train(self, mode: bool = True):
        # absorbed tensors are snapshots of the weights; going back into training mode must
        # invalidate them so gradients flow through the real parameters again
        if mode:
            self.release_absorbed_matrices()
        return super().train(mode)

    def forward(self, hidden_states, rotary_cosine, rotary_sine, key_value_cache: Optional[List[torch.Tensor]] = None):
        batch_size, seq_len, embedding_dim = hidden_states.shape

        # --- queries: down-project (optionally), then up-project to full per-head size ---
        if self.query_compressed_latent_dim > 0:
            compressed_query = self.query_down_projection(hidden_states)
            query = self.query_up_projection(self.query_down_projection_norm(compressed_query))
        else:
            query = self.query_projection(hidden_states)
        query = query.view(batch_size, seq_len, self.num_heads, self.query_key_head_dim).transpose(1, 2)  # (batch, heads, seq, head_dim)
        query_content, query_rotary = torch.split(query, [self.content_head_dim, self.rotary_head_dim], dim=-1)
        query_rotary = apply_rotary_embeddings(query_rotary, rotary_cosine, rotary_sine)

        # --- keys/values: down-project to ONE shared compressed latent + a shared rotary key ---
        kv_down_projected = self.kv_down_projection(hidden_states)  # (batch, seq, kv_compressed_latent_dim + rotary_head_dim)
        compressed_kv_latent, key_rotary = torch.split(
            kv_down_projected, [self.kv_compressed_latent_dim, self.rotary_head_dim], dim=-1
        )
        # the rotary key is shared across all heads (unlike query_rotary, which is per-head) --
        # this is exactly what makes it cheap to cache
        key_rotary = key_rotary.view(batch_size, seq_len, 1, self.rotary_head_dim).transpose(1, 2)  # (batch, 1, seq, rotary_dim)
        key_rotary = apply_rotary_embeddings(key_rotary, rotary_cosine, rotary_sine)

        if self.is_absorbed:
            # --- Inference Matrix Absorption Path ---
            normed_latent = self.kv_down_projection_norm(compressed_kv_latent)
            
            if key_value_cache is not None:
                if key_value_cache[0] is not None:
                    normed_latent = torch.cat([key_value_cache[0], normed_latent], dim=1)
                    key_rotary = torch.cat([key_value_cache[1], key_rotary], dim=2)
                key_value_cache[0], key_value_cache[1] = normed_latent, key_rotary
                
            cached_seq_len = normed_latent.shape[1]
            
            # Absorb W_UK into query content
            query_content_absorbed = torch.einsum('bhsc,hcl->bhsl', query_content, self.W_UK)
            query_full = torch.cat([query_content_absorbed, query_rotary], dim=-1)
            query_full = query_full * self.scale_correction
            
            normed_latent_broadcast = normed_latent.unsqueeze(1).expand(batch_size, self.num_heads, cached_seq_len, self.kv_compressed_latent_dim)
            key_rotary_broadcast = key_rotary.expand(batch_size, self.num_heads, cached_seq_len, self.rotary_head_dim)
            
            key = torch.cat([normed_latent_broadcast, key_rotary_broadcast], dim=-1)
            value = normed_latent_broadcast
            
            use_causal_mask = key_value_cache is None or cached_seq_len == seq_len
            attention_output = F.scaled_dot_product_attention(
                query_full, key, value,
                dropout_p=self.dropout_probability if self.training else 0.0,
                is_causal=use_causal_mask,
            )
            return torch.einsum('bhsl,hle->bse', attention_output, self.W_O_absorbed)
        else:
            # --- Standard Training Path ---
            if key_value_cache is not None:
                # append this step's newly-computed latent/rotary-key onto whatever was cached from
                # previous steps, so attention below sees the FULL sequence so far, not just the
                # newest token(s) -- this is what makes autoregressive generation fast: each step
                # only computes the new token's projections, never redoes the old ones
                if key_value_cache[0] is not None:
                    compressed_kv_latent = torch.cat([key_value_cache[0], compressed_kv_latent], dim=1)  # concat along the sequence dimension
                    key_rotary = torch.cat([key_value_cache[1], key_rotary], dim=2)  # concat along the sequence dimension
                key_value_cache[0], key_value_cache[1] = compressed_kv_latent, key_rotary
    
            cached_seq_len = compressed_kv_latent.shape[1]  # total sequence length: cached tokens + this step's new token(s)
            # re-expand the (cached + new) compressed latent back out to full per-head keys/values.
            # See the "Simplification" note in the module docstring: DeepSeek's production code
            # avoids repeating this every step via matrix absorption; this implementation redoes
            # it every step, which is simpler to verify correct and produces identical results.
            key_value_expanded = self.kv_up_projection(self.kv_down_projection_norm(compressed_kv_latent))
            key_value_expanded = key_value_expanded.view(
                batch_size, cached_seq_len, self.num_heads, self.content_head_dim + self.value_head_dim
            ).transpose(1, 2)  # (batch, heads, cached_seq_len, content_head_dim + value_head_dim)
            key_content, value = torch.split(key_value_expanded, [self.content_head_dim, self.value_head_dim], dim=-1)
    
            # broadcast the single shared rotary key out to every head, then glue the "what" and
            # "where" parts back together into the full-size key that attention actually uses
            key_rotary_broadcast = key_rotary.expand(batch_size, self.num_heads, cached_seq_len, self.rotary_head_dim)
            key = torch.cat([key_content, key_rotary_broadcast], dim=-1)          # (batch, heads, cached_seq_len, query_key_head_dim)
            query_full = torch.cat([query_content, query_rotary], dim=-1)          # (batch, heads, seq_len, query_key_head_dim)
    
            # causal masking only matters when processing a full new sequence (no cache yet, or the
            # very first fill of the cache); single-token decode steps naturally attend to
            # everything cached so far without needing an explicit mask
            use_causal_mask = key_value_cache is None or cached_seq_len == seq_len
            # Flash attention requires queries, keys and values to share one head size. MLA's values
            # are deliberately smaller, which silently pushed PyTorch onto a slower attention kernel.
            # Zero-padding the values up to the query/key size and slicing the result back is exactly
            # equivalent -- the output is a weighted sum of values, so the padded dimensions stay zero --
            # and makes the fused flash kernel eligible. The attention scale is unchanged because it
            # comes from the query size, which is untouched.
            padded_value_dim = self.query_key_head_dim - self.value_head_dim
            if padded_value_dim > 0:
                value = F.pad(value, (0, padded_value_dim))
            attention_output = F.scaled_dot_product_attention(
                query_full, key, value,  # value has a DIFFERENT last dimension (value_head_dim) than query/key -- PyTorch's
                                          # scaled_dot_product_attention explicitly supports this, which is what lets MLA
                                          # use a smaller value size than its query/key attention-score dimension
                dropout_p=self.dropout_probability if self.training else 0.0,
                is_causal=use_causal_mask,
            )
            if padded_value_dim > 0:
                attention_output = attention_output[..., :self.value_head_dim]
            attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.num_heads * self.value_head_dim)
            return self.output_projection(attention_output)


class SwiGLUFeedForward(nn.Module):
    """Gated feed-forward block used by every expert (and every dense
    feed-forward layer): two parallel projections, one passed through the
    SiLU/Swish activation and multiplied elementwise with the other, then
    projected back down. Reference: Shazeer, 'GLU Variants Improve
    Transformer', 2020 (arXiv:2002.05202)."""

    def __init__(self, embedding_dimension: int, hidden_dimension: int, dropout_probability: float = 0.0):
        super().__init__()
        self.gate_projection = nn.Linear(embedding_dimension, hidden_dimension, bias=False)
        self.up_projection = nn.Linear(embedding_dimension, hidden_dimension, bias=False)
        self.down_projection = nn.Linear(hidden_dimension, embedding_dimension, bias=False)
        self.dropout = nn.Dropout(dropout_probability)

    def forward(self, hidden_states):
        gated = F.silu(self.gate_projection(hidden_states)) * self.up_projection(hidden_states)
        return self.dropout(self.down_projection(gated))


LEGACY_EXPERT_PROJECTIONS = (
    ("expert_gate_weights", "gate_projection"),
    ("expert_up_weights", "up_projection"),
    ("expert_down_weights", "down_projection"),
)


def stack_legacy_expert_weights(state_dict, prefix=""):
    """Converts one MoE layer's routed experts, in place, from the older
    layout (one SwiGLUFeedForward module per expert, keys like
    'routed_experts.3.gate_projection.weight') to the stacked layout this
    file now uses ('expert_gate_weights', with a leading expert dimension).
    Returns how many experts were converted; 0 if already stacked."""
    expert_count = 0
    while f"{prefix}routed_experts.{expert_count}.gate_projection.weight" in state_dict:
        expert_count += 1
    if expert_count == 0:
        return 0
    for stacked_name, projection_name in LEGACY_EXPERT_PROJECTIONS:
        keys = [f"{prefix}routed_experts.{index}.{projection_name}.weight" for index in range(expert_count)]
        state_dict[prefix + stacked_name] = torch.stack([state_dict.pop(key) for key in keys])
    return expert_count


def convert_legacy_expert_keys(state_dict):
    """Whole-model version of stack_legacy_expert_weights: returns a new dict
    with every MoE layer in the stacked layout. Used by upcycle.py, which
    edits expert tensors directly rather than through a live module."""
    converted = dict(state_dict)
    prefixes = sorted({key.split("routed_experts.")[0] for key in converted if ".routed_experts." in key})
    for prefix in prefixes:
        stack_legacy_expert_weights(converted, prefix)
    return converted
