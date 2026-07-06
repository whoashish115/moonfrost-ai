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


class DeepSeekMixtureOfExperts(nn.Module):
    """DeepSeekMoE feed-forward layer: a per-token router picks the top-k
    routed experts; shared experts run for every token.

    Two implementation choices matter for training speed:

    * Routed experts are stored as three STACKED weight tensors with a
      leading expert dimension, rather than as a list of separate modules.
      The earlier layout had to torch.stack every expert's weights on every
      forward pass in order to run them together -- a copy of all
      routed-expert parameters per step, which grows with the expert count
      and dominates step time once there are dozens of experts.

    * Tokens reach the experts through fixed-capacity queues (GShard /
      Switch Transformer style). Every shape is static and there are no
      GPU->CPU synchronizations, so torch.compile can fuse the layer, and
      all experts run inside one batched matmul. During training, tokens
      beyond an expert's capacity skip that expert and the residual
      connection carries them forward unchanged. At inference the capacity
      is large enough that nothing is ever dropped.

    Checkpoints written in the older one-module-per-expert layout still load
    unchanged; see _load_from_state_dict."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.num_routed_experts = config.num_routed_experts
        self.num_activated_experts_per_token = config.num_activated_experts_per_token
        self.load_balance_loss_weight = config.load_balance_loss_weight
        self.capacity_factor = getattr(config, "capacity_factor", 1.25)
        self.expert_dropout_probability = config.dropout_probability

        embedding_dimension = config.embedding_dimension
        hidden_dimension = config.expert_feedforward_hidden_dim
        self.router = nn.Linear(embedding_dimension, self.num_routed_experts, bias=False)
        # expert e is a SwiGLU block: gate and up project embedding -> hidden, down projects back
        self.expert_gate_weights = nn.Parameter(torch.empty(self.num_routed_experts, hidden_dimension, embedding_dimension))
        self.expert_up_weights = nn.Parameter(torch.empty(self.num_routed_experts, hidden_dimension, embedding_dimension))
        self.expert_down_weights = nn.Parameter(torch.empty(self.num_routed_experts, embedding_dimension, hidden_dimension))
        for weight in (self.expert_gate_weights, self.expert_up_weights, self.expert_down_weights):
            nn.init.normal_(weight, mean=0.0, std=0.02)  # the init nn.Linear layers get in GPT._initialize_weights

        self.shared_experts = nn.ModuleList([
            SwiGLUFeedForward(config.embedding_dimension, config.expert_feedforward_hidden_dim, config.dropout_probability)
            for _ in range(config.num_shared_experts)
        ])
        self.last_load_balance_loss = torch.tensor(0.0)  # read by the parent GPT model after forward() to add to the training loss

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        stack_legacy_expert_weights(state_dict, prefix)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def forward(self, hidden_states):
        batch_size, seq_len, embedding_dim = hidden_states.shape
        flat_hidden_states = hidden_states.reshape(-1, embedding_dim)  # (num_tokens, embedding_dim)
        num_tokens = flat_hidden_states.shape[0]
        num_experts = self.num_routed_experts
        top_k = self.num_activated_experts_per_token

        router_logits = self.router(flat_hidden_states)  # (num_tokens, num_routed_experts)
        router_probabilities = F.softmax(router_logits, dim=-1, dtype=torch.float32)  # float32 for a stable softmax
        top_k_probabilities, top_k_expert_indices = torch.topk(router_probabilities, top_k, dim=-1)
        # renormalize so each token's chosen experts' weights sum to 1
        top_k_probabilities = (top_k_probabilities / top_k_probabilities.sum(dim=-1, keepdim=True)).to(hidden_states.dtype)

        if self.training:
            capacity = math.ceil(num_tokens * top_k / num_experts * self.capacity_factor)
        elif num_tokens <= 4096:
            capacity = num_tokens  # inference: large enough that no token can ever be dropped
        else:
            capacity = math.ceil(num_tokens * top_k / num_experts * 2.0)  # large no-grad evaluation batches
        capacity = max(1, min(capacity, num_tokens))

        flat_expert_indices = top_k_expert_indices.reshape(-1)  # (num_tokens * top_k,)
        # one-hot by comparison rather than F.one_hot, whose input-range check forces a device sync
        assignment_one_hot = (
            flat_expert_indices.unsqueeze(-1) == torch.arange(num_experts, device=flat_expert_indices.device)
        ).to(torch.int32)  # (num_tokens * top_k, num_experts)
        position_in_expert = (assignment_one_hot.cumsum(dim=0) * assignment_one_hot).sum(dim=-1) - 1
        is_kept = position_in_expert < capacity
        overflow_slot = num_experts * capacity  # one scratch row that dropped assignments write into
        slot = torch.where(is_kept, flat_expert_indices * capacity + position_in_expert,
                           torch.full_like(position_in_expert, overflow_slot))

        token_index = torch.arange(num_tokens, device=flat_hidden_states.device).repeat_interleave(top_k)
        dispatched = flat_hidden_states.new_zeros(overflow_slot + 1, embedding_dim).index_add(
            0, slot, flat_hidden_states.index_select(0, token_index)
        )
        expert_inputs = dispatched[:overflow_slot].view(num_experts, capacity, embedding_dim)

        gated = F.silu(torch.bmm(expert_inputs, self.expert_gate_weights.transpose(1, 2))) * torch.bmm(
            expert_inputs, self.expert_up_weights.transpose(1, 2)
        )
        expert_outputs = torch.bmm(gated, self.expert_down_weights.transpose(1, 2))  # (experts, capacity, embedding_dim)
        if self.training and self.expert_dropout_probability > 0:
            expert_outputs = F.dropout(expert_outputs, self.expert_dropout_probability, training=True)

        expert_outputs = torch.cat([expert_outputs.reshape(overflow_slot, embedding_dim),
                                    expert_outputs.new_zeros(1, embedding_dim)], dim=0)
        assignment_weights = (top_k_probabilities.reshape(-1) * is_kept.to(top_k_probabilities.dtype)).unsqueeze(-1)
        weighted = expert_outputs.index_select(0, slot) * assignment_weights.to(expert_outputs.dtype)
        combined_output = weighted.view(num_tokens, top_k, embedding_dim).sum(dim=1).to(flat_hidden_states.dtype)

        for shared_expert in self.shared_experts:
            combined_output = combined_output + shared_expert(flat_hidden_states)

        # Load-balancing auxiliary loss (Switch-Transformer / DeepSeekMoE style): for each expert,
        # the hard fraction of top-k slots routed to it times the soft average router probability
        # on it. Minimizing their dot product discourages the router from collapsing onto a few
        # favourite experts and leaving the rest of the model's capacity unused.
        fraction_routed_to_expert = (assignment_one_hot.sum(dim=0).to(router_probabilities.dtype)
                                     / (num_tokens * top_k))  # (num_routed_experts,)
        average_router_probability = router_probabilities.mean(dim=0)  # (num_routed_experts,)
        self.last_load_balance_loss = (
            self.load_balance_loss_weight * num_experts * torch.sum(fraction_routed_to_expert * average_router_probability)
        )

        return combined_output.reshape(batch_size, seq_len, embedding_dim)


class TransformerBlock(nn.Module):
    """One transformer block: attention (with a residual connection) followed
    by a feed-forward network (with a residual connection), each normalized
    beforehand ('pre-norm'). Whether the feed-forward network is a plain
    dense SwiGLUFeedForward or a DeepSeekMixtureOfExperts is decided once,
    at construction time, by is_mixture_of_experts_layer."""

    def __init__(self, config: ModelConfig, is_mixture_of_experts_layer: bool):
        super().__init__()
        self.is_mixture_of_experts_layer = is_mixture_of_experts_layer
        self.pre_attention_norm = RMSNorm(config.embedding_dimension, config.normalization_epsilon)
        self.attention = MultiHeadLatentAttention(config)
        self.pre_feedforward_norm = RMSNorm(config.embedding_dimension, config.normalization_epsilon)
        self.feed_forward = (
            DeepSeekMixtureOfExperts(config) if is_mixture_of_experts_layer
            else SwiGLUFeedForward(config.embedding_dimension, config.dense_feedforward_hidden_dim, config.dropout_probability)
        )

    def forward(self, hidden_states, rotary_cosine, rotary_sine, key_value_cache=None):
        hidden_states = hidden_states + self.attention(
            self.pre_attention_norm(hidden_states), rotary_cosine, rotary_sine, key_value_cache
        )
        hidden_states = hidden_states + self.feed_forward(self.pre_feedforward_norm(hidden_states))
        return hidden_states

    def get_load_balance_loss(self):
        return self.feed_forward.last_load_balance_loss if self.is_mixture_of_experts_layer else 0.0


class GPT(nn.Module):
    """The full decoder-only language model: a token embedding, a stack of
    TransformerBlocks, a final normalization, and an output projection back
    to vocabulary-sized logits."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocabulary_size, config.embedding_dimension)
        self.blocks = nn.ModuleList([
            TransformerBlock(config, is_mixture_of_experts_layer=(layer_index >= config.num_initial_dense_layers))
            for layer_index in range(config.num_layers)
        ])
        self.final_norm = RMSNorm(config.embedding_dimension, config.normalization_epsilon)
        self.output_projection = nn.Linear(config.embedding_dimension, config.vocabulary_size, bias=False)
        if config.tie_input_output_embeddings:
            # sharing one matrix between the input embedding and the output projection is a
            # well-established parameter-saving trick (both are conceptually "a lookup table
            # between token ids and embedding_dimension-sized vectors", just used in opposite directions)
            self.output_projection.weight = self.token_embedding.weight

        rotary_cosine, rotary_sine = precompute_rotary_embeddings(
            config.query_key_rotary_head_dim, config.max_sequence_length, config.rope_theta_base, device="cpu"
        )
        self.register_buffer("rotary_cosine", rotary_cosine, persistent=False)
        self.register_buffer("rotary_sine", rotary_sine, persistent=False)

        self.apply(self._initialize_weights)
        # scaled initialization for residual output projections, following GPT-2's approach --
        # this keeps activation magnitudes from growing as more layers are stacked. Matches
        # MultiHeadLatentAttention's output_projection and every SwiGLUFeedForward's
        # down_projection, including inside Mixture-of-Experts experts (their parameter names
        # still end with these same suffixes, since they're just nested SwiGLUFeedForward modules)
        for parameter_name, parameter in self.named_parameters():
            if (parameter_name.endswith("output_projection.weight")
                    or parameter_name.endswith("down_projection.weight")
                    or parameter_name.endswith("expert_down_weights")):
                nn.init.normal_(parameter, mean=0.0, std=0.02 / math.sqrt(2 * config.num_layers))

    def _initialize_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def count_parameters(self):
        # nn.Module.parameters() de-duplicates by tensor identity, so a tied output_projection
        # weight (the same Parameter object as token_embedding.weight) is only counted once here.
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(self, token_ids: torch.Tensor, target_token_ids: Optional[torch.Tensor] = None,
                key_value_caches: Optional[List[List[torch.Tensor]]] = None, start_position: int = 0):
        batch_size, seq_len = token_ids.shape
        hidden_states = self.token_embedding(token_ids)
        rotary_cosine = self.rotary_cosine[start_position:start_position + seq_len].to(hidden_states.device)
        rotary_sine = self.rotary_sine[start_position:start_position + seq_len].to(hidden_states.device)

        for layer_index, block in enumerate(self.blocks):
            this_layer_cache = key_value_caches[layer_index] if key_value_caches is not None else None
            hidden_states = block(hidden_states, rotary_cosine, rotary_sine, this_layer_cache)
        hidden_states = self.final_norm(hidden_states)

        if target_token_ids is not None:
            logits = self.output_projection(hidden_states)
            # reshape, not view: a caller that builds targets as a slice (the natural
            # `tokens[:, 1:]`) hands us a non-contiguous tensor, and .view() raises on those.
            # reshape falls back to a copy only when it has to, so contiguous callers pay nothing.
            next_token_prediction_loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), target_token_ids.reshape(-1), ignore_index=-1
            )
            load_balance_loss = sum(block.get_load_balance_loss() for block in self.blocks)
            return logits, next_token_prediction_loss + load_balance_loss
        else:
            # generation only reads the final position, so the output projection runs on one
            # point computing (and allocating memory for) logits for every earlier position too
            logits = self.output_projection(hidden_states[:, [-1], :])
            return logits, None

    def create_empty_key_value_caches(self):
        return [[None, None] for _ in range(self.config.num_layers)]

    def enable_inference_absorption(self):
        """Switches every attention layer onto the matrix-absorption fast
        path (see MultiHeadLatentAttention.absorb_matrices). Idempotent, so
        the generate* methods can call it unconditionally."""
        for block in self.blocks:
            block.attention.absorb_matrices()

    def disable_inference_absorption(self):
        """Undoes enable_inference_absorption() and frees the derived
        tensors. Called automatically by .train()."""
        for block in self.blocks:
            block.attention.release_absorbed_matrices()

    def estimate_key_value_cache_bytes(self, sequence_length: int, bytes_per_element: int = 4) -> int:
        """How much memory the KV cache needs for a sequence of the given
        length. MLA caches only the compressed latent plus the shared rotary
        key, per layer -- which is why this number is small enough that a
        6GB card can hold a full-context conversation."""
        per_token_per_layer = self.config.kv_compressed_latent_dim + self.config.query_key_rotary_head_dim
        return per_token_per_layer * self.config.num_layers * sequence_length * bytes_per_element

    def configure_optimizer(self, weight_decay, learning_rate, adam_betas, device_type):
        parameters_with_weight_decay, parameters_without_weight_decay = [], []
        for parameter_name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            # only matrices (2D+) get weight decay; biases and 1D norm weights don't, following
            # standard practice (decaying those tends to hurt rather than help)
            (parameters_with_weight_decay if parameter.dim() >= 2 else parameters_without_weight_decay).append(parameter)
        parameter_groups = [
            {"params": parameters_with_weight_decay, "weight_decay": weight_decay},
            {"params": parameters_without_weight_decay, "weight_decay": 0.0},
        ]
        fused_adamw_available = device_type == "cuda"
        return torch.optim.AdamW(parameter_groups, lr=learning_rate, betas=adam_betas, fused=fused_adamw_available)


    # ------------------------------------------------------------------
    # Sampling and generation
    # ------------------------------------------------------------------

    def _apply_repetition_penalties(self, logits, recent_token_ids, repetition_penalty,
                                     frequency_penalty, presence_penalty, protected_token_ids):
        """Discourages the model from looping on text it just produced.

        Three separate mechanisms, all optional:
          - repetition_penalty (CTRL-style, multiplicative): divides the
            logit of any recently-seen token by the penalty.
          - frequency_penalty (OpenAI-style, additive): subtracts
            penalty * (how many times the token appeared).
          - presence_penalty (additive): subtracts a flat penalty from any
            token that appeared at all.

        Two things the previous version of this got wrong, both of which
        visibly wrecked output quality on a small model:

          1. It penalized the ENTIRE sequence including the prompt. In a
             chat setting the prompt contains the user's question, so every
             word of the question became less likely in the answer -- which
             is exactly backwards, since a good answer usually reuses the
             question's vocabulary. Only `recent_token_ids` (a sliding
             window over what the model itself just generated) is penalized
             now.
          2. It penalized special tokens too, including <|endoftext|>. That
             made the stop token progressively harder to emit the longer
             generation ran, so replies rambled instead of stopping.
             protected_token_ids are now exempt.
        """
        if recent_token_ids is None or recent_token_ids.numel() == 0:
            return logits
        if repetition_penalty == 1.0 and frequency_penalty == 0.0 and presence_penalty == 0.0:
            return logits

        for batch_index in range(logits.size(0)):
            token_ids_this_row = recent_token_ids[batch_index]
            unique_token_ids, occurrence_counts = torch.unique(token_ids_this_row, return_counts=True)
            if protected_token_ids:
                protected = torch.tensor(sorted(protected_token_ids), device=logits.device, dtype=unique_token_ids.dtype)
                keep_mask = ~torch.isin(unique_token_ids, protected)
                unique_token_ids, occurrence_counts = unique_token_ids[keep_mask], occurrence_counts[keep_mask]
            if unique_token_ids.numel() == 0:
                continue

            if repetition_penalty != 1.0:
                seen_logits = logits[batch_index, unique_token_ids]
                # a negative logit has to be MULTIPLIED by the penalty to become less likely,
                # while a positive one has to be divided -- hence the branch
                logits[batch_index, unique_token_ids] = torch.where(
                    seen_logits > 0, seen_logits / repetition_penalty, seen_logits * repetition_penalty
                )
            if frequency_penalty != 0.0:
                logits[batch_index, unique_token_ids] -= frequency_penalty * occurrence_counts.to(logits.dtype)
            if presence_penalty != 0.0:
                logits[batch_index, unique_token_ids] -= presence_penalty
        return logits

    def _sample_next_token(self, logits, temperature, top_k, top_p, min_p=None,
                            recent_token_ids=None, repetition_penalty=1.1,
                            frequency_penalty=0.0, presence_penalty=0.0,
                            protected_token_ids=(), generator=None):
        """Turns the model's raw logits for the next position into one
        sampled token id, applying (in order): repetition penalties,
        temperature, then the top-k / top-p / min-p truncation filters.

        temperature <= 0 means greedy decoding (always take the most likely
        token). The previous version divided by max(temperature, 1e-5)
        instead, which turns every logit into +/-inf and then NaN -- so
        "temperature 0" used to crash rather than being deterministic."""
        logits = logits[:, -1, :].float().clone()  # only the most recently predicted position matters
        logits = self._apply_repetition_penalties(
            logits, recent_token_ids, repetition_penalty, frequency_penalty, presence_penalty, protected_token_ids
        )

        if temperature is None or temperature <= 0.0:
            return torch.argmax(logits, dim=-1, keepdim=True)
        logits = logits / temperature

        if top_k is not None and top_k > 0:
            top_k = min(top_k, logits.size(-1))
            top_k_values, _ = torch.topk(logits, top_k)
            logits[logits < top_k_values[:, [-1]]] = -float("inf")

        if min_p is not None and min_p > 0.0:
            # min-p keeps tokens whose probability is at least min_p times the top token's
            # probability. Unlike top-p it adapts to how confident the model is: a confident
            # step keeps very few tokens, an uncertain one keeps many.
            probabilities = F.softmax(logits, dim=-1)
            threshold = probabilities.max(dim=-1, keepdim=True).values * min_p
            logits[probabilities < threshold] = -float("inf")

        if top_p is not None and 0.0 < top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            sorted_probabilities = F.softmax(sorted_logits, dim=-1)
            cumulative_probability = torch.cumsum(sorted_probabilities, dim=-1)
            # "- sorted_probabilities" shifts the comparison so the token that CROSSES the
            # threshold is kept, guaranteeing at least one survivor even if the top token
            # alone already exceeds top_p
            tokens_to_remove = (cumulative_probability - sorted_probabilities) > top_p
            sorted_logits[tokens_to_remove] = -float("inf")
            logits = torch.full_like(logits, -float("inf")).scatter(-1, sorted_indices, sorted_logits)

        probabilities = F.softmax(logits, dim=-1)
        # a filter combination can leave a row that's all -inf (and therefore all-NaN after
        # softmax); fall back to greedy on that row rather than letting multinomial raise
        invalid_rows = ~torch.isfinite(probabilities).all(dim=-1) | (probabilities.sum(dim=-1) <= 0)
        if invalid_rows.any():
            return torch.argmax(logits.masked_fill(~torch.isfinite(logits), -1e30), dim=-1, keepdim=True)
        return torch.multinomial(probabilities, num_samples=1, generator=generator)

    def _prefill(self, token_ids, key_value_caches):
        """Runs the whole prompt through the model in one pass, filling the
        key/value caches, and returns the logits for the last position."""
        logits, _ = self.forward(token_ids, key_value_caches=key_value_caches, start_position=0)
        return logits

    @torch.no_grad()
    def generate(self, prompt_token_ids: torch.Tensor, max_new_tokens: int, temperature: float = 0.8,
                 top_k: Optional[int] = 50, top_p: Optional[float] = None, min_p: Optional[float] = None,
                 repetition_penalty: float = 1.1, frequency_penalty: float = 0.0, presence_penalty: float = 0.0,
                 repetition_window: int = 256, protected_token_ids=(), stop_token_id: Optional[int] = None,
                 seed: Optional[int] = None):
        """Batched, non-streaming generation (used by sample.py). Returns the
        full tensor of prompt + generated token ids."""
        self.eval()
        self.enable_inference_absorption()

        generator = None
        if seed is not None:
            generator = torch.Generator(device=prompt_token_ids.device)
            generator.manual_seed(seed)

        batch_size, prompt_length = prompt_token_ids.shape
        max_sequence_length = self.config.max_sequence_length
        if prompt_length > max_sequence_length:
            prompt_token_ids = prompt_token_ids[:, -max_sequence_length:]
            prompt_length = max_sequence_length

        key_value_caches = self.create_empty_key_value_caches()
        logits = self._prefill(prompt_token_ids, key_value_caches)
        current_length = prompt_length
        generated_token_ids = prompt_token_ids
        newly_generated_token_ids = prompt_token_ids[:, :0]  # empty, correct dtype/device

        finished = torch.zeros(batch_size, dtype=torch.bool, device=prompt_token_ids.device)
        for _ in range(max_new_tokens):
            next_token_id = self._sample_next_token(
                logits, temperature, top_k, top_p, min_p,
                recent_token_ids=newly_generated_token_ids[:, -repetition_window:],
                repetition_penalty=repetition_penalty, frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty, protected_token_ids=protected_token_ids, generator=generator,
            )
            generated_token_ids = torch.cat([generated_token_ids, next_token_id], dim=1)
            newly_generated_token_ids = torch.cat([newly_generated_token_ids, next_token_id], dim=1)
            if stop_token_id is not None:
                finished |= next_token_id.squeeze(1) == stop_token_id
                if bool(finished.all()):
                    break
            if current_length >= max_sequence_length:
                break
            logits, _ = self.forward(next_token_id, key_value_caches=key_value_caches, start_position=current_length)
            current_length += 1
        return generated_token_ids

    @torch.no_grad()
    def generate_stream(self, prompt_token_ids: torch.Tensor, max_new_tokens: int, temperature: float = 0.8,
                         top_k: Optional[int] = 50, top_p: Optional[float] = None, min_p: Optional[float] = None,
                         stop_token_id: Optional[int] = None, stop_token_ids=(),
                         repetition_penalty: float = 1.1, frequency_penalty: float = 0.0,
                         presence_penalty: float = 0.0, repetition_window: int = 256,
                         protected_token_ids=(), seed: Optional[int] = None,
                         should_stop=None):
        """Single-sequence (batch_size=1) generator yielding one new token id
        per step, for streaming chat.

        Context handling: when the conversation reaches max_sequence_length,
        the oldest half of the context is dropped and the model is re-filled
        from the newer half. The previous version threw away the ENTIRE
        cache and continued from just the single most recent token, which
        made the model forget the question it was halfway through answering
        and start generating unrelated text -- the "it goes off the rails on
        long replies" symptom.

        `should_stop` is an optional zero-argument callable checked once per
        token; returning True ends generation cleanly. The web server uses it
        to implement the Stop button.
        """
        self.eval()
        self.enable_inference_absorption()

        generator = None
        if seed is not None:
            generator = torch.Generator(device=prompt_token_ids.device)
            generator.manual_seed(seed)

        stop_token_id_set = set(stop_token_ids)
        if stop_token_id is not None:
            stop_token_id_set.add(stop_token_id)

        batch_size, prompt_length = prompt_token_ids.shape
        assert batch_size == 1, "generate_stream only supports a single sequence at a time"
        max_sequence_length = self.config.max_sequence_length
        if prompt_length > max_sequence_length:
            prompt_token_ids = prompt_token_ids[:, -max_sequence_length:]
            prompt_length = max_sequence_length

        key_value_caches = self.create_empty_key_value_caches()
        logits = self._prefill(prompt_token_ids, key_value_caches)
        current_length = prompt_length
        # the full window the model is currently conditioned on, used only for re-prefilling
        context_token_ids = prompt_token_ids
        newly_generated_token_ids = prompt_token_ids[:, :0]

        for _ in range(max_new_tokens):
            if should_stop is not None and should_stop():
                return
            next_token_id = self._sample_next_token(
                logits, temperature, top_k, top_p, min_p,
                recent_token_ids=newly_generated_token_ids[:, -repetition_window:],
                repetition_penalty=repetition_penalty, frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty, protected_token_ids=protected_token_ids, generator=generator,
            )
            next_token_id_value = int(next_token_id.item())
            context_token_ids = torch.cat([context_token_ids, next_token_id], dim=1)
            newly_generated_token_ids = torch.cat([newly_generated_token_ids, next_token_id], dim=1)
            yield next_token_id_value
            if next_token_id_value in stop_token_id_set:
                return

            if current_length >= max_sequence_length:
                # slide the window: keep the most recent half and rebuild the cache from it,
                # so recent conversation survives instead of being discarded
                keep_length = max(1, max_sequence_length // 2)
                context_token_ids = context_token_ids[:, -keep_length:]
                key_value_caches = self.create_empty_key_value_caches()
                logits = self._prefill(context_token_ids, key_value_caches)
                current_length = context_token_ids.shape[1]
                continue

            logits, _ = self.forward(next_token_id, key_value_caches=key_value_caches, start_position=current_length)
            current_length += 1
