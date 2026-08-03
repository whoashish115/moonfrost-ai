"""
Model + training size presets.

Architecture is DeepSeek-V2/V3's actual recipe, not a simplified stand-in:
  - Multi-head Latent Attention (MLA): queries/keys/values are projected
    through a low-rank "latent" bottleneck instead of full-size per-head
    projections, and only the small latent (+ a small decoupled RoPE key) is
    KV-cached during generation -- this is DeepSeek's specific inference-
    memory optimization.
  - DeepSeekMoE feed-forward: most layers use a bank of small "routed"
    expert feed-forward networks (a per-token router picks the top-k) plus
    one or more always-on "shared" expert feed-forward networks, instead of
    one big dense feed-forward network. Total *stored* parameter count is
    much bigger than a dense model of similar quality, but only a fraction
    of experts actually run per token.

Four sizes:
  tiny   - smoke tests, CPU-feasible, sanity checking the pipeline
  small  - default target for local RTX 3050 (6GB) training
  base   - extended local/rented-GPU run
  cloud  - meant for a rented multi-GPU box (see CLOUD_TRAINING.md); launch
           train.py under torchrun for multi-GPU data-parallel training

Run this file directly (`python config.py`) to print exact parameter counts
per preset, both TOTAL stored parameters and ACTIVE parameters per token
(these differ because of the Mixture-of-Experts layers).

The batch sizes in TRAIN_PRESETS were measured on a 6GB card rather than
estimated. The comment above TRAIN_PRESETS explains why a large vocabulary
makes that harder to predict than it looks.
"""
from dataclasses import dataclass


@dataclass
class ModelConfig:
    # --- tokenizer / sequence shape ---
    vocabulary_size: int = 65536   # number of distinct tokens the tokenizer can produce
    max_sequence_length: int = 1024  # maximum number of tokens the model can process at once (the "context window")

    # --- overall depth/width ---
    num_layers: int = 12            # how many transformer blocks are stacked
    embedding_dimension: int = 640  # size of the vector representing each token as it flows through the model
    num_attention_heads: int = 10   # how many parallel attention "heads" split embedding_dimension between them

    # Multi-head Latent Attention.
    #
    # Standard attention gives every head a slice of embedding_dimension directly. MLA instead
    # compresses the input into a small shared "latent" vector first, then expands back out to
    # per-head dimensions only when attention math actually needs them. These control the size
    # of that expanded per-head space and the compressed bottleneck.
    query_key_content_head_dim: int = 64   # per-head size of the "content" (non-positional) part of queries/keys
    query_key_rotary_head_dim: int = 32    # per-head size of the RoPE (positional) part of queries/keys -- shared across heads for keys
    value_head_dim: int = 64               # per-head size of values (can differ from the query/key head size)
    kv_compressed_latent_dim: int = 256    # size of the compressed latent that actually gets cached during generation -- THIS is MLA's memory saving
    query_compressed_latent_dim: int = 384  # size of the compressed latent used for queries (0 disables query compression entirely)

    # --- DeepSeekMoE feed-forward dimensions ---
    num_initial_dense_layers: int = 1     # keep the first N layers as a plain dense feed-forward network (stability, as in DeepSeek-V2)
    dense_feedforward_hidden_dim: int = 1728  # hidden size of the dense feed-forward network used in those initial layers
    expert_feedforward_hidden_dim: int = 640  # hidden size of EACH individual expert feed-forward network -- smaller than the dense one
    num_routed_experts: int = 6           # total number of "routed" experts a token can be sent to
    num_activated_experts_per_token: int = 2  # how many routed experts (top-k) actually process each token
    num_shared_experts: int = 1           # experts that process EVERY token, regardless of routing
    load_balance_loss_weight: float = 0.01  # weight of the auxiliary loss that discourages the router favoring a few experts
    capacity_factor: float = 1.25  # during training each expert accepts at most capacity_factor x its balanced share of tokens (see model.py)

    dropout_probability: float = 0.0
    rope_theta_base: float = 10000.0   # base frequency constant for the rotary position embedding formula
    normalization_epsilon: float = 1e-5  # small constant added inside RMSNorm to avoid dividing by zero
    tie_input_output_embeddings: bool = True  # share one matrix between the input token embedding and the output prediction layer

    def __post_init__(self):
        assert self.num_activated_experts_per_token <= self.num_routed_experts


MODEL_PRESETS = {
    "tiny": ModelConfig(
        vocabulary_size=65536, max_sequence_length=256, num_layers=4, embedding_dimension=256, num_attention_heads=4,
        query_key_content_head_dim=48, query_key_rotary_head_dim=16, value_head_dim=48,
        kv_compressed_latent_dim=96, query_compressed_latent_dim=0,
        num_initial_dense_layers=1, dense_feedforward_hidden_dim=640, expert_feedforward_hidden_dim=256,
        num_routed_experts=4, num_activated_experts_per_token=1, num_shared_experts=1,
    ),
    "small": ModelConfig(
        vocabulary_size=65536, max_sequence_length=1024, num_layers=10, embedding_dimension=640, num_attention_heads=10,
        query_key_content_head_dim=64, query_key_rotary_head_dim=32, value_head_dim=64,
        kv_compressed_latent_dim=256, query_compressed_latent_dim=384,
        num_initial_dense_layers=1, dense_feedforward_hidden_dim=1728, expert_feedforward_hidden_dim=576,
        num_routed_experts=6, num_activated_experts_per_token=2, num_shared_experts=1,
    ),
    "base": ModelConfig(
        vocabulary_size=65536, max_sequence_length=1536, num_layers=16, embedding_dimension=896, num_attention_heads=14,
        query_key_content_head_dim=64, query_key_rotary_head_dim=32, value_head_dim=64,
        kv_compressed_latent_dim=384, query_compressed_latent_dim=576,
        num_initial_dense_layers=1, dense_feedforward_hidden_dim=2432, expert_feedforward_hidden_dim=768,
        num_routed_experts=8, num_activated_experts_per_token=2, num_shared_experts=1,
    ),
    "cloud": ModelConfig(
        vocabulary_size=65536, max_sequence_length=2048, num_layers=24, embedding_dimension=1536, num_attention_heads=24,
        query_key_content_head_dim=64, query_key_rotary_head_dim=32, value_head_dim=64,
        kv_compressed_latent_dim=512, query_compressed_latent_dim=1024,
        num_initial_dense_layers=1, dense_feedforward_hidden_dim=4096, expert_feedforward_hidden_dim=1024,
        num_routed_experts=16, num_activated_experts_per_token=4, num_shared_experts=2,
    ),
    "chat": ModelConfig(
        # The run as it happened: 777M total, 161M active per token, 6B tokens of
        # FineWeb-Edu. The 32 routed experts supply capacity while top-3 routing keeps the
        # per-token cost near 161M, and the 32,768-token vocabulary halves what the output
        # projection costs against a 65,536-token one.
        vocabulary_size=32768, max_sequence_length=1024, num_layers=14, embedding_dimension=896, num_attention_heads=14,
        query_key_content_head_dim=64, query_key_rotary_head_dim=32, value_head_dim=64,
        kv_compressed_latent_dim=320, query_compressed_latent_dim=512,
        num_initial_dense_layers=1, dense_feedforward_hidden_dim=2432, expert_feedforward_hidden_dim=608,
        num_routed_experts=32, num_activated_experts_per_token=3, num_shared_experts=1,
    ),
    "tiny_dense": ModelConfig(
        vocabulary_size=65536, max_sequence_length=256, num_layers=4, embedding_dimension=256, num_attention_heads=4,
        query_key_content_head_dim=48, query_key_rotary_head_dim=16, value_head_dim=48,
        kv_compressed_latent_dim=96, query_compressed_latent_dim=0,
        num_initial_dense_layers=4, dense_feedforward_hidden_dim=512, expert_feedforward_hidden_dim=0,
        num_routed_experts=0, num_activated_experts_per_token=0, num_shared_experts=0,
    ),
}


@dataclass
