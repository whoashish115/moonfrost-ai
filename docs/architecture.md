# Architecture

Moonfrost follows DeepSeek-V2/V3. Two ideas do the work: attention that compresses its
cache, and feed-forward layers where only a few experts fire per token.

| Property | Value |
|---|---|
| Layers | 14 (layer 0 dense, layers 1-13 MoE) |
| Hidden size | 896 |
| Attention heads | 14 |
| Total parameters | 777,148,032 |
| Active per token | 161,036,224 |
| Context | 1,024 tokens |
| Vocabulary | 32,768 |

## Multi-head Latent Attention

Standard attention caches a key and a value per head per token. At 14 heads that is 28
vectors of size 64 for every token. MLA instead projects the input down to **one shared
latent of 320 numbers**, caches that, and reconstructs keys and values from it when
needed.

Position is the complication. RoPE rotates keys by their position, so a rotated key
cannot be reconstructed from an unrotated latent. The fix is to keep them apart:

| Part | Size | Carries |
|---|---|---|
| Content key/value | 320, shared | what the token means, no position |
| Rotary key | 32, shared across heads | where the token is |
| Query content | 64 per head | |
| Query rotary | 32 per head | |

The attention dot product runs over the two concatenated, 96 numbers per head. The KV
cache holds 352 numbers per token instead of 1,792 - about **5x smaller**.

At inference the up-projection matrices are folded into the query and output projections
once, so reconstruction disappears from the per-token cost. `enable_inference_absorption()`
does this; `.train()` releases it.

## DeepSeekMoE

Each MoE layer holds **32 routed experts and 1 shared expert**. A router scores the token
against all 32, the **top 3** run, and the shared expert runs for every token. So a token
passes through 4 of 33 experts.

```
token -> router -> top-3 of 32 routed experts  ┐
      -> shared expert (always)                ┴-> weighted sum
```

This is where the gap between 777M and 161M comes from: all 33 experts are stored, 4 are
used.

Expert weights are stored as **stacked tensors** of shape `(32, hidden, ffn)` rather than
32 separate modules, and dispatch is done by capacity, GShard style. Each expert takes at
most `capacity_factor x tokens x k / experts` tokens, the rest are dropped for that layer.
The earlier implementation looped over experts in Python and ran at 6,056 tokens/second;
the stacked version runs at **179,000**.

Two losses keep the router honest:

- **load balancing**, weight 0.01, penalising uneven expert use
- the router's own softmax, which decides the weighted sum

Layer 0 is a plain SwiGLU feed-forward. Routing from the very first layer destabilised
early training, and one dense layer costs little.

## The rest

- **RMSNorm** before attention and feed-forward, no bias terms anywhere
- **SwiGLU** activations
- **RoPE** with theta 10,000, computed in float32 then cast back to the input dtype.
  Doing it in bfloat16 lost enough precision at long positions to matter
- **Tied embeddings**: the input embedding and output projection are one 32,768 x 896
  matrix, saving 29M parameters

## Tokenizer

Byte-level BPE, **32,768 tokens**, trained from scratch on 1.5GB of the same FineWeb-Edu
text. A 65k vocabulary would have spent about 18% of every step in the output layer
alone.

Special tokens: `<|endoftext|>`, `<|pad|>`, `<|system|>`, `<|user|>`, `<|assistant|>`,
plus four reserved for images that are unused.

## Chat format

```
<|system|>You are Moonfrost...<|user|>Why is the sky blue?<|assistant|>Because...<|endoftext|>
```

Loss is masked on system and user turns, so the model only learns to produce assistant
text: **60.7% of packed positions are supervised**. Both `<|endoftext|>` and `<|user|>`
stop generation, because an undertrained model will otherwise write the next user turn
itself.

## Where the parameters are

| Component | Parameters | Share |
|---|---|---|
| Routed experts (13 layers x 32) | 543M | 70% |
| Embedding (tied) | 29M | 4% |
| Attention (14 layers) | 118M | 15% |
| Shared experts + dense layer | 84M | 11% |

Seventy per cent of the model is experts that are mostly idle for any given token. That
is the trade: the capacity of a 777M model at roughly the compute of a 161M one.
