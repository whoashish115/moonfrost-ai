# The pipeline, block by block

This walks the whole project in the order it runs, from raw text to a served model. Every
section names the file, the functions inside it, and what each one actually does to the
data. Read it top to bottom and you have followed a token from a web page into a reply.

```
download_pretrain_data.py  ->  tokenizer_train.py  ->  train.py  ->  sft_data.py  ->  sft_train.py  ->  export/serve
     raw text                   32,768 BPE merges       base LM       packed chat rows    chat model
```

| Stage | Script | Input | Output |
|---|---|---|---|
| 1 | `download_pretrain_data.py` | FineWeb-Edu shards | `train.bin`, `val.bin` |
| 2 | `tokenizer_train.py` | 1.5GB of that text | `tokenizer/tokenizer.json` |
| 3 | `config.py` | nothing | the shape of the model |
| 4 | `model.py` | token ids | logits |
| 5 | `train.py` | `train.bin` | a pretrained checkpoint |
| 6 | `sft_data.py`, `persona_data.py` | chat datasets | packed rows + label masks |
| 7 | `sft_train.py` | those rows | the chat checkpoint |
| 8 | `evaluate_model.py` | a checkpoint | `docs/eval.json` |
| 9 | `server.py`, `export_to_huggingface.py` | a checkpoint | a chat interface, a HF repo |

`cloud_run.py` is the wrapper that runs stages 1, 5, 6, 7 and 9 on rented GPUs.

---

## 1. Text in, integers out

**`download_pretrain_data.py`**

`main()` streams FineWeb-Edu `sample/10BT` one shard at a time rather than downloading the
whole set, because the whole set does not fit on the disk this was built on. For each
document it calls `clean_text()`, which collapses runs of whitespace and drops documents
below a length threshold, then encodes it with the trained tokenizer and appends an
`<|endoftext|>` token so the model learns where documents end.

The token ids are written to a flat `uint16` binary file. Not JSON, not a tensor file: a
raw array, because the training loop wants to memory-map it and slice random windows out
of it without parsing anything. `uint16` is enough for a 32,768-token vocabulary and halves
the file size against `uint32`.

Two files come out, `train.bin` and `val.bin`, from disjoint shards. Phase 1 of training
reads shards 0 to 2 and phase 2 reads shards 3 to 7, so no document is seen twice across
the whole run.

## 2. Building the vocabulary

**`tokenizer_train.py`**

`main()` trains a byte-level BPE tokenizer on about 1.5GB of the same text, targeting
**32,768 tokens**. Byte-level means the base alphabet is the 256 byte values, so there is
no unknown token: any input encodes to something.

Vocabulary size is a real architectural decision, not a detail. The output projection is a
`hidden x vocabulary` matrix, and at hidden size 896 a 65,536-token vocabulary would have
put roughly 18% of every forward pass in that one layer. 32,768 halves it.

Seven special tokens are added: `<|endoftext|>`, `<|pad|>`, `<|system|>`, `<|user|>`,
`<|assistant|>`, and four reserved image tokens that nothing uses yet. They are added as
real vocabulary entries so they can never be produced by merging ordinary text.

## 3. The shape of the model

**`config.py`**

`ModelConfig` is a dataclass holding every structural number: 14 layers, hidden size 896,
14 attention heads, KV latent 320, rotary dimension 32, 32 routed experts, top-3 routing,
1 shared expert, context 1,024. `__post_init__()` validates the combination, because most
ways of getting these wrong produce a model that trains for an hour and then fails a shape
assertion.

`count_total_parameters()` and `count_active_parameters_per_token()` walk that config and
add up what a model built from it would contain. They exist separately because for a
mixture of experts they give different answers, **777,148,032** and **161,036,224**, and
the gap between them is the entire point of the architecture. The nested
`swiglu_feedforward_params()` helper in each counts one feed-forward block: three matrices,
not two, because SwiGLU has a gate.

`TrainConfig` holds the run: learning rates, warmup, batch size, accumulation, the
schedule.

## 4. The model itself

**`model.py`**, 908 lines, the only file where the architecture lives.

### `RMSNorm`

`forward()` divides by the root mean square of the activations and multiplies by a learned
gain. No mean subtraction and no bias, which is what separates it from LayerNorm. Cheaper,
and at this scale indistinguishable in quality.

### Rotary embeddings

`precompute_rotary_embeddings()` builds the cosine and sine tables once at startup for
every position up to the context length. `rotate_half()` swaps and negates the two halves
of a vector, and `apply_rotary_embeddings()` combines them into the rotation itself.

These are computed in **float32 and then cast back**. Doing the whole thing in bfloat16
lost enough precision at high positions that late-sequence tokens attended slightly wrongly.

### `MultiHeadLatentAttention`

This is the first of the two ideas. `__init__()` builds:

- `kv_down_projection`, hidden to **320 + 32**. The 320 is the compressed content latent
  shared by every head; the 32 is a rotary key, also shared.
- `query_down_projection` and `query_up_projection`, a low-rank path for queries.
- up-projections that reconstruct per-head keys and values from the latent when needed.

`forward()` does, in order:

1. Project the input to queries, then split each head's query into **content (64)** and
   **rotary (32)** parts. Rotate only the rotary part.
2. Project the input once to the shared `(320 + 32)` KV vector. Rotate only the 32.
3. If generating, concatenate this step's latent onto the cache. **The cache holds the
   320-wide latent and the 32-wide rotary key, not reconstructed keys and values.** That
   is where the memory saving is: 352 numbers per token instead of 1,792.
4. Reconstruct keys and values from the latent, concatenate content and rotary parts, and
   hand the result to `scaled_dot_product_attention`, which uses the fused FlashAttention
   kernel where it can.

Splitting content from position is not an optimisation, it is a necessity. RoPE rotates a
key by its position; a rotated key cannot be reconstructed from a latent that has no
position in it. So position is carried on its own small vector that is never compressed.

`absorb_matrices()` is the inference trick. The key up-projection can be folded into the
query projection and the value up-projection into the output projection, because both are
linear and the fold is mathematically exact. After absorption the model never reconstructs
keys and values at all: it attends directly against the cached latent.
`release_absorbed_matrices()` undoes it and `train()` calls that automatically, so a model
cannot accidentally be trained in the absorbed state.

### `DeepSeekMixtureOfExperts`

The second idea, and the subtler code in the file. `__init__()` allocates the experts as
**three stacked tensors** of shape `(32, hidden, ffn)`, one each for gate, up and down,
rather than as 32 separate modules.

`forward()`:

1. `router(tokens)` scores every token against all 32 experts; softmax in float32 for
   stability; `topk` takes the best 3; the three weights are renormalised to sum to 1.
2. **Capacity** is computed: `tokens x k / experts x capacity_factor`. Each expert accepts
   at most that many token-slots. During training a token beyond capacity is dropped for
   that layer, which is what makes the whole thing a fixed-shape tensor operation. At
   inference capacity is set to the full token count so nothing is ever dropped.
3. Assignments are turned into slot indices by a cumulative sum over a one-hot matrix. The
   one-hot is built by comparison against `arange` rather than with `F.one_hot`, because
   `F.one_hot` range-checks its input and that check forces a GPU-to-CPU synchronisation
   every layer of every step.
4. Tokens are scattered into a `(experts, capacity, hidden)` buffer with `index_add`.
   Dropped assignments all write into one scratch row that is discarded afterwards.
5. All 32 experts run as **three batched matrix multiplies**: `bmm` for gate, `bmm` for up,
   SiLU and multiply, `bmm` for down.
6. Outputs are gathered back to their tokens, weighted by the router probabilities, and
   summed. The shared expert runs over every token and is added on top.
7. The **load-balancing loss** is computed as the dot product of the hard fraction of slots
   each expert received with the soft mean router probability on that expert, scaled by
   expert count and weight 0.01. Minimising it pushes the router away from collapsing onto
   a few favourites and leaving most of the model idle.

The version of this that looped over experts in Python ran at **6,056 tokens per second**.
This one runs at **179,000**. Same mathematics, same results, thirty times the speed,
because the loop synchronised with the GPU thirty-two times per layer.

`stack_legacy_expert_weights()` and `convert_legacy_expert_keys()` exist so checkpoints
saved by the old per-module implementation still load.

### `TransformerBlock` and `GPT`

`TransformerBlock.forward()` is the standard pre-norm residual pair: normalise, attend,
add; normalise, feed-forward, add. The only variation is which feed-forward it holds.
Layer 0 gets a plain `SwiGLUFeedForward`; layers 1 to 13 get the mixture. Routing from
layer 0 destabilised early training, and one dense layer costs almost nothing.

`GPT.__init__()` builds the embedding, the 14 blocks, the final norm and the output
projection, then **ties** the output projection to the embedding matrix. One
`32,768 x 896` matrix does both jobs, saving 29M parameters.

`_initialize_weights()` scales the initialisation of residual-path projections by
`1/sqrt(2 x layers)`, so that the variance of the residual stream does not grow with depth.

`forward()` embeds, runs the blocks, normalises, projects to logits, and if targets were
passed computes cross-entropy plus the sum of every layer's load-balancing loss.

`generate_stream()` is the inference loop: run the prompt once to fill the cache, then one
token at a time, applying temperature, top-k and top-p, yielding each id as it is produced
and stopping on any of the stop tokens.

## 5. Pretraining

**`train.py`**

`load_random_batch()` memory-maps the `.bin` file and slices `batch_size` random windows of
`sequence_length + 1` tokens. Inputs are the first `n`, targets the last `n`. No shuffling
pass and no epoch bookkeeping: the file is so much larger than the run that uniform random
windows are equivalent and cost nothing.

`cosine_between()` and `learning_rate_at_step()` implement the schedule: linear warmup,
then a cosine decay from peak to minimum. The important detail is that progress is measured
as a **fraction of total training time, not of total steps**, which is what let the run
cross machines. Phase 2 was started with `--schedule-start 0.5227` and continued the same
curve rather than restarting it.

`estimate_loss()` runs a fixed number of no-grad batches on both splits with the model in
`eval()` mode, and is used for the checkpoint decision rather than the noisy training loss.

`main()` is the loop:

1. Initialise distributed training if there is more than one GPU (`ddp_utils.py`).
2. Build the model from `ModelConfig`, wrap it in `DistributedDataParallel` if needed.
3. Create AdamW with **two parameter groups**: weight decay on matrices, none on norms,
   biases and the embedding.
4. Per step, loop `gradient_accumulation_steps` times: forward under `bf16` autocast,
   divide the loss by the accumulation count, backward. Gradients accumulate in fp32
   master weights.
5. Clip gradients to norm 1.0, step the optimiser, zero the gradients.
6. Every `eval_interval` steps call `evaluate_and_maybe_save_best()`, which measures both
   losses, appends a line to the JSONL log via `append_log()`, and calls
   `save_checkpoint()` only if the validation loss improved.

`save_checkpoint()` writes through `checkpoint_utils.save_checkpoint_atomically()`, which
writes a temporary file and renames it, so an interrupted save cannot leave a half-written
checkpoint where a good one was.

## 6. Chat data

Three files cooperate here.

**`sft_data.py`** is storage plumbing. `convert_npz_to_npy()` turns compressed archives
into plain arrays, because `.npz` has to be decompressed into RAM in full while `.npy`
memory-maps. `_is_up_to_date()` compares modification times so the conversion only happens
once. `load_split()` and `save_split()` move packed splits around, and `describe()` prints
what a split contains.

**`persona_data.py`** generates the identity data, the reason the model can answer "who
made you". `conversation()` and `fill()` are small builders over templated turns.
`_single_exchange()` produces one question and answer about identity, capability or
ignorance. `_user_facts()` invents details for a user to state so the model learns to hold
them across turns. `generate_session()` composes several of these into one **multi-turn
conversation**, and `generate()` produces 140,000 of them with a fixed seed.

The multi-turn part was a correction. The first version emitted 20,000 short exchanges,
which packed into 650 rows out of 368,175 and would have shown the model each identity fact
about twice. Rebuilt as sessions, the same facts occupy 12,739 rows, **3.3% of the chat
tokens**.

**`chat_format.py`** owns the wire format. `ChatSpecialTokens` resolves the special token
ids from the tokenizer once. `build_prompt_token_ids()` renders a conversation to ids:

```
<|system|>system text<|user|>user text<|assistant|>assistant text<|endoftext|>
```

`stop_ids` returns both `<|endoftext|>` **and** `<|user|>`, because an undertrained model
that finishes an answer will happily carry on and write the user's next line.
`IncrementalTextDecoder` exists because a BPE token is not a character: `push()` buffers
ids until they decode to complete text, so streaming never emits a broken multi-byte
character.

**`sft_data.py`** also turns conversations into training rows. `encode_conversation()` walks the
turns, encodes each, and builds a **label array in parallel with the id array**, writing
`-100` for every position that belongs to a system or user turn. Cross-entropy ignores
`-100`, so the model is trained to produce assistant text and never to imitate the user.

`RowPacker` fills fixed 1,024-token rows with whole conversations, starting a new row only
when the next conversation will not fit, and `_grow()` doubles its buffer when it fills.
`pack_conversations()` drives the whole thing.

Packing rather than padding raised useful tokens per batch from roughly 40% to **61%**.
**60.7% of positions in the final set are supervised.**

## 7. Chat fine-tuning

**`sft_train.py`** mirrors `train.py` with three differences.

`load_random_batch()` selects random **rows** from the packed arrays instead of random
offsets into a stream, and returns labels alongside inputs, since the mask is per position.

The schedule runs from 2e-4 down to 2e-5, an order of magnitude below pretraining, over
100 warmup steps. `estimate_loss()` reports loss on assistant positions only, which is why
1.2484 is not comparable to the pretraining 2.976.

The mixture is 84% smol-smoltalk, 12% `everyday-conversations` **repeated 25 times**, and
3.3% persona data. The repeats are there because small talk is a rounding error in
smol-smoltalk, and without them the model answered "hi" with a lecture on quantum physics.

## 8. Measurement

**`evaluate_model.py`**

`as_choices()` turns a row of any supported benchmark into `(context, options, answer)`.
`sequence_logprob()` scores one continuation. `run_benchmark()` scores every option for
every question and takes the best by **total** log-probability and separately by
**length-normalised** log-probability, reporting both, since the two disagree in a way that
depends on how long the right answer happens to be.

`HuggingFaceAdapter` wraps a `transformers` model behind the same two methods the native
model exposes, so **the reference models are scored by exactly this code on exactly these
examples**. Numbers copied from other people's model cards would compare harnesses as much
as models.

`held_out_perplexity()` measures on a FineWeb-Edu shard no phase trained on.
`inference_speed()` measures generation throughput and time to first token.

Results are written to `docs/eval.json`, which the About page in the chat interface reads
directly, so what is on screen is what was last measured.

## 9. Serving and export

**`server.py`** is FastAPI plus SQLite. `ModelRunner` loads a checkpoint, holds the
tokenizer, and reloads on request when the checkpoint picker changes. `/api/chat` streams
tokens over server-sent events, pushing each id through an `IncrementalTextDecoder` and
checking a per-generation `threading.Event` so a stop request takes effect immediately.
Chats, messages, archive and pin flags live in SQLite; `ensure_schema()` adds columns in
place so an old database keeps working.

**`checkpoint_utils.py`** is shared by everything that touches a `.pt` file.
`normalize_state_dict_keys()` strips `module.` and `_orig_mod.` prefixes left by
`DistributedDataParallel` and `torch.compile`. `load_for_inference()` rebuilds the config
from the checkpoint, constructs the model, loads the weights and puts it in eval mode.
`strip_optimizer_state()` drops the optimiser tensors, which are two thirds of the file
size and useless for inference.

**`export_to_huggingface.py`** writes a `transformers`-loadable repository: config,
modelling code, tokenizer and safetensors. It carries a fix worth stating, because the
failure was silent. RoPE tables are `persistent=False` buffers, so they are not in the
checkpoint, and `transformers` **fills any tensor it does not find with zeros** instead of
running `__init__`. Zero cosine and sine erase position entirely: the exported model
answered `"made made made made"` while the same weights answered correctly through the
native path. The exporter now marks those buffers persistent and writes them into the
safetensors file, so a regression shows up as missing keys rather than as nonsense.

**`cloud_run.py`** defines the Modal app. Each stage is a function with a GPU type, a
timeout and a cost estimate, and the module **refuses to import** if any account's
worst-case bill exceeds its cap. Jobs are launched with `.spawn()` rather than `.remote()`,
because a blocking call keeps a client connection open and the job dies with the terminal;
one run was lost that way 1,730 steps in.

---

## Reading the code in order

```
config.py            the numbers
   ↓
model.py             the architecture
   ↓
train.py             pretraining
   ↓
sft_data.py          how a conversation becomes a row
   ↓
sft_train.py         chat tuning
   ↓
evaluate_model.py    how the numbers were measured
   ↓
server.py            how it is served

cloud_run.py         how all of the above was run on rented GPUs
```

`training/tests/verify_all.py` runs 40 checks on the model and data path, and
`verify_cloud_run.py` runs 17 on the training plan and the budget, before anything paid
starts.
