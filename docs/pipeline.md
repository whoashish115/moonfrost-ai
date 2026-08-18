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
