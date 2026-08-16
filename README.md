# escha-tp-fix

Enables **`--tp-size N --ep-size N`** (tensor + expert parallelism) on the [Escha runtime](https://huggingface.co/EschaLabs/Qwen3.6-35B-A3B-Escha-W2) for hybrid Qwen3.5 / Qwen3.6 MoE models quantized with `eschamoe` 2-bit.

Two bugs in the shipped wheel's `sglang.srt.models.qwen3_5` prevented multi-GPU tensor+expert parallelism from producing correct outputs.
With both fixes applied, `--tp-size 2 --ep-size 2` on 2×3090 produces the same greedy tokens as the DP=1 baseline and gives ~5× the total KV cache pool.

## Background

The Escha runtime ships a heavily patched `sglang` fork inside a single precompiled wheel (`escha-*.whl`).
It supports the `eschamoe` 2-bit expert quantization used by [`EschaLabs/Qwen3.6-35B-A3B-Escha-W2`](https://huggingface.co/EschaLabs/Qwen3.6-35B-A3B-Escha-W2) and other Escha releases.

- Model: **Qwen3.6-35B-A3B-Escha-W2** — 40-layer hybrid (30 GatedDeltaNet mamba + 10 full-attention), 256 routed experts, top-k 8, `moe_intermediate_size=512`, 2-bit MoE weights via `eschamoe`.
- Before this patch, the only working multi-GPU config was **`--dp-size 2`** (full model replicated on each GPU).
- `--tp-size 2 --ep-size 2` **appeared to start** but returned silently corrupted outputs: arithmetic drift (12·17 → 191 / 196 / 199), degenerate loops with non-English fragments (`本发明的\n手上的user`), infinite `<think>` blocks that never close.

## What breaks — the two bugs

### Bug 1: eschamoe checkpoint loader ignores the EP rank offset

`load_weights()` in `qwen3_5.py` has a special path for eschamoe expert code tensors that bypasses sglang's standard `expert_params_mapping`.
Under EP `>1`, the model parameter `.data` is preallocated with shape `[num_local_experts, ...]` (e.g. 128 on each rank when `num_experts=256` and `ep_size=2`), but the checkpoint tensor stacks **all** experts (`[256, ...]`).
When the shapes don't match, the loader falls into an `else` branch that was written for a different case (mixed-bit K3 vs K2 last-dim mismatch) and simply **replaces the parameter's storage with the full tensor**:

```python
_qp[_pn].data = _w.to(_qp[_pn].data.device, _qp[_pn].data.dtype)
```

`PackedQwen35MoeExperts.__init__` (built later, in `process_weights_after_loading`) then iterates `for e in range(num_local_experts)` — **so both ranks pick up experts `[0:128]` and experts `[128:256]` are never loaded on any rank.**

The eschamoe apply step remaps non-local expert IDs to `-1` via the `StandardDispatcher`, and my earlier mask (`weight = 0 for topk_ids < 0`) turned those into no-op contributions.
So `routing_prob(expert)` for `expert ∈ [128, 256)` never contributed to the token output — **half of the top-k routing was silently zeroed**, producing consistent numerical drift.

**Fix**: slice `_w` by EP rank *before* the shape check:

```python
if (_ep_size > 1 and _w.dim() >= 1
        and _w.shape[0] != _qp[_pn].data.shape[0]
        and _w.shape[0] == _qp[_pn].data.shape[0] * _ep_size):
    _num_local = _qp[_pn].data.shape[0]
    _w = _w[_ep_rank * _num_local:(_ep_rank + 1) * _num_local]
```

### Bug 2: `SGLANG_INT8_EMBED` install ignores TP sharding

The 24 GB auto-enable path (`SGLANG_INT8_EMBED=1` or VRAM ≤ `SGLANG_INT8_AUTO_VRAM_GB`) keeps `embed_tokens` in int8 and registers the raw checkpoint tensor as a buffer:

```python
_emb = self.model.embed_tokens
del _emb.weight
_emb.register_buffer("weight_int8", _ibuf[_b]["i"].to(_dev).contiguous(), ...)
```

`self.model.embed_tokens` is a `VocabParallelEmbedding` — under TP `>1` its `.weight` was preallocated with shape `[vocab_per_rank, hidden]` (e.g. `[124160, 2048]` on each rank when `vocab=248320`, `tp=2`).
The int8 checkpoint tensor has the **full** `[248320, hidden]` shape, so the raw registration leaves every rank pointing at a full-vocab table indexed with **local** row indices.
The lookup returns garbage rows for most tokens → embeddings incoherent from step 0 → output degenerates within the first few tokens.

**Fix**: detect the `vocab_full = vocab_per_rank * tp_size` case and slice by TP rank *before* registering:

```python
_emb_i = _ibuf[_b]["i"]
_emb_s = _ibuf[_b]["s"]
_local_vocab = _emb.weight.shape[0]
if _emb_i.shape[0] != _local_vocab:
    _tp_size = _emb_i.shape[0] // _local_vocab
    _tp_rank = get_tensor_model_parallel_rank()  # or _emb.tp_rank
    _emb_i = _emb_i[_tp_rank * _local_vocab:(_tp_rank + 1) * _local_vocab]
    _emb_s = _emb_s[_tp_rank * _local_vocab:(_tp_rank + 1) * _local_vocab]
```

`lm_head` did **not** exhibit the same issue in our bisect — its int8 install path either replicates the full vocab or is served by a kernel path that indexes it globally.
Only the embed lookup was broken.

## Verification (bisect summary)

Prompt: `"What is 12*17? Think briefly."`, greedy (`temperature=0`).
Baseline (DP=1 single 3090): `12·17 = 204`, coherent thinking chain that closes `</think>` cleanly.

| Config | Output | Verdict |
|---|---|---|
| DP=1 baseline | `204` ✓ | correct |
| TP=2 EP=2, int8 default, fp8 KV | `191`, infinite `<think>` | **broken** (upstream) |
| TP=2 EP=2, int8 default, fp16 KV | same | fp8 KV *not* the cause |
| TP=2 EP=2, all int8 OFF | `199` / `144` | **broken** — not caused by the int8 installs |
| TP=2 EP=2, `--enable-dp-attention` | still wrong | attention shard not the cause |
| TP=2 EP=2, `GRAPHS=0` | still wrong | CUDA graph capture not the cause |
| TP=2 EP=2, `ESCHA_FORCE_SORTED_MOE=1` | still wrong | fused MoE kernel not the cause |
| **After EP-slice fix, int8 OFF** | `204` ✓ | ✓ |
| After EP-slice fix + `int8_attn=1` | `204` ✓ | attn shard OK |
| After EP-slice fix + `int8_gdn=1` | `204` ✓ | GDN shard OK |
| After EP-slice fix + `int8_shexp=1` | `204` ✓ | shared-expert shard OK |
| After EP-slice fix + `int8_lm_head=1` | `204` ✓ | lm_head OK |
| After EP-slice fix + `int8_embed=1` | garbage loop | **second bug** — embed |
| **After both fixes + all int8 ON** | `204` ✓ | ✓ |

## Performance (2×3090, no NVLink)

Single-user greedy decode, 256-token completions, prompt ~50 tokens:

| Config | Latency (single) | KV pool total | Best for |
|---|---|---|---|
| DP=2 (upstream) | ~116 tok/s | ~340K tokens per replica | Multi-user throughput (scales 2× linearly) |
| **TP=2 + EP=2 (this patch)** | ~105 tok/s | **~1.36M tokens** | Long contexts (>100K), or when you want intra-request parallelism |

The ~10 % single-request latency cost is the per-layer all-reduce paid over PCIe (no NVLink between the two 3090s).
With NVLink the gap disappears.
The KV pool win is decisive when serving long contexts on the same card class.

## Apply the fix

The patch targets `sglang/srt/models/qwen3_5.py` inside the Escha wheel's install location.

```bash
cd /path/to/escha-runtime/.venv/lib/python3.12/site-packages/sglang/srt/models
cp qwen3_5.py qwen3_5.py.bak
patch -p0 < path/to/escha-tp-fix/patches/qwen3_5.py.tp-ep-fix.diff
```

Then launch with `--tp-size 2 --ep-size 2` (see `REPRO.md` for a full-runtime reproduction).

The patch is **compatible with all existing single-GPU / DP configurations** — the sliced code paths only trigger when `ep_size > 1` (bug 1) or when the checkpoint tensor's dim 0 is a multiple of the local vocab (bug 2).

## Scope and limitations

- Applies to `Qwen3_5MoeForConditionalGeneration` (hybrid 30 GDN + 10 attn architecture with 256 routed experts + `eschamoe` 2-bit quant).
- Does **not** enable `--tp-size N --ep-size 1` (pure TP, no EP): the precompiled `eschamoe` CUDA kernel doesn't accept sharding *inside* an expert. Only `ep_size == tp_size` (whole-expert distribution) works.
- Tested on 2×RTX 3090, sglang fork `escha-1.0.2+qwen3moe`, PyTorch 2.9 + CUDA 12.x, Python 3.12.
- The patch is model-agnostic in principle: any hybrid Qwen3.5 model that inherits the same `load_weights` path in `qwen3_5.py` benefits from bug 1 under EP, and any TP-sharded `VocabParallelEmbedding` with int8 embed enabled benefits from bug 2.

## License

Apache-2.0, matching the upstream Escha runtime license.

## Related

- Escha model: <https://huggingface.co/EschaLabs/Qwen3.6-35B-A3B-Escha-W2>
- Upstream sglang: <https://github.com/sgl-project/sglang> (unrelated to this patch — the `sglang` inside the Escha wheel is a private fork with the eschamoe integration; the bugs live in the Escha-specific code, not upstream sglang)
