# escha-tp-fix

Two-line patch that enables `--tp-size N --ep-size N` (tensor + expert parallelism) on the [Escha runtime](https://huggingface.co/EschaLabs/Qwen3.6-35B-A3B-Escha-W2) for hybrid Qwen3.5 / Qwen3.6 MoE models quantized with `eschamoe` 2-bit.

Without this patch, `--tp-size 2 --ep-size 2` starts cleanly but returns silently corrupted outputs — arithmetic drift (12·17 → 191), degenerate loops, `<think>` blocks that never close.
`--dp-size N` was the only working multi-GPU config.

## Why

On 2×3090, TP=2+EP=2 gives **~5× the KV cache pool** of DP=2 (1.36 M vs ~340 K tokens per replica), at ~10 % single-request latency cost (all-reduce over PCIe).
Decisive for long-context / agentic workloads where the max context per request is bound by one replica's leftover VRAM under DP.

## The two bugs (in `sglang/srt/models/qwen3_5.py` inside `escha-*.whl`)

1. **eschamoe checkpoint loader ignores the EP rank offset.**
   The manual eschamoe branch in `load_weights` falls into a shape-mismatch `else` written for mixed-bit K3-vs-K2 cases and does `_qp[_pn].data = _w.to(...)` without slicing on dim 0.
   Under `ep_size > 1`, both ranks load experts `[0:num_local)`; experts `[num_local:num_experts)` are never loaded on any rank.
   The `topk_ids < 0 → weight = 0` mask silently zeroes 4/8 top-k contributions per token → consistent 5-15 % numerical drift.

2. **`SGLANG_INT8_EMBED` install ignores TP sharding.**
   Registers the full `[vocab, hidden]` int8 tensor as a buffer into a `VocabParallelEmbedding` whose `.weight` was preallocated as `[vocab/tp, hidden]`.
   Every rank indexes a full-vocab table with local row indices → garbage embeddings → immediate degeneration.

Both fixes are Python-only (no kernel changes) and only trigger under the conditions that were broken, so they're non-invasive for DP / single-GPU users.

## Apply

```bash
cd /path/to/escha-runtime/.venv/lib/python3.12/site-packages/sglang/srt/models
cp -n qwen3_5.py qwen3_5.py.bak
patch -p0 < /path/to/escha-tp-fix/patches/qwen3_5.py.tp-ep-fix.diff
```

Then launch with `--tp-size 2 --ep-size 2`.
See `REPRO.md` for the full bug-reproduction + fix-verification procedure.

## Bisect summary

Prompt: `"What is 12*17? Think briefly."`, greedy.
Baseline (DP=1): `204` ✓.

| Config | Result |
|---|---|
| TP=2 EP=2 (upstream) | garbage (`191`, chinese loops) |
| ... with all int8 OFF | still garbage → **not the int8 installs** |
| ... with `--enable-dp-attention` | still garbage → **not the attn shard** |
| ... with `GRAPHS=0` | still garbage → **not CUDA-graph capture** |
| ... with `ESCHA_FORCE_SORTED_MOE=1` | still garbage → **not the fused MoE kernel** |
| **+ EP-slice fix, int8 OFF** | `204` ✓ |
| ... + `int8_embed=1` | garbage again → **second bug** |
| **+ EP-slice fix + embed-slice fix, all int8 ON** | `204` ✓ |

## Scope

- `Qwen3_5MoeForConditionalGeneration` (30 GDN + 10 attn hybrid, 256 experts, top-k 8, 2-bit `eschamoe`).
- Only `ep_size == tp_size` works (whole-expert distribution); pure TP with `ep_size=1` still fails because the precompiled eschamoe kernel doesn't accept sharding *inside* an expert.
- Tested: 2×RTX 3090, `escha-1.0.2+qwen3moe`, PyTorch 2.9 + CUDA 12.x, Python 3.12.

## License

Apache-2.0, matching upstream Escha.

## Links

- Model: <https://huggingface.co/EschaLabs/Qwen3.6-35B-A3B-Escha-W2>
- Upstream sglang: <https://github.com/sgl-project/sglang> (the `sglang` inside the Escha wheel is a private fork — the bugs are in the Escha-specific code, not upstream sglang)
