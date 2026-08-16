# Reproducing the bug and the fix

Minimum hardware: 2 GPUs with combined VRAM ≥ 32 GB (tested on 2×RTX 3090 24 GB).

## 0. Prerequisites

- Escha runtime installed per the wheel's `INSTALL.md` (Linux x86-64, Python 3.12, PyTorch 2.9 + CUDA 12.x).
- Model downloaded: `EschaLabs/Qwen3.6-35B-A3B-Escha-W2` in a flat folder.
- `sglang/serve.sh` from the Escha runtime available.

## 1. Confirm the DP=1 baseline is correct

```bash
MODEL=/path/to/Qwen3.6-35B-A3B-Escha-W2 bash sglang/serve.sh --kv-cache-dtype fp8_e5m2
# In a second terminal:
curl -s http://127.0.0.1:30000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"escha-qwen36-35b-a3b-w2",
       "messages":[{"role":"user","content":"What is 12*17? Think briefly."}],
       "max_tokens":256,"temperature":0}' \
  | jq -r '.choices[0].reasoning_content, .choices[0].message.content'
```

Expected: coherent reasoning containing `204`, and `content: "204"` (or similar).
This confirms your DP=1 install works.

## 2. Reproduce the TP=2+EP=2 corruption (upstream, no patch)

Same model, two GPUs:

```bash
MODEL=/path/to/Qwen3.6-35B-A3B-Escha-W2 \
CUDA_VISIBLE_DEVICES=0,1 \
bash sglang/serve.sh \
  --tp-size 2 --ep-size 2 \
  --kv-cache-dtype fp8_e5m2 \
  --disable-piecewise-cuda-graph
```

Re-hit the same `/v1/chat/completions` request.
You will see one of these silent-failure signatures:

- Arithmetic drift: `12*17 = 191` (or `196`, `199`, `144`, etc — deterministic per run but wrong).
- `reasoning_content` degenerates into a loop of `本发明的\n手上的user\nWhat is 12*12?...` after a few tokens (int8 embed corruption).
- `<think>` block never closes and `content` stays `null`.
- Raw `/v1/completions` on the same server also drifts.

## 3. Apply the fix

Stop the server (`pkill -f sglang.launch_server`).
Then patch:

```bash
cd /path/to/escha-runtime/.venv/lib/python3.12/site-packages/sglang/srt/models
cp -n qwen3_5.py qwen3_5.py.bak    # keep a backup, first time only
patch -p0 < /path/to/escha-tp-fix/patches/qwen3_5.py.tp-ep-fix.diff
```

## 4. Confirm the fix

Relaunch with the exact same TP=2+EP=2 command, hit the same request.
Expected: `content: "204"` with a coherent reasoning chain — matches the DP=1 baseline.

You should also see two log lines confirming both fixes activated:

```
eschamoe: loaded 480 expert escha tensors           # on each rank
SGLANG_INT8_EMBED: embed_tokens kept int8 (124160 x 2048), dequantized per-row at lookup
                                        # 124160 = vocab_per_rank, NOT 248320 = full_vocab
```

If you see `(248320 x ...)` on either rank instead of `(124160 x ...)`, the embed slice didn't apply — check that your patch landed cleanly.

## 5. Verify KV pool size increase

The TP=2+EP=2 config gives ~5× the KV pool of DP=2 (freed by halved weight footprint per rank).
Grep the launch log for:

```
KV Cache is allocated. #tokens: 1359159, K size: 6.48 GB, V size: 6.48 GB
max_total_num_tokens=1359159
```

## 6. Regression: DP=1 still works

Sanity-check the patch is non-invasive by re-running step 1.
The DP=1 baseline should still return `204`.
Both patched code paths only trigger when `ep_size > 1` (bug 1) or when the checkpoint tensor is `tp_size ×` larger than the local param (bug 2), so DP=1 falls through unchanged.

## Micro-repro of Bug 1 in isolation (optional)

If you want to bisect the eschamoe EP loader without spinning up the full model:

- Add a `print` in `load_weights` right after the `_qp[_pn].data = _w.to(...)` fallback branch, printing `_pn`, `_w.shape`, `_qp[_pn].data.shape`, and the current EP rank.
- Run with `--tp-size 2 --ep-size 2` and grep the ranks' logs — before the fix you'll see `_w.shape=(256, ...)` and `_qp[_pn].data.shape=(128, ...)` on both ranks; after the fix the slice runs and `_w.shape=(128, ...)` matches per-rank.
