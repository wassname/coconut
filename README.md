# coconut — TRM-SVFT: recursive steering adapters in a frozen LLM

Train an LLM to reason without words. Instead of generating chain-of-thought
tokens, the model thinks in continuous vectors inside its own layers.

This branch (`adapter_recurse4_simpler`, now merged to `main`) does that with
**recursive steering adapters**: each adapted layer of a frozen
Qwen3-0.6B-Math-Expert gets a small recurrent module that refines two latent
states (`zL`, `zH`) in a rank-64 SVD subspace of that layer's weight matrix,
then writes the result back as a modification of the frozen layer's output.
The recursion scheme follows the Tiny Recursive Model (TRM) pattern: many
weight-shared refinement cycles, all but the last one run under `no_grad`.

- Model: frozen `Qwen3-0.6B-Math-Expert`; only the adapters train (~small % of params).
- Data: GSM8k → `di-zhang-fdu/GSM8K_RL_train` (10k), loss on answer tokens only
  (latent "thought" tokens are masked out of the loss).
- Entry points: `just coconut-svft` (train), `just coconut-svft-eval` (eval).
  Configs are TOML in `configs/`; training logs land in `outputs/<run>/terminal.log`.
- Full experiment log: `mjc_research_journal.md`.

## Did it work?

Partially. It learned math a bit better than chance, then overfit.

On the GSM-mini harness (84-question greedy eval subset of the GSM8k train
split, `outputs/*/terminal.log`, `eval/acc` lines):

| model | eval/acc (peak) |
|---|---|
| base model, untrained adapter (epoch −1) | 0.00 (0/84) |
| TRM-SVFT, canonical run `trmsvft-qwen3-0.6b_20251105-185222`, epoch 15 | **0.095** (8/84) |
| same run, epochs 16–28 | collapses to 0.01–0.04 as train loss → 3e-5 |

So: **0% → ~10% peak on GSM-mini, then memorizes the 10k train set.**
`eval/loss` kept dropping (1.18 → 0.06) even as accuracy collapsed, so the
loss and the accuracy tell different stories — classic overfit, visible in
the terminal log of the run above. An earlier hybrid variant
(TRMLoRA adapters + coconut curriculum, `trmsvft-qwen3-0.6b_20251031-090744`)
peaked at 0.205 on the same-style eval, but that run mixed in other changes,
so treat it as a hint, not a result.

No notebook contains a scored eval or a compute-properties demo — the
notebooks (`scripts/`) are qualitative sample dumps and KV-cache shape
debugging. The numbers live in `outputs/*/terminal.log` and the journal.

## How it works (pseudocode)

Canonical version, commit `2c99374`. For each adapted frozen linear layer:

```py
# frozen decomposition of an adapted Qwen matrix
W ≈ U @ diag(S) @ V.T + W_res

# ── initialize or retrieve recurrent state ─────────────
xV ← x @ V                          # project layer input into rank-r SVD space
zL ← cache.zL or learned_zL         # latent states, b r
zH ← cache.zH or learned_zH

xV ← fold_sequence_into_batch(xV)   # canonical code treats token positions independently
zL, zH ← repeat_per_token(zL, zH)

# ── shared recursive computation ───────────────────────
def recurse(xV, zL, zH):
    for _ in range(l_cycles):
        zL ← L_net(zL, xV + zH)     # L_net: small weight-shared transformer block
    zH ← L_net(zH, zL)
    return zL, zH

with no_grad():
    for _ in range(h_cycles - 1):
        zL, zH ← recurse(xV, zL, zH)   # early cycles detached
zL, zH ← recurse(xV, zL, zH)           # only final cycle gets gradients

# ── write back into the frozen layer ───────────────────
scale ← output_head(zH)
S_eff ← transform(S, scale)            # rescale the singular values
δy ← apply_svd_adapter(x, U, S_eff, V, W_res)
y ← frozen_linear(x) + δy
cache.zL, cache.zH ← last_token(zL, zH)   # persists across autoregressive steps
```

Properties worth knowing:

- `zL`/`zH` live in the adapter rank (64–128), not Qwen's 2560-dim residual stream.
- One shared `L_net` is reused across all cycles and all adapted layers.
- The output modifies real frozen-Qwen layers; there is no prediction-only
  side channel the base model can ignore.
- Loss is answer-token NLL; latent tokens are masked.

## Branch map

| branch | what it is | did it work? |
|---|---|---|
| `main` | this — TRM-SVFT recursive steering adapters | partially: 0% → ~10% peak on GSM-mini, then overfits |
| [`seq`](../../tree/seq) | first committed plain replication of the COCONUT paper (latent-token CoT) | replication target, see its journal |
| [`seq-ocr`](../../tree/seq-ocr) | uncommitted WIP hacking coconut into an OCR proof-of-concept | unknown / unfinished |
| [`wip-trm-seq`](../../tree/wip-trm-seq) | newest WIP: recursion over the whole sequence instead of per-token | self-labeled "wrong track" in its own commit message; kept for reference |

## Setup

```bash
just install          # uv: flash-attn, pytorch, etc.
uv run scripts/check_gpu.py
```

Checkpoints save under `outputs/<run>/`. `just weights` pulls the base model.

---

merged branch notes and README rewrite by claude (pi), 2026-08; pseudocode
section preserved verbatim-ish from wassname's summary for the record.
