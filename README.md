# coconut — TRM-SVFT: recursive steering adapters in a frozen LLM

Train an LLM to reason without words. Instead of generating chain-of-thought
tokens, the model thinks in continuous vectors inside its own layers.

This repo: a frozen `Qwen3-0.6B-Math-Expert` gets small adapters on its linear
layers. Inside each adapter sits a tiny recurrent module (the Tiny Recursive
Model pattern): two latent states `zL`/`zH` are refined over several
weight-shared cycles, and the result modifies the frozen layer's output.
Only the adapters train; the loss is answer-token NLL on GSM8k, with latent
"thought" tokens masked out.

## Did it work?

Partially. It learned math a bit better than chance, then overfit.

On the GSM-mini harness (84-question greedy eval subset of GSM8k train,
`outputs/*/terminal.log`, `eval/acc` lines):

| model | eval/acc (peak) |
|---|---|
| base model, untrained adapter (epoch −1) | 0.00 (0/84) |
| TRM-SVFT, canonical run `trmsvft-qwen3-0.6b_20251105-185222`, epoch 15 | **0.095** (8/84) |
| same run, epochs 16–28 | collapses to 0.01–0.04 as train loss → 3e-5 |

So: **0% → ~10% peak on GSM-mini, then it memorizes the 10k train set.**
`eval/loss` kept dropping (1.18 → 0.06) even as accuracy collapsed — loss and
accuracy tell different stories, classic overfit. An earlier hybrid variant
(TRMLoRA + coconut curriculum, `trmsvft-qwen3-0.6b_20251031-090744`) peaked at
0.205 on the same-style eval, but that run mixed in other changes, so treat it
as a hint, not a result.

## How it works (pseudocode)

**Before: a plain SVFT/LoRA-style adapter.** Each adapted frozen layer gets a
rank-`r` SVD decomposition, and training only rescales the singular values:

```py
# frozen decomposition of an adapted Qwen matrix
W ≈ U @ diag(S) @ V.T + W_res        # U, S, V, W_res all frozen

# a normal adapter: one shot, no state
scale ← head(x @ V)                  # tiny learned vector
y ← frozen_linear(x) + apply_svd_adapter(x, U, transform(S, scale), V, W_res)
```

**After: TRM recursion inside the adapter.** The singular-value scale is no
longer computed in one shot — it comes from a small recurrent state that is
refined over several weight-shared cycles (canonical version, commit `2c99374`):

```py
xV ← x @ V                          # project layer input into rank-r SVD space
zL ← cache.zL or learned_zL         # two latent states, b r
zH ← cache.zH or learned_zH

xV ← fold_sequence_into_batch(xV)   # canonical code treats token positions independently
zL, zH ← repeat_per_token(zL, zH)

def recurse(xV, zL, zH):
    for _ in range(l_cycles):
        zL ← L_net(zL, xV + zH)     # L_net: small weight-shared transformer block
    zH ← L_net(zH, zL)
    return zL, zH

with no_grad():
    for _ in range(h_cycles - 1):
        zL, zH ← recurse(xV, zL, zH)   # early cycles detached
zL, zH ← recurse(xV, zL, zH)           # only the final cycle gets gradients

scale ← output_head(zH)
S_eff ← transform(S, scale)
y ← frozen_linear(x) + apply_svd_adapter(x, U, S_eff, V, W_res)
cache.zL, cache.zH ← last_token(zL, zH)   # persists across autoregressive steps
```

Properties worth knowing:

- `zL`/`zH` live in the adapter rank (64–128), not Qwen's 2560-dim residual stream.
- One shared `L_net` is reused across all cycles and all adapted layers.
- The output modifies real frozen-Qwen layers; there is no prediction-only
  side channel the base model can ignore.
- Loss is answer-token NLL; latent tokens are masked.

---

# Messy notes below — legacy / WIP material, read with care

Everything from here down is rough working notes and leftovers from older
branches. Parts may not apply to the TRM-SVFT branch above.

## Branch map

| branch | what it is | did it work? |
|---|---|---|
| `main` | TRM-SVFT recursive steering adapters (this) | partially: 0% → ~10% peak on GSM-mini, then overfits |
| [`seq`](../../tree/seq) | first committed plain replication of the COCONUT paper (latent-token CoT) | replication target, see its journal |
| [`seq-ocr`](../../tree/seq-ocr) | uncommitted WIP hacking coconut into an OCR proof-of-concept | unknown / unfinished |
| [`wip-trm-seq`](../../tree/wip-trm-seq) | newest WIP: recursion over the whole sequence instead of per-token | self-labeled "wrong track" in its own commit message; kept for reference |

## Run

Configs and run targets may lag the code.

```bash
just install          # uv: flash-attn, pytorch, etc.
uv run scripts/check_gpu.py
just coconut-svft     # train TRM-SVFT on GSM8k
just coconut-svft-eval
```

## Old readme: COCONUT experiments (older branches)

Code for the paper ["Training Large Language Model to Reason in a Continuous Latent Space"](https://arxiv.org/abs/2412.06769).

LLMs usually reason by producing [chain-of-thought](https://arxiv.org/abs/2201.11903). But the most efficient part of the language might be the text it generates. Humans instead reason in a latent space, so why can't we get AI to do the same? If we can, we might be able to learn a super language, plan and more.

The COCONUT paper solves this. I gave this my own shot, using SFT with tensors instead of LoRA.

### Setup

```bash
just install
uv run scripts/check_gpu.py
```

### Results

**ICOCONUT PROSOCIAL**

Same method as the original paper except with PROSOCIAL instead of GSM8k.
More info in the [research journal](mjc_research_journal.md#icoconut-2025-04-30-0500).

<img src="docs/prosocial/uv_sanity.png" alt="ico" width="300">

### Eval commands

```bash
# evaluate GSM8k
just eval gsm_icoconut
just eval gsm_coconut
just eval gsm_cot

# evaluate prosocial
just eval prosocial_icoconut
just eval prosocial_coconut
just eval prosocial_cot
```

### Training commands

```bash
# TRAININGS
# gsm8k runs
uv run coconut.py gsm_coconut.yaml
uv run coconut.py gsm_icoconut.yaml
uv run coconut.py gsm_cot.yaml

# prosocial runs
uv run coconut.py prosocial_coconut.yaml
uv run coconut.py prosocial_icoconut.yaml
uv run coconut.py prosocial_cot.yaml
```
