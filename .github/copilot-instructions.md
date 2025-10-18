# Copilot Instructions: COCONUT + TRM

## Project Overview

This is a research implementation combining two approaches for latent reasoning in LLMs:
- **COCONUT** (Training LLMs to Reason in Continuous Latent Space): Uses special `<|latent|>` tokens for internal reasoning
- **TRM** (Tiny Recursive Models): Adds a small recursive adapter that iteratively refines latent representations with a frozen, quantized LLM

The core idea: frozen 4-bit LLM handles perception/generation, while a tiny trainable TRM (~7M params) performs recursive refinement on latent tokens.

## Architecture

### Special Tokens System
Three special tokens enable latent reasoning:
- `<|start-latent|>`: Marks beginning of latent reasoning block
- `<|latent|>`: Placeholder tokens for internal computation (not decoded to text)
- `<|end-latent|>`: Marks end of latent block

Example: `"The capital of France is <start-latent> <latent> <latent> ... <end-latent> Paris."`

### Two Operating Modes

**COCONUT mode** (`use_trm=False`, default):
- Trainable LLM learns to use latent tokens for reasoning
- Hidden states at latent positions replaced via `hs2ie()` (hidden-state-to-input-embedding)
- Multiple replacement methods in `coconut/hs2ie.py`: `"supressed[0.75:]"`, `"ie+supressed[0.5:]"`, etc.

**TRM mode** (`use_trm=True`):
- Frozen 4-bit LLM + trainable TRM adapter (`coconut/trm_adapter.py`)
- TRM has dual networks (L_net, H_net) that recurse hierarchically
- Detached recursions: early passes use `torch.no_grad()`, only last N passes backprop
- See `CoconutTRM.hrm()` for recursive refinement matching paper pseudocode

## Key Files

- `coconut/coconut.py`: Main model wrapper, handles multi-pass latent token processing
- `coconut/trm_adapter.py`: TRM recursive reasoning module (L_net, H_net, transcoder)
- `coconut/configs.py`: All experiment configs as Pydantic dataclasses (TRM, Debug, etc.)
- `coconut/dataset.py`: Tokenizes GSM8K with latent tokens, handles staged training
- `coconut/hs2ie.py`: Hidden-state replacement methods, including novel "suppressed activations" experiment
- `coconut/vcr_loss.py`: SEQ-VCR loss for intermediate state stability (experimental)
- `scripts/run.py`: Training loop with staged curriculum (CoT → 1 latent → 2 latents → ...)

## Training Workflow

### Staged Curriculum
Training progresses through stages, gradually increasing latent tokens:
- **Stage -1** (epochs 0-2): Pure CoT with `<start-latent>` and `<end-latent>` but no `<latent>` tokens
- **Stage 0+**: Add `c_thought` latent tokens per reasoning step, up to `max_latent_stage`

Controlled by: `cot_epochs`, `epochs_per_stage`, `max_latent_stage` in configs.

### Running Experiments
```bash
# Use justfile recipes
just run_smol          # Standard COCONUT training
just run_trm           # TRM mode with frozen LLM

# Or directly with tyro CLI
uv run python scripts/run.py TRM --batch_size_training=8 --max_size=10000
uv run python scripts/run.py Debug  # Fast iteration with tiny model
```

Configs are Pydantic dataclasses in `coconut/configs.py`. CLI args override config fields via tyro.

### Key Config Parameters
- `n_detached_recursions`: Number of early passes to detach gradients (TRM-style)
- `replacement_method`: How to inject latent hidden states (`"supressed[0.75:]"` uses top 25% layers)
- `loss_seq_vcr`: Enable experimental VCR loss for latent stability
- `use_position_ids`: Add positional encoding to latent tokens
- `trm_n_sup`, `trm_num_layers`, `trm_num_heads`: TRM architecture params

## Development Patterns

### Type Annotations
Use jaxtyping for tensor shapes:
```python
from jaxtyping import Float, Int
from torch import Tensor

def forward(self, hs: Float[Tensor, 'b t h']) -> Float[Tensor, 'b t h']:
    # b=batch, t=sequence, h=hidden_dim
```

### Logging
Use loguru, not print:
```python
from loguru import logger
logger.info("Training stage {}", stage)
logger.debug("Hidden states shape: {}", hs.shape)
```

### Caching
Use anycache for expensive preprocessing:
```python
from anycache import anycache

@anycache('.anycache')
def get_dataset(path, tokenizer, ...):
    # Expensive tokenization cached to disk
```

### Code Markers
- `TODO`: Incomplete implementation or decision needed
- `FIXME`: Known bug or suboptimal code
- `HACK`: Temporary workaround

Current TODOs in `coconut/trm_adapter.py` and `coconut/hs2ie.py` relate to context pooling and cache invalidation.

## Common Pitfalls

1. **Gradient flow in TRM mode**: LLM is frozen but `lm_head` must stay trainable for loss backprop. See `coconut/coconut.py:__init__()`.

2. **Multi-pass processing**: `coconut.py` processes input in multiple passes (one per latent token). KV cache carries over between passes. Don't assume single forward pass.

3. **Special token IDs**: Must be set after tokenizer initialization. Stored in `conf.latent_token_id`, `bot_id`, `eot_id`.

4. **Detached recursions**: When `should_detach=True`, gradients don't flow through early passes. This is intentional (TRM paper design).

5. **Dataset structure**: Each sample has `question_tokenized`, `steps_tokenized` (list), `answer_tokenized`. See `coconut/dataset.py:tokenize_sample()`.

## Debugging

- Set `debug=True` in config for small dataset (1000 samples) and reduced batch size
- Use `Debug` or `TRMDebug` configs with `yujiepan/qwen3-tiny-random` for fast iteration
- Check `wandb` logs for loss curves, or set `WANDB_MODE=disabled` for local runs
- Eval metrics: `get_answer_preference()` compares latent vs non-latent perplexity

## References

- COCONUT paper: https://arxiv.org/abs/2412.06769
- TRM paper: https://arxiv.org/abs/2510.04871 (see `docs/trm_paper.md`)
- Reference TRM code: `docs/trm_reference_code/`
