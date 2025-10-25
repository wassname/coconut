# Copilot Instructions: COCONUT + TRM

I am applying this TRM (tiny recursive models) as an adapter (peft lora style) in a COCONUT style setup.

## Project Overview

This is a research implementation combining two approaches for latent reasoning in LLMs see README.md for details:

### Special Tokens System
Three special tokens enable latent reasoning:
- `<|start-latent|>`: Marks beginning of latent reasoning block
- `<|latent|>`: Placeholder tokens for internal computation (not decoded to text)
- `<|end-latent|>`: Marks end of latent block

## Key Files

- `coconut/coconut.py`: Main model wrapper, handles multi-pass latent token processing
- `coconut/configs.py`: All experiment configs as Pydantic dataclasses (TRM, Debug, etc.)
- `coconut/dataset.py`: Tokenizes GSM8K with latent tokens, handles staged training
- `coconut/recursive_lora.py`: TRM LoRA adapter implementation
- `scripts/run.py`: Training loop with staged curriculum (CoT → 1 latent → 2 latents → ...)

## Training Workflow

### Staged Curriculum
Training progresses through stages, gradually increasing latent tokens:
- **Stage -1** (epochs 0-2): Pure CoT with `<start-latent>` and `<end-latent>` but no `<latent>` tokens
- **Stage 0+**: Add `c_thought` latent tokens per reasoning step, up to `max_latent_stage`

example output

    Full llm output: `['<<', '1', '0', '0', '/', '1', '2', '=', '8', '.', '3', '3', '>>\n', '###', ' ', '8', '.', '3', '3', '\n', '<|im_end|>',]`. 
    Extracted llm Output: `8.33` (=? 300) ❌.
    ideal_CoT = '<<4-2=2>>
            <<2/.5=4>>
            <<12/4=3>>
            <<100*3=300>>'.
    Answer = '300' .



# Results: trm-qwen3-0.6b_20251019-204459


|    |   eval/acc |   eval/cot_em |   eval/ratios |   epoch |   stage |   train/minutes |   train/loss |   eval/loss |
|---:|-----------:|--------------:|--------------:|--------:|--------:|----------------:|-------------:|------------:|
|  0 |      0.434 |         0.028 |        0.9543 |       8 |       1 |         13.6735 |     0.693636 |      0.8054 |

Example results where

- cot_em is exact match of chain-of-thought reasoning steps
- ratios is perplexity ratio of correct vs corrupted answers (lower is better)
- acc is the final answer accuracy

### Running Experiments
```bash

# see options
uv run scripts/run.py --help
uv run scripts/run.py TRMLora --help

# Or directly with tyro CLI
uv run python scripts/run.py TRM --batch_size_training=8 --max_size=10000
uv run python scripts/run.py Debug  # Fast iteration with tiny model

uv run pytest # to get type errors and do an integration test on a tiny model
```

Configs are Pydantic dataclasses in `coconut/configs.py`. CLI args override config fields via tyro. 

### Key Config Parameters

see ./coconut/configs.py for full details

## Development Patterns

- I use jaxtyping for annotating tensor shapes:
    ```python
    from jaxtyping import Float, Int
    from torch import Tensor

    def forward(self, hs: Float[Tensor, 'b t h']) -> Float[Tensor, 'b t h']:
    ```

- I use loguru to log
- I use anycache to cache to disc as a function wrapper

## Common Pitfalls

1. **Gradient flow in TRM mode**: LLM is frozen but used for grad in places.

2. **Multi-pass processing**: `coconut.py` processes input in multiple passes (one per latent token). KV cache carries over between passes. Don't assume single forward pass. We also have a recursion context for TRM state carryover.

3. **Adapter shapes during generation**: During both training and generation, we process latent tokens one at a time (`s=1`). Adapters must output `[b, 1, out]` to match base model output shape, even though they may only process the last token. Use `.unsqueeze(1)` after computing deltas.

4. **KV cache and TRM modifications**: When processing latent token N, the KV cache contains positions 0 to N-1. TRM modifications to the current token's hidden state flow through remaining layers but only get added to KV cache after the full forward pass completes. Next latent token (N+1) will then attend to the TRM-modified representation of token N via KV cache.

5. **Adapter enable/disable per pass**: Non-latent tokens are processed with adapter disabled (base model only). Latent tokens are processed with adapter enabled (TRM active). This is controlled via `set_adapter()` context manager in `coconut.py`.

6. TRM detaches all but the last one or two recursions. This is intentional, from the paper. In combination with deep supervision (or curriculum learning here) this may still solve the fixed point problem.

