# Copilot Instructions: COCONUT + TRM

## Project Overview

This is a research implementation combining two approaches for latent reasoning in LLMs see README.md for details:

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
- See `CoconutTRM.hrm()` for recursive refinement matching paper pseudocode
see README.md for more details

## Key Files

- `coconut/coconut.py`: Main model wrapper, handles multi-pass latent token processing
- `coconut/trm_adapter.py`: TRM recursive reasoning module (L_net, H_net, transcoder)
- `coconut/configs.py`: All experiment configs as Pydantic dataclasses (TRM, Debug, etc.)
- `coconut/dataset.py`: Tokenizes GSM8K with latent tokens, handles staged training
- `coconut/hs2ie.py`: Hidden-state replacement methods, including novel "suppressed activations" experiment
<!-- - `coconut/vcr_loss.py`: SEQ-VCR loss for intermediate state stability (experimental) -->
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

    Test accuracy: 0.25. eval_8: 100%|| 32/32 [01:55<00:00,  3.62s/it]
    Correct=124, CoT_correct=6, Total=500. eval_8                                       
    Accuracy on val:  124 / 500 =  24.8000%                                             
    CoT match on val: 6 / 500 =  1.2000%                                                
    ratio nll_ans/nll_corrupted_ans = 0.9249       

### Running Experiments
```bash

# Or directly with tyro CLI
uv run python scripts/run.py TRM --batch_size_training=8 --max_size=10000
uv run python scripts/run.py Debug  # Fast iteration with tiny model
```

Configs are Pydantic dataclasses in `coconut/configs.py`. CLI args override config fields via tyro. `uv run scripts/run.py --help` for full options.

### Key Config Parameters

see ./coconut/configs.py for full details

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
Use anycache for expensive preprocessing this wraps a function and caches to disc based on a hash of the function arguments

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

## Common Pitfalls

1. **Gradient flow in TRM mode**: LLM is frozen but used for grad in places.

2. **Multi-pass processing**: `coconut.py` processes input in multiple passes (one per latent token). KV cache carries over between passes. Don't assume single forward pass.

3. **Special token IDs**: Must be set after tokenizer initialization. Stored in `conf.latent_token_id`, `bot_id`, `eot_id`.

4. **Detached recursions**: When `should_detach=True`, gradients don't flow through early passes. This is intentional (TRM paper design).


- Eval metrics: `get_answer_preference()` compares perplexity on good vs wrong answers as a ratio, lower is better

## References

- README.md for public project description
- COCONUT paper: https://arxiv.org/abs/2412.06769
- TRM paper: https://arxiv.org/abs/2510.04871 (see `docs/trm_paper.md`)
- Reference TRM code: `docs/trm_reference_code/`
