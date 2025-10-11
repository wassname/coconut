# TRM-Style Detached Recursions Implementation

## Summary

Minimal implementation of TRM-style training where early recursive passes are detached (no gradients), and only the last N passes backpropagate gradients. This forces the model to learn to clean up its own accumulated errors.

## Changes Made

### 1. Config (`coconut/configs.py`)
Added parameter to `BaseConfig`:
```python
n_detached_recursions: int = 0  # 0=disabled, 2=keep gradients only for last 2 passes
```

Created config classes:
```python
@dataclass
class TRMTest(BaseConfig):
    """TRM-style detached recursions test"""
    name: str = "gsm-qwen-trm-test"
    n_detached_recursions: int = 2  # Only backprop last 2 of 4 passes
    max_latent_stage: int = 4
    # ... other settings

@dataclass
class TRMBaseline(TRMTest):
    """Baseline for comparison"""
    name: str = "gsm-qwen-trm-baseline"
    n_detached_recursions: int = 0  # Backprop all passes (standard)
```

### 2. Model Config (`coconut/coconut.py`)
Added to `CoconutConfig`:
```python
self.n_detached_recursions = None
```

Added assertion check in `CoconutQwen3ForCausalLM.__init__`:
```python
assert self.config.n_detached_recursions is not None
```

### 3. Model Loading (`coconut/load_model.py`)
Pass the parameter to the config in both `load_new_model` and `resume_model`:
```python
model_config = CoconutConfig.from_pretrained(
    conf.model_id,
    # ... other params ...
    n_detached_recursions=conf.n_detached_recursions,
)
```

### 3. Forward Pass (`coconut/coconut.py`)
Modified the latent token loop to conditionally detach:

```python
for pass_idx in range(max_n_latents):
    # TRM-style: detach gradients for early passes, keep gradients for last N passes
    should_detach = (
        self.training 
        and self.config.n_detached_recursions > 0 
        and pass_idx < (max_n_latents - self.config.n_detached_recursions)
    )
    
    if should_detach:
        ctx = torch.no_grad()
    else:
        ctx = torch.enable_grad()
    
    with ctx:
        # ... existing forward logic ...
    
    # Detach inputs_embeds after detached passes
    if should_detach:
        inputs_embeds = inputs_embeds.detach()
```

### 4. Test Config (`args/gsm_qwen_trm_test.yaml`)
Created test configuration with:
- `n_detached_recursions: 2` (last 2 passes get gradients)
- `max_latent_stage: 4` (4 latent tokens total)
- Reduced batch size for memory

## How It Works

**Example**: If `max_n_latents=4` and `n_detached_recursions=2`:
- **Pass 0,1**: Run with `torch.no_grad()` 
  - Model does "blind" recursions
  - Accumulates its own errors/junk
  - No gradient computation (faster, less memory)
  
- **Pass 2,3**: Run with gradients enabled
  - Model sees accumulated errors from passes 0,1
  - Learns to clean up and be robust to its own mistakes
  - Gradients flow back to update weights

## Key Insight from TRM Paper

> "After many of its own steps, it will be filled with its own junk, it will have to learn to clean it up! And that cleaning step might be the element needed to make it stable and convergent!"

This is like training a denoising autoencoder on the model's own outputs - it learns an implicit error correction dynamic.

## Testing

### Using Config Classes (Recommended)

Run the TRM experiment:
```bash
uv run python scripts/run.py TRMTest
```

Run the baseline for comparison:
```bash
uv run python scripts/run.py TRMBaseline
```

You can override parameters:
```bash
uv run python scripts/run.py TRMTest --n-detached-recursions 3 --max-latent-stage 6
```

### Using YAML Config Files

Alternatively, use the YAML config:
```bash
uv run python scripts/run.py args/gsm_qwen_trm_test.yaml
```

## Expected Results

- **Memory**: Slightly lower during detached passes (no gradient storage)
- **Speed**: Slightly faster overall (fewer backward passes)
- **Stability**: Potentially more stable training (error correction mechanism)
- **Accuracy**: Should match or exceed baseline if hypothesis is correct

## Next Steps

1. Run full experiment comparing n_detached_recursions=0,1,2,3
2. Monitor loss curves and accuracy
3. If promising, try with larger models on H100
4. Consider adding EMA (Exponential Moving Average) for extra stability
