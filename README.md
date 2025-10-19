## Experiment: Tiny Recursive Models (TRM) for Latent Reasoning

This experiment adapts principles from Tiny Recursive Models ([TRM](https://arxiv.org/abs/2510.04871)) to the [COCONUT](https://arxiv.org/abs/2412.06769) framework. The goal is to replace the simple projection of latent tokens with a more structured, iterative reasoning process performed by a small, dedicated model.

The core idea is to use a frozen, quantized LLM for perception and generation, while a small, trainable TRM performs iterative refinement on the latent representations. This involves two learned components:


`uv run scripts/run.py TRM`

### Proposed Architecture and Training

The process integrates the LLM and the TRM as follows:

We have a sentence like ["The capital of France is <start-latent> <latent> <latent> ... <latent> <end-latent> "] where the `<latent>` tokens are to be filled in by the TRM. Then we generate the rest of the sentence, e.g. "Paris."

1.  The frozen LLM is loaded in 4bit and processes the input prompt up to the `<start-latent>` token, producing a sequence of hidden states.
2.  These hidden states are detached and fed into the TRM.
3.  The TRM's **Recurser** iteratively refines the latent representation over several steps.
4.  The **Transcoder** converts the final latent state from the TRM back into the LLM's input embedding space.
5.  The LLM's decoder then uses this final embedding to generate the output tokens.

The following pseudocode outlines this modified training loop, incorporating the LLM wrapper into the original TRM algorithm.


# v2

TODO merge above and below into a coherent description

We combine [COCONUT](https://arxiv.org/abs/2412.06769)[[code](https://github.com/facebookresearch/coconut)] and [TRM](https://arxiv.org/abs/2510.04871) [[code](https://github.com/SamsungSAILMontreal/TinyRecursiveModels)]. 
  - Like COCONUT we iterate on the hidden states of a pretrained LLM, using this to update the input_embeddings for the next LLM forward pass.
  - Like TRM recursion happens in latent space, with latent z and output y being updated via multiple passes of net (the TRMTranscoder).
    - Unlike TRM we use an approximation of deep supervision to account for expensive LLM forwards.
    - We output both an input_embedding diff (like COCONUT) and a Q_hat (like TRM) to allow early stopping.
- Note we might disable ACT for simplity
- LLM is a 4bit frozen LLM (e.g. Qwen-3-0.6B)



Instead of applying deep supervision at every layer—which would require an expensive LLM rollout for each step—I perform a single LLM rollout using the exponential moving average (EMA) of the hidden states during training. This approach ensures that all hidden states receive some supervision, aiming to capture the stabilizing effects of deep supervision while avoiding the computational cost associated with full LLM-based supervision at each layer.



```py
def latent_recursion(x, y, z, n=6):
    for i in range(n): # latent reasoning
        z = net(x, y, z)
    y = net(y, z) # refine output answer
    return y, z

def deep_recursion(x, y, z, n=6, T=3):
    # recursing T-1 times to improve y and z (no gradients needed)
    with torch.no_grad():
        for j in range(T-1):
            y, z = latent_recursion(x, y, z, n)
    # recursing once to improve y and z
    y, z = latent_recursion(x, y, z, n)
    return (y.detach(), z.detach()), output_head(y), Q_head(y)

# Deep Supervision
for x_input, y_true in train_dataloader:
    y, z = y_init, z_init
    ie_diff_ema = None
    alpha = 0.9 # ema smoothing factor
    x_hs = LLM.forward(x_input).hidden_states[-4] # new, our input/context space is pretrained LLM hidden states (as in COCONUT)
    ie = LLM.get_input_embeddings()(x_input) # new, our output space is LLM input embeddings (as in COCONUT)
    for step in range(N_supervision):
        (y, z), ie_diff, q_hat = deep_recursion(x_hs.detach(), y, z)

        # new: because LLM.forward is expensive in terms of memory/time, we use an EMA of input_embeddings_diff to stabilize training by providing supervision from multiple steps
        if ie_diff_ema is None:
            ie_diff_ema = ie_diff
        else:
            ie_diff_ema = alpha * ie_diff_ema + (1 - alpha) * ie_diff

    y_hat = LLM.generate(input_embed=ie + ie_diff_ema).logits # new, our output is added to LLM input embeddings
    loss = softmax_cross_entropy(y_hat, y_true)
    loss += binary_cross_entropy(q_hat, (y_hat == y_true))
    loss.backward()
    opt.step()
    opt.zero_grad()
    if q_hat > 0: # early-stopping
        break
```
Figure 1: Our TRM deep supervision adaptation to recurse on LLM hidden states and use EMA for supervision


```py
def latent_recursion(x, y, z, n=6):
    for i in range(n): # latent reasoning
        z = net(x, y, z)
    y = net(y, z) # refine output answer
    return y, z

def deep_recursion(x, y, z, n=6, T=3):
    # recursing T-1 times to improve y and z (no gradients needed)
    with torch.no_grad():
        for j in range(T-1):
            y, z = latent_recursion(x, y, z, n)
    # recursing once to improve y and z
    y, z = latent_recursion(x, y, z, n)
    return (y.detach(), z.detach()), output_head(y), Q_head(y)

# Deep Supervision
for x_input, y_true in train_dataloader:
    y, z = y_init, z_init
    for step in range(N_supervision):
        x = input_embedding(x_input)
        (y, z), y_hat, q_hat = deep_recursion(x, y, z)
        loss = softmax_cross_entropy(y_hat, y_true)
        loss += binary_cross_entropy(q_hat, (y_hat == y_true))
        loss.backward()
        opt.step()
        opt.zero_grad()
        if q_hat > 0: # early-stopping
            break
```
Figure 2: Original TRM deep supervision 



----

# Replicating and Extending COCONUT  
(Training Large Language Models to Reason in a Continuous Latent Space)

Replication and extension of [*Training Large Language Models to Reason in a Continuous Latent Space*](https://arxiv.org/abs/2412.06769), with added features:

- **SEQ-VCR loss** integration ([coconut/vcr_loss.py])  
- **Positional encodings** for latent tokens  
- **Qwen3-0.6B** configuration ([coconut/configs.py])  
- **Hidden-state reinjection** variants ([coconut/hs2ie.py])  
- General **refactoring**, packaging, and debug support  

## Contributions

- Added SEQ-VCR loss (see [coconut/vcr_loss.py])  
- Implemented latent-token positional encoding  
- Replicated on Qwen3-0.6B ([coconut/configs.py])  
- Explored hidden-state reinjection strategies:  
  - Suppressed neurons  
  - Second-to-last layer  
  - Projections of the last hidden state  
- Refactored codebase for clarity and single-GPU debugging  

## Findings

- Maintains accuracy with far fewer output tokens; more training will likely improve results.  
- Training time grows exponentially with token count—consider partial backpropagation or gradient checkpointing to improve compute efficiency.

![Accuracy vs. Tokens & Training Time](img/ksnip_20250518-095710.png)  
Full logs on [Weights & Biases](https://wandb.ai/wassname/coconut/runs/xvwpx0dj)


### Finding: The last hidden state is a poor choice for injection


|                        | eval/acc | eval/cot_em | 
| ---------------------: | -------: | ----------: | 
|       supressed[0.75:] |   0.3383 |      0.0074 | 
|       supressed[0.90:] |   0.2379 |      0.0112 | 
|                 hs[-4] |   0.2342 |      0.0112 | 
|                 hs[-3] |   0.2268 |      0.0112 | 
|        supressed[0.5:] |    0.223 |      0.0112 | 
|                 hs[-2] |   0.1896 |      0.0149 | 
|                 hs[-1] |   0.1747 |      0.0112 | 

In the table above we train for one epoch to see which method of hidden state injection works best. The first column is the method used, the second column is the accuracy on the eval set. The methods are `hs[-1]` (last hidden state), `hs[-2]` (second to last hidden state), and `supressed[0.5:]` (isolating the [suppressed activations](https://github.com/wassname/eliciting_suppressed_knowledge) in the last 50% of layers). As you can see the default `hs[-1]` is the worst performing method. The `supressed[0.75:]` method is the best performing method.

## Install

```bash
git clone https://github.com/wassname/coconut.git
cd coconut
uv sync
python3 -m venv .venv
source .venv/bin/activate
bash scripts/preprocessing/gsm_icot.bash
```

## Usage

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
source .venv/bin/activate
python scripts/run.py args/gsm_smol.yaml
```

## Project Plan & Experiments

- [x] Single-GPU setup (easier debugging)  
- [x] Refactoring & comments  
  - [x] Use `uv`  
  - [x] Package structure  
- [x] Switched to Qwen2.5-0.5B for higher capacity  
- [x] VSCode debugging  
- [ ] Full replication  
- **Ongoing experiments**:  
  - [ ] Suppressed-neuron injection  
  - [ ] Second-to-last layer hidden state  
  - [ ] Projected last hidden state (normalized)  

## Citation

If you use this code base, please cite the original paper:

```bibtex
@article{hao2024training,
  title={Training Large Language Models to Reason in a Continuous Latent Space},
  author={Hao, Shibo and Sukhbaatar, Sainbayar and Su, DiJia and Li, Xian and Hu, Zhiting and Weston, Jason and Tian, Yuandong},
  journal={arXiv preprint arXiv:2412.06769},
  year={2024}
}
```

And this replication:

```bibtex
@software{wassname2024coconut,
  author={Clark, M.J.},
  title={Replicating and Extending: Training Large Language Models to Reason in a Continuous Latent Space},
  year={2025},
  publisher={GitHub},
  journal={GitHub repository},
  url={https://github.com/wassname/coconut},
  commit={<commit hash>}
}
```

## License

Released under the MIT License. See [LICENSE](LICENSE) for details.
