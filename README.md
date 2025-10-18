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

```py
# where output_head converts zH to input embeddings
# where x are output hidden states from LLM
# hs are embeddings 
# where the llm is 4bit and frozen

def hrm(z, x, n=2, T=2): # hierarchical reasoning
    zH, zL = z
    with torch.no_grad():
        for i in range(nT - 2):
            zL = L_net(zL, zH, x)
            if (i + 1) % T == 0:
                zH = H_net(zH, zL)
    # 1-step grad
    zL = L_net(zL, zH, x)
    zH = H_net(zH, zL)
    return (zH, zL), output_head(zH), 0 # Q_head(zH)

# def ACT_halt(q, y_hat, y_true):
#     target_halt = (y_hat == y_true)
#     loss = 0.5*binary_cross_entropy(q[0], target_halt)
#     return loss

# def ACT_continue(q, last_step):
#     if last_step:
#         target_continue = sigmoid(q[0])
#     else:
#         target_continue = sigmoid(max(q[0], q[1]))
#     loss = 0.5*binary_cross_entropy(q[1], target_continue)
#     return loss

# Deep Supervision
for x_input, y_true in train_dataloader:
    z = z_init
    for step in range(N_sup): # deep supervision
        with torch.no_grad():
            # LLM converts input tokens to output hidden states
            x_hs = LLM(x_input).hidden_states[-1]
        z, embed_pred, q = hrm(z, x_hs)
        y_pred = LLM(embed_pred) # new
        loss = loss_fn(y_pred, y_true)

        # Note I have disabled ACT for now, it's only for efficiency
        # Adaptive computational time (ACT) using Q-learning
        # loss += ACT_halt(q, y_pred, y_true)  # ablation shows not needed
        # _, _, q_next = hrm(z, x_hs) # extra forward pass
        # loss += ACT_continue(q_next, step == N_sup - 1) # ablation shows not needed

        z = z.detach()
        loss.backward()
        opt.step()
        opt.zero_grad()
        # if q[0] > q[1]: # early-stopping
        #     break
```
Figure 2: Pseudocode of Hierarchical Reasoning Models (HRMs).


Instead of applying deep supervision at every layer—which would require an expensive LLM rollout for each step—I perform a single LLM rollout using the exponential moving average (EMA) of the hidden states during training. This approach ensures that all hidden states receive some supervision, aiming to capture the stabilizing effects of deep supervision while avoiding the computational cost associated with full LLM-based supervision at each layer.
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
