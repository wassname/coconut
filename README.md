## Experiment: Tiny Recursive Models (TRM) for Latent Reasoning

This experiment adapts principles from Tiny Recursive Models ([TRM](https://arxiv.org/abs/2510.04871)) to the [COCONUT](https://arxiv.org/abs/2412.06769) framework. The goal is to replace the simple projection of latent tokens with a more structured, iterative reasoning process performed by a small, dedicated model.

The core idea is to use a frozen, quantized LLM for perception and generation, while a small, trainable TRM performs iterative refinement on the latent representations. This involves two learned components:


`uv run scripts/run.py TRMLoRA`

`uv run pytest --beartype-packages='coconut'`

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
    - Unlike TRM which uses deep supervision (where the latent state is supervised at every recursive step for the same input) we use the coconut curriculum of multiple LLM forward passes which serves the same purpose: "the models learns to take any (zL, zH ) and improve it through a full recursion process, hopefully making zH closer to the solution"
    - We output both an input_embedding diff (like COCONUT) and a Q_hat (like TRM) to allow early stopping.
- Note we might disable ACT for simplity
- LLM is a 4bit frozen LLM (e.g. Qwen-3-0.6B)


FIXME: update the below to reflect lora adapter usage
Open questions:
- make sure we cary zH and zL through the recursion? how
- do we have it active all the time? or only on think
- ?

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
    x_hs = LLM.forward(x_input).hidden_states[-4] # new, our input/context space is pretrained LLM hidden states (as in COCONUT)
    ie = LLM.get_input_embeddings()(x_input) # new, our output space is LLM input embeddings (as in COCONUT)

    # Instead of deep supervision we use curriculum learning (as in COCONUT)
    for curriculum_stage in range(N_curriculum):
      for step in range(N_supervision):
          (y, z), ie_diff, q_hat = deep_recursion(x_hs.detach(), y, z)
    y_hat = LLM.generate(input_embed=ie + ie_diff).logits # new, our output is added to LLM input embeddings
    loss = softmax_cross_entropy(y_hat, y_true)
    # loss += binary_cross_entropy(q_hat, (y_hat == y_true)) # not currently used
    loss.backward()
    opt.step()
    opt.zero_grad()
    # if q_hat > 0: # early-stopping
    #     break
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

# 2025-10-22 20:44:53 Update

Adding to the input_embeddings seems to fragile, I'm going to try a peft style adapter in this branch


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
