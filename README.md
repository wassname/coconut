## Experiment: Tiny Recursive Models (TRM) for Latent Reasoning

This experiment adapts principles from Tiny Recursive Models ([TRM](https://arxiv.org/abs/2510.04871)) to the [COCONUT](https://arxiv.org/abs/2412.06769) framework. The goal is to replace the simple projection of latent tokens with a more structured, iterative reasoning process performed by a small, dedicated model.

The core idea is to use a frozen, quantized LLM for perception and generation, while a small, trainable TRM performs iterative refinement on the latent representations. This involves two learned components:

```py

uv run scripts/run.py TRMLoRA --help

# run
uv run scripts/run.py TRMLoRA
```


```py
uv run pytest --beartype-packages=''

# pytest with beartype type checking
uv run pytest --beartype-packages='coconut'

uv run scripts/run.py TRMLoRADebug
```

Testing

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
  - Like TRM recursion happens in latent space, with latent zL and output y being updated via multiple passes of net (the TRMTranscoder).
    - Unlike TRM which uses deep supervision (where the latent state is supervised at every recursive step for the same input) we use the coconut curriculum of multiple LLM forward passes which serves the same purpose: "the models learns to take any (zL, zH ) and improve it through a full recursion process, hopefully making zH closer to the solution"
    - We output both an input_embedding diff (like COCONUT) and a Q_hat (like TRM) to allow early stopping.
- Note we might disable ACT for simplity
- LLM is a 4bit frozen LLM (e.g. Qwen-3-0.6B)


FIXME: update the below to reflect lora adapter usage

```py

# FIXME y -> zH, zL -> zL, x -> hs

def latent_recursion(hs, zH, zL, n=6):
    for i in range(n): # latent reasoning
        zL = net(hs, zL, zH)
    zH = net(zH, zL) # refine output answer
    return zH, zL

def deep_recursion(hs, zH, zL, n=6, T=3):
    # recursing T-1 times to improve zH and zL (no gradients needed)
    with torch.no_grad():
        for j in range(T-1):
            zH, zL = latent_recursion(hs, zH, zL, n)
    # recursing once to improve zH and zL
    zH, zL = latent_recursion(hs, zH, zL, n)
    return (zH.detach(), zL.detach())





class TRMLora(nn.Module):
    def __init__(self, llm: LLMWrapper, trm_config: TRMConfig):
        super().__init__()

    def forward(self, x_input):
        # LORA  [h = W_0 @ x + B @ (A @ x)]
        # FIXME is this logic right

        # base layer forward [ W_0 @ x]
        hs = self.base_layer(x_input)

        zH = self.zH # provided by recursion context
        zL = self.zL

        # project down (A @ x)
        z_hs = self.down_proj(hs)

        self.zH, self.zL = deep_recursion(z_hs, zH, zL)

        # lora like up-projection [B @ (...)]
        features = (self.trmlora_B[adapter] @ zH.T).T

        # lora like addition [ h = ... + ...]
         result = result + features.unsqueeze(1)  # [b, 1, out] broadcast to [b, s, out]



for x_input, y_true in train_dataloader:
    zH, zL = y_init, z_init

    # where there are no <latent> tokens we disable the adapter, running just the frozen base model
    with disable_adapter(LLM):
        for _ in range(N_prefix):
            input_embed = LLM.forward(x_input).input_embed

    # we use use this context or pass the zL and zH states to the next step, here the adapter is enabled
    with recursion_context(net, {}):
        for _ in range(curriculum_stage):
            # Instead of deep supervision we use curriculum learning (as in COCONUT)
            input_embed = LLM.forward(input_embed=x_hs.detach()).input_embed

    with disable_adapter(LLM):
        for _ in range(N_suffix):
            input_embed = LLM.forward(input_embed=input_embed).input_embed
        y_hat = LLM.forward(input_embed=input_embed).logits
    loss = softmax_cross_entropy(y_hat, y_true)
    loss.backward()
    opt.step()
    opt.zero_grad()
```
Figure 1: Our TRM deep supervision adaptation to recurse on LLM hidden states and use EMA for supervision


```py
def latent_recursion(x, zH, zL, n=6):
    for i in range(n): # latent reasoning
        zL = net(x, zH, zL)
    zH = net(zH, zL) # refine output answer
    return zH, zL

def deep_recursion(x, zH, zL, n=6, T=3):
    # recursing T-1 times to improve zH and zL (no gradients needed)
    with torch.no_grad():
        for j in range(T-1):
            zH, zL = latent_recursion(x, zH, zL, n)
    # recursing once to improve zH and zL
    zH, zL = latent_recursion(x, zH, zL, n)
    return (zH.detach(), zL.detach()), output_head(zH), Q_head(zH)

# Deep Supervision
for x_input, y_true in train_dataloader:
    zH, zL = y_init, z_init
    for step in range(N_supervision):
        x = input_embedding(x_input)
        (zH, zL), y_hat, q_hat = deep_recursion(x, zH, zL)
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
