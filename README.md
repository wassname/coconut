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
