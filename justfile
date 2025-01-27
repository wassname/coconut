

setup:
  uv sync
  . ./.venv/bin/activate
  bash scripts/preprocessing/gsm_icot.bash

run1:
  #!/bin/bash
  export CUDA_DEVICE_ORDER=PCI_BUS_ID
  export CUDA_VISIBLE_DEVICES=1
  . ./.venv/bin/activate
  python scripts/run.py args/gsm_qwen.yaml

run2:
  #!/bin/bash
  export CUDA_DEVICE_ORDER=PCI_BUS_ID
  export CUDA_VISIBLE_DEVICES=1
  . ./.venv/bin/activate
  python scripts/run.py args/gsm_qwen_1.5b.yaml
