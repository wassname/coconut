

setup:
  uv sync
  . ./.venv/bin/activate
  bash scripts/preprocessing/gsm_icot.bash

run1:
  #!/bin/bash
  export CUDA_DEVICE_ORDER=PCI_BUS_ID
  export CUDA_VISIBLE_DEVICES=1
  . ./.venv/bin/activate
  python scripts/run.py gsm_qwen

run2:
  #!/bin/bash
  export CUDA_DEVICE_ORDER=PCI_BUS_ID
  export CUDA_VISIBLE_DEVICES=1
  . ./.venv/bin/activate
  python scripts/run.py gsm_qwen_1.5b


vast:
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export CUDA_DEVICE_ORDER=PCI_BUS_ID
  . ./.venv/bin/activate
  python scripts/run.py GsmQwen1_5b_H100
