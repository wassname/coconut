

setup:
  uv sync
  . ./.venv/bin/activate
  bash scripts/preprocessing/gsm_icot.bash

run1:
  export CUDA_DEVICE_ORDER=PCI_BUS_ID
  export CUDA_VISIBLE_DEVICES=1
  . ./.venv/bin/activate
  python scripts/run.py args/gsm_cot_qwen.yaml

  # python scripts/run.py args/gsm_coconut_qwen.yaml

  # python scripts/run.py args/gsm_coconut_qwen_eval.yaml
