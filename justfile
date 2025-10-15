

setup:
  uv sync
  . ./.venv/bin/activate
  bash scripts/preprocessing/gsm_icot.bash

run_smol:
  #!/bin/bash
  . ./.venv/bin/activate
  python scripts/run.py GSMQwen 

run_trm:
  #!/bin/bash
  export CUDA_DEVICE_ORDER=PCI_BUS_ID
  export CUDA_VISIBLE_DEVICES=1
  . ./.venv/bin/activate


run_h100:
  #!/bin/bash
  . ./.venv/bin/activate
  python scripts/run.py TRM_H100
  python scripts/run.py GsmQwen_H100

run3:
  #!/bin/bash
  export CUDA_DEVICE_ORDER=PCI_BUS_ID
  export CUDA_VISIBLE_DEVICES=1
  . ./.venv/bin/activate
  python scripts/run.py args/gsm_qwen_trm_test.yaml

run2:
  #!/bin/bash
  export CUDA_DEVICE_ORDER=PCI_BUS_ID
  export CUDA_VISIBLE_DEVICES=1
  . ./.venv/bin/activate
  python scripts/run.py args/gsm_qwen_1.5b.yaml

sync_file:
  rsync -avz --progress outputs/qwen3-0.6b_20250514-194730/checkpoint_2/ vast:/workspace/coconut/outputs/qwen3-0.6b_20250514-194730/checkpoint_2/ -v

# push checkpoint to vast (all checkpoints, slw)
sync_outputs_push:
  rsync -avz --progress outputs/qwen3-0.6b_20250514-194730/ vast:/workspace/coconut/outputs/qwen3-0.6b_20250514-194730/ -v

sync_outputs_pull:
  rsync -avz --progress vast:/workspace/coconut/outputs/qwen3-0.6b_20250514-194730/ outputs/qwen3-0.6b_20250514-194730/ -v

# push data to vast
sync_data_push:
  rsync -avz --progress data/ vast:/workspace/coconut/data/ -v

sync_data_pull:
  rsync -avz --progress vast:/workspace/coconut/data/ data/ -v
