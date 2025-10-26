

setup:
  uv sync
  . ./.venv/bin/activate
  bash scripts/preprocessing/gsm_icot.bash

run_smol:
  #!/bin/bash
  . ./.venv/bin/activate
  python scripts/run.py TRMDelora
  python scripts/run.py TRMLoRA
  python scripts/run.py TRMHra


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
