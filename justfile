

setup:
  uv sync
  . ./.venv/bin/activate
  bash scripts/preprocessing/gsm_icot.bash

run:
  #!/bin/bash
  . ./.venv/bin/activate
  python scripts/run.py TRMDelora --lr=3e-3 --layers_spacing_adapter=4 --trm_h_cycles=3 --trm_l_cycles=6 --trm_num_heads=8 --trm_expansion=8 --gradient_accumulation_steps=4
  python scripts/run.py TRMLoRA
  python scripts/run.py TRMHra
  python scripts/run.py TRMDelora --no-trm-persistent-steering --loss-nll-ratio-margin


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
