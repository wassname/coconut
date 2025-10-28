

setup:
  uv sync
  . ./.venv/bin/activate
  bash scripts/preprocessing/gsm_icot.bash

run:
  #!/bin/bash
  # set -x +e
  . ./.venv/bin/activate
  # python scripts/run.py TRMDelora --lr=3e-3 --layers_spacing_adapter=4 --trm_h_cycles=3 --trm_l_cycles=6 --trm_num_heads=8 --trm_expansion=8 --gradient_accumulation_steps=4
  # python scripts/run.py TRMLoRA
  # python scripts/run.py TRMHra
  # python scripts/run.py TRMDelora --no-trm-persistent-steering --loss-nll-ratio-margin
  
  # uv run scripts/run.py TRMSvft --adapter-svft-mode='replace_mul' --layers-spacing-adapter=5
  # uv run scripts/run.py TRMSvft --target-modules-pattern='.+\.(o_proj|v_proj).*$' --adapter-svft-mode='adapter_add'
  # uv run scripts/run.py TRMSvft --target-modules-pattern='.+\.(up_proj|down_proj).*$' --adapter-svft-mode='adapter_add'
  # uv run scripts/run.py TRMSvft --target-modules-pattern='.+\.(gate_proj).*$' --adapter-svft-mode='adapter_add'
  # uv run scripts/run.py TRMSvft --target-modules-pattern='.+\.(q_proj|k_proj).*$' --adapter-svft-mode='adapter_add'
  # uv run scripts/run.py TRMSvft --adapter-svft-mode='replace_add' --layers-spacing-adapter=5
  # uv run scripts/run.py TRMSvft --adapter-svft-mode='adapter_mult' --layers-spacing-adapter=5
  # uv run scripts/run.py TRMSvft --adapter-svft-mode='adapter_add'  --layers-spacing-adapter=5 --no-persistent-steering

  uv run scripts/run.py TRMSvft
  uv run scripts/run.py TRMSvft --adapter-svft-mode='adapter_add'  --gradient-accumulation-steps=1 --layers-spacing-adapter=2 --r=2048 --lr=1e-3 --scheduler=cosine --weight-decay=0 --trm-expansion=4 --loss-nll-ratio-margin
  uv run scripts/run.py TRMSvft --target-modules-pattern='.+\.(o_proj).*$' --adapter-svft-mode='adapter_add'
  uv run scripts/run.py TRMSvft --target-modules-pattern='.+\.(v_proj|k_proj).*$' --adapter-svft-mode='adapter_add'
  uv run scripts/run.py TRMSvft --target-modules-pattern='.+\.(gate_proj).*$' --adapter-svft-mode='adapter_add'
  uv run scripts/run.py TRMSvft --target-modules-pattern='.+$' --adapter-svft-mode='adapter_add'  --layers-spacing-adapter=2
  uv run scripts/run.py TRMSvft --adapter-svft-mode='adapter_mult'
  uv run scripts/run.py TRMSvft --adapter-svft-mode='replace_mul'
  uv run scripts/run.py TRMSvft --adapter-svft-mode='replace_add'
  python scripts/run.py TRMLoRA
  python scripts/run.py TRMDelora --loss-nll-ratio-margin
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
