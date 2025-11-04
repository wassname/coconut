

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

  uv run scripts/run.py TRMSvft  --target-modules-pattern='.+\.(gate_proj).*$'
  uv run scripts/run.py TRMSvft  --target-modules-pattern='.+\.(o_proj).*$'
  uv run scripts/run.py TRMSvft  --layers_spacing_adapter=2  --adapter-svft-mode='replace_add'
  uv run scripts/run.py TRMSvft  --target-modules-pattern='.+\.(gate_proj).*$' --layers_spacing_adapter=2 --adapter_r=512

  uv run scripts/run.py TRMSvft --trm_h_cycles=0 --trm_l_cycles=0 --trm_num_heads=1 --trm_expansion=1
  uv run scripts/run.py TRMSvft --lr=1e-1 --gradient-accumulation-steps=1 --weight_decay=1 --num_epochs=6 --max_size=1000 --adapter_r=128 --scheduler=cosine  --target-modules-pattern='.+\.(gate_proj|down_proj).*$' --layers-spacing-adapter=15
  # do one with <think>...</think>
  # uv run scripts/run.py TRMSvft --bot_token="<think>" --eot_token="</think>" --lr=1e-2 --weight_decay=.1 --num_epochs=2
  # uv run scripts/run.py TRMSvft --bot_token="🤔" --eot_token="💭" --latent_token="∴" --lr=1e-2 --weight_decay=.1 --num_epochs=2
  # uv run scripts/run.py TRMSvft --bot_token="🤔" --eot_token="➡️" --latent_token="🔄" --lr=1e-2 --weight_decay=.1 --num_epochs=2
  uv run scripts/run.py TRMSvft --bot_token="Wait" --eot_token="Ans" --latent_token="..." --lr=1e-2 --weight_decay=.1 --num_epochs=2
  uv run scripts/run.py TRMSvft --adapter-svft-mode='adapter_mult'
  uv run scripts/run.py TRMSvft --adapter-svft-mode='replace_mul'
  uv run scripts/run.py TRMSvft
  python scripts/run.py TRMLoRA
  python scripts/run.py TRMDelora --loss-nll-ratio-margin
  python scripts/run.py TRMHra
  uv run scripts/run.py TRMSvft --adapter-svft-mode='adapter_add'  --gradient-accumulation-steps=1 --layers-spacing-adapter=2 --r=2048 --lr=1e-3 --scheduler=cosine --weight-decay=0 --trm-expansion=4 --loss-nll-ratio-margin
  uv run scripts/run.py TRMSvft --adapter-svft-mode='replace_add' --lr=1e-4 --weight-decay=0 --scheduler=cosine --layers-spacing-adapter=2 --r=512 --gradient-accumulation-steps=1
  uv run scripts/run.py TRMSvft --adapter-svft-mode='replace_add' --lr=1e-2 --weight-decay=1 --scheduler=cosine --layers-spacing-adapter=2 --r=512 --gradient-accumulation-steps=12
  uv run scripts/run.py TRMSvft --target-modules-pattern='.+\.(o_proj).*$' --adapter-svft-mode='adapter_add'  --layers-spacing-adapter=10
  uv run scripts/run.py TRMSvft --target-modules-pattern='.+\.(v_proj|k_proj).*$' --adapter-svft-mode='adapter_add' --layers-spacing-adapter=8
  uv run scripts/run.py TRMSvft --target-modules-pattern='.+\.(gate_proj).*$' --adapter-svft-mode='adapter_add'  --layers-spacing-adapter=10
  uv run scripts/run.py TRMSvft --target-modules-pattern='.+$' --adapter-svft-mode='adapter_add'  --layers-spacing-adapter=2

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
