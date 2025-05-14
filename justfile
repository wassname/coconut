

setup:
  uv sync
  . ./.venv/bin/activate
  bash scripts/preprocessing/gsm_icot.bash

run_smol:
  #!/bin/bash
  . ./.venv/bin/activate
  python scripts/run.py GSMQwen 
