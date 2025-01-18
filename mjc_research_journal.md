```bash
uv sync
. ./.venv/bin/activate
bash scripts/preprocessing/gsm_icot.bash
# torchrun --nnodes 1 --nproc_per_node 1 scripts/run.py args/gsm_coconut.yaml
python scripts/run.py args/gsm_coconut.yaml
```
