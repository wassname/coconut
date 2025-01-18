```bash
uv sync
. ./.venv/bin/activate
bash scripts/preprocessing/gsm_icot.bash
# torchrun --nnodes 1 --nproc_per_node 1 scripts/run.py args/gsm_coconut.yaml
python scripts/run.py args/gsm_coconut.yaml
```


# 2025-01-18 14:37:07

replace model with 'plaguss/Qwen2.5-0.5B-Math-Shepherd-PRM-0.2', or 'HuggingFaceH4/Qwen2.5-Math-1.5B-Instruct-PRM-0.2', or Qwen/Qwen2.5-0.5B", ?


gpt2 batch 64, 20 mins per batch, 5 epochs, bf16. This fills up the gpu. GPT2 is 137M params. 


So qwen 0.5 is ~3-4x. Mean batchsize should be 16, but lets see if gpt2 works first


### 2025-01-18 15:19:06 Lets look through the code

- custom training
- loss from CrossEntropyLoss, similar to casuallm
- no comments
- acc is from `text_output.split("#")[-1].replace(",", "").strip()`
  - so ? what format does it expect?
- doesn't use prompt format, tokenises as 3 parts. So it would fail for chatml

```py
question_tokenized = tokenizer.encode(
    sample["question"] + "\n", add_special_tokens=True
)
steps_tokenized = [
    tokenizer.encode(s + "\n", add_special_tokens=False)
    for s in sample["steps"]
]
answer_tokenized = tokenizer.encode(
    "### " + sample["answer"], add_special_tokens=False
) + [tokenizer.eos_token_id]

tokens = (
    sample["question_tokenized"]
    + ([] if no_special_marker else [start_id])
    + [latent_id] * n_latent_tokens
    + ([] if no_special_marker else [end_id])
    + list(
        itertools.chain.from_iterable(sample["steps_tokenized"][n_skip_steps:])
    )
    + sample["answer_tokenized"]
)
```
it looks like casper train, train, train is cleaner? but at least thisi s explicit


coconut.Forward...
- it get the latent indices, seems to handle batches?
- first does a forward pass over the first steps without latent tokens, nice
- then they loop over untill the last latent step
- next_compute_range keep track
- they just do one token at a time, so they can use kv cache. except for the final non latent forward
- they modify kv cache if needed, just getting up to the start of the range! using it as just tuples
