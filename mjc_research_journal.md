```bash
uv sync
. ./.venv/bin/activate
bash scripts/preprocessing/gsm_icot.bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
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
- oh we can use debug to make it fast

Example output format, ah so it expect `\n## A`


    Question 2: Answer = '1400' CoT = '<<30/100*2000=600>>
    <<2000-600=1400>>'
    Full output: 'Travis wants to fly to Australia. The regular tickets cost about $2000. As Travis is a student, he will get a 30% discount on this price. How much does he need to pay for his ticket?
    <<2000*0.3=600>>
    <<2000-600=1400>>
    ### 1400<|endoftext|>'
    Extracted Output: '1400'
    Test accuracy: 0.33:   1%|▊                                                                                                                                        | 3/500 [00:01<02:49,  2.93it/s]Setting `pad_token_id` to `eos_token_id`:None for open-end generation.
    Question 3: Answer = '15' CoT = '<<21/7=3>>
    <<5*3=15>>'
    Full output: 'A set of 7 spoons costs $21. If each spoon would be sold separately, how much would 5 spoons cost?
    <<21*5=105>>
    <<105*7=525>>
    ### 525<|endoftext|>'
    Extracted Output: '525'
    Test accuracy: 0.25:   1%|█                                                                                                                                        | 4/500 [00:01<02:49,  2.93it/s]Setting `pad_token_id` to `eos_token_id`:None for open-end generation.
    Question 4: Answer = '240' CoT = '<<200*3=600>>
    <<600*.4=240>>'
    Full output: 'Tom bought his games for $200.  They tripled in value and he then sold 40% of them.  How much did he sell the games for?
    <<200*3=600>>
    <<600*40/100=240>>
    <<600+240=720>>
    ### 720<|endoftext|>'


Could also try proper amp accel

# 2025-01-19 09:00:56


GPT2

Cor=71, CoT=27, Total=500
Accuracy on validation set: 71 / 500 = 0.142
CoT match on validation set: 27 / 500 = 0.054
saving model. outputs/gsm-cot/checkpoint_1

Cor=89, CoT=33, Total=500
Accuracy on validation set: 89 / 500 = 0.178
CoT match on validation set: 33 / 500 = 0.066

Accuracy on validation set: 99 / 500 = 0.198
CoT match on validation set: 32 / 500 = 0.064
saving model. outputs/gsm-cot/checkpoint_3

Accuracy on validation set: 112 / 500 = 0.224
CoT match on validation set: 40 / 500 = 0.08
saving model. outputs/gsm-cot/checkpoint_4

Test accuracy: 0.23: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 500/500 [01:21<00:00,  6.11it/s]
Cor=115, CoT=42, Total=500
Accuracy on validation set: 115 / 500 = 0.23
CoT match on validation set: 42 / 500 = 0.084
saving model. outputs/gsm-cot/checkpoint_5
wandb: 🚀 View run gsm-cot at: https://wandb.ai/wassname/coconut/runs/xxi8rd6h

So they both got higher slowly, I can see why 25 epochs

TODO get rid of 

After moving to Qwen (slower) and amp (faster) my speed is this

    Training Epoch: 4/5, batch 312/313 completed (loss: 0.1889: 100%|██████████████████████████████| 313/313 [02:52<00:00,  1.81it/s]

so it's 3.7x slower than gpt2, but it's 3.7x bigger. So amp didn't seem to help


oh waitn ot it's ~3.5 it/s, so only half as slow... oh but it slow down in later epochs, why is that?


I tried bnb 8 bit adam... it doesn't seem to help with mem. Maybe speed?


# 2025-01-19 13:56:50

So it's now 3h per epoch, 5 epochs.... hmmm. This just seems slow.

I wonder if converting to huggingface train would make it faster? It also seems good to do the runs in order so I can leave it overnight, rather than having to manually trigger each step

