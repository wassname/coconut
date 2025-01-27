```bash
uv sync
. ./.venv/bin/activate
bash scripts/preprocessing/gsm_icot.bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
python scripts/run.py args/gsm_cot_qwen.yaml
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


2025-01-20 11:30:59.734 | INFO     | __main__:evaluate:296 - Cor=186, CoT=77, Total=500
2025-01-20 11:30:59.734 | INFO     | __main__:evaluate:297 - Accuracy on validation set:  186 / 500 = 0.372
2025-01-20 11:30:59.734 | INFO     | __main__:evaluate:298 - CoT match on validation set: 77 / 500 = 0.154
2025-01-20 11:31:02.566 | INFO     | __main__:save_model:56 - saving model. outputs/gsm-cot-qwen/checkpoint_0


2025-01-20 15:24:38.974 | INFO     | __main__:evaluate:296 - Cor=60, CoT=0, Total=500
2025-01-20 15:24:38.974 | INFO     | __main__:evaluate:297 - Accuracy on validation set:  60 / 500 = 0.12
2025-01-20 15:24:38.974 | INFO     | __main__:evaluate:298 - CoT match on validation set: 0 / 500 = 0.0
2025-01-20 15:24:41.670 | INFO     | __main__:save_model:56 - saving model. outputs/gsm-cot-qwen/checkpoint_1
2025-01-20 15:24:41.670 | INFO     | __main__:main:163 - Training stage 2

100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 32135/32135 [4:04:01<00:00,  2.19it/s]
2025-01-20 19:29:51.205 | INFO     | __main__:evaluate:296 - Cor=65, CoT=0, Total=500
2025-01-20 19:29:51.205 | INFO     | __main__:evaluate:297 - Accuracy on validation set:  65 / 500 = 0.13
2025-01-20 19:29:51.206 | INFO     | __main__:evaluate:298 - CoT match on validation set: 0 / 500 = 0.0
2025-01-20 19:29:54.066 | INFO     | __main__:save_model:56 - saving model. outputs/gsm-cot-qwen/checkpoint_2
Test accuracy: 0.13: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 500/500 [00:52<00:00,  9.54it/s]


# which model

Qwen/Qwen2.5-0.5B starts with loss 0.6, loss 0.29 by step 500. 0.133 by end of epoch, .18 by 0.5 epochs
plaguss/Qwen2.5-0.5B-Math-Shepherd-PRM-0.2 starts 0.97, 0.28 by 500, 0.18 by 0.5
Qwen/Qwen2.5-Coder-0.5B 0.6, 0.3 by 500, 0.19 at 0.5 epochs

{'loss': 0.6079, 'grad_norm': 13.604103088378906, 'learning_rate': 6.222775357809583e-06, 'epoch': 0.0}                                                                          
{'loss': 0.2347, 'grad_norm': 8.023530960083008, 'learning_rate': 1.2445550715619167e-05, 'epoch': 0.01}                                                                         
{'loss': 0.2437, 'grad_norm': 6.747913360595703, 'learning_rate': 1.8668326073428747e-05, 'epoch': 0.01}                                                                         
{'loss': 0.2598, 'grad_norm': 6.684649467468262, 'learning_rate': 2.4891101431238333e-05, 'epoch': 0.01}                                                                         
{'loss': 0.2888, 'grad_norm': 6.1623854637146, 'learning_rate': 3.111387678904792e-05, 'epoch': 0.02}                                                                            
{'loss': 0.3016, 'grad_norm': 6.5927910804748535, 'learning_rate': 3.7336652146857495e-05, 'epoch': 0.02}                                                                        
{'loss': 0.3543, 'grad_norm': 3.8870067596435547, 'learning_rate': 4.3559427504667084e-05, 'epoch': 0.02}                                                                        
{'loss': 0.32, 'grad_norm': 4.897699356079102, 'learning_rate': 4.9782202862476667e-05, 'epoch': 0.02}                                                                           
{'loss': 0.3348, 'grad_norm': 22.028501510620117, 'learning_rate': 5.6004978220286256e-05, 'epoch': 0.03}                                                                        
{'loss': 0.3334, 'grad_norm': 5.9049763679504395, 'learning_rate': 6.222775357809584e-05, 'epoch': 0.03}                                                                         
{'loss': 0.3385, 'grad_norm': 5.6262736320495605, 'learning_rate': 6.845052893590542e-05, 'epoch': 0.03}                                                                         
{'loss': 0.3512, 'grad_norm': 3.506831169128418, 'learning_rate': 7.467330429371499e-05, 'epoch': 0.04}                                                                          
{'loss': 0.3667, 'grad_norm': 52.041961669921875, 'learning_rate': 8.089607965152459e-05, 'epoch': 0.04}                                                                         
{'loss': 0.35, 'grad_norm': 3.949807643890381, 'learning_rate': 8.711885500933417e-05, 'epoch': 0.04}                                                                            
{'loss': 0.3614, 'grad_norm': 4.492116451263428, 'learning_rate': 9.334163036714375e-05, 'epoch': 0.05}                                                                          
{'loss': 0.3487, 'grad_norm': 4.225961685180664, 'learning_rate': 9.956440572495333e-05, 'epoch': 0.05}                                                                          
{'loss': 0.3774, 'grad_norm': 2.8433144092559814, 'learning_rate': 0.00010578718108276292, 'epoch': 0.05}                                                                        
{'loss': 0.3629, 'grad_norm': 3.389758825302124, 'learning_rate': 0.00011200995644057251, 'epoch': 0.06}                                                                         
{'loss': 0.3671, 'grad_norm': 4.647830486297607, 'learning_rate': 0.00011823273179838208, 'epoch': 0.06}                                                                         
{'loss': 0.3845, 'grad_norm': 2.632505178451538, 'learning_rate': 0.00012445550715619168, 'epoch': 0.06}                     


model_id: Qwen/Qwen2.5-Coder-0.5B

    To determine how much John pays per year for his grass cutting we need to calculate the number of months it takes for his grass to grow from 2 inches to 4 inches and then determine the cost based on the number of months.

    1. Calculate the number of months it takes for the grass to grow from'

plaguss/Qwen2.5-0.5B-Math-Shepherd-PRM-0.2

    In the first month the grass grows 0.5 inches so it reaches 2 + 0.5 = 2.5 inches. In the second month it grows 0.5 + 0.5 = 1.0 so it reaches 2 + 0.5 +'

Qwen/Qwen2.5-0.5B

    To determine how much John pays per year for his grass cutting service we need to follow these steps:

    1. **Determine the number of cuts needed:**
    - John starts with 2 inches of grass.
    - It grows at a rate of 0.5 inches per month.
    - After'

Qwen/Qwen2.5-0.5B-Instruct



hm distilling r1 into qwer
https://huggingface.co/Qwen/Qwen2.5-Math-1.5B
Qwen2.5-Math-1.5B  79.7 -> 83.9 on the MATH benchmark using TIR. 


            To determine how much John pays per year for cutting his grass, we need to follow these steps:

            1. Calculate how many times John needs to cut his grass in a year.
            2. Determine the cost per cut.
            3. Multiply the number of cuts by the cost per cut to get the total annual cost.

            Let

starts at 

# 2025-01-26 16:55:17

Ah a nice replication of r1-Zero came out. learnings.
Doesn't matter if you use instruct or not
the 0.5 model kind of suck, 1.5 is better
you don't need 40,000 samples, they used 8000 (gsm8k)


they used 0.5b, le=1e-6


train_batch size = 256 # Reward batch size
ppo mini batch size 64 # One sample is split into multiple sub-batches with batch_size=ppo_mini_batch_size for PPO updates (grad accum size)
ppo_micro_batch_size=1  #  Similar to gradient accumulation, the micro_batch_size for one forward pass, trading speed for GPU memory
log_prob_micro_bathc size 4 (the real size)

Speed
- 0_0 5min
- 1_0 9min for 10k
- 1_1 12mins for 10k
- test is always 5mins for 500

so the latent part does slow us down

an alternative method might be to just always do the latent forward, 1 step at a time with cache, but only recurse the loss if the input or output is latent



# with and without bf16

    2025-01-27 17:15:37.425 | INFO     | __main__:evaluate:325 - Question 0: Answer = '300' CoT = '<<4-2=2>>                                                                                           
            <<2/.5=4>>
            <<12/4=3>>
            <<100*3=300>>'
    Extracted llm Output: 'John cuts his grass to 2 inche...' (=? 300) ❌.
    Full llm output: 'John cuts his grass to 2 inches.  It grows .5 inches per month.  When it gets to 4 inches he cuts it back down to 2 inches.  It cost $100 to get his grass cut.  How much does he pay per year?
            <|start-latent|><|end-latent|><<<
            To determine how much John pays per year for cutting his grass, we need to follow these steps:

            1. Calculate the number of times John needs to cut his grass in a year.
            2. Determine the cost per cut.
            3. Multiply the number of cuts by the cost per cut to get the total annual'. 

with 
2025-01-27 17:36:39.008 | INFO     | __main__:evaluate:321 - Question 0: Answer = '300' CoT = '<<4-2=2>>                                                                                           
        <<2/.5=4>>
        <<12/4=3>>
        <<100*3=300>>'
Extracted llm Output: 'John cuts his grass to 2 inche...' (=? 300) ❌.
Full llm output: 'John cuts his grass to 2 inches.  It grows .5 inches per month.  When it gets to 4 inches he cuts it back down to 2 inches.  It cost $100 to get his grass cut.  How much does he pay per year?
        <|start-latent|><|end-latent|><<<
        To determine how much John pays per year for cutting his grass, we need to follow these steps:

        1. Calculate the number of times John needs to cut his grass in a year.
        2. Determine the cost per cut.
        3. Multiply the number of cuts by the cost per cut to get the total annual'. 
