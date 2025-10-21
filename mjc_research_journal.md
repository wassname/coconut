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
So.. they are all about the same?

| model_id     | loss 0.5 | loss 500 | loss 0.5 epochs |
| ------------ | -------- | -------- | --------------- |
| Qwen         | 0.6      | 0.29     | 0.18            |
| Math-PRM-0.2 | 0.97     | 0.28     | 0.18            |
| Coder-0.5B   | 0.6      | 0.3      | 0.19            |


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


So right now I'm getting

| epoch | thoughts | acc  | mins | notes        |
| ----- | -------- | ---- | ---- | ------------ |
| 0     | 0        | 0.7  | 23   | test is 5min |
| 1     | 2        | 0.6  | 30   |
| 2     | 4        | 0.47 | 32   |
| 3     | 6        | 0.36 | 32   |
| 4     | 6        | 0.31 | 33   |





- 0 thought, 0.7, 23min, test is 5min
- 2, 0.6, 30min
- 4, 0.47, 32min
- 6, 0.36, 30min
- 6, 0.31, 33

2h total

so it's getting worse with more latent tokens. It seems it having trouble adapting with replacement_method=-1. Maybe I just need more training. 

Also lr might be too high as it spikes the loss at the beginning of epoch?

# 2025-01-28 07:07:16

using 0.5 did eventually start improving after 10k steps


39 min but double the data


what's the right lr?
1e-4 d=0.01 only goes to acc 0.5
what about 1e-5 and 1d=0.001? it gets 0.63 so yeah
1e-6 wd=0.001 0.43 hmm


Hm Ideally I need a better way to work out supressed neurons or hidden states.

Ideally I can use (hs*w_out).diff(), do I need grad?


# 2025-01-29 18:59:47

0.64
0.50

0.71
0.63
0.47

ok don't reset opt seems proimsing

need to get save working
do I need to make coconut a subclass of transformer
and config it a subclass or model or modelconfig?
yeah seems good


# ok I did a long run with 0,5 no good

coconut.utils.Config object at 0x79fe3a95e710>
|      | eval/acc | eval/cot_em | epoch |
| ---: | -------: | ----------: | ----: |
|    0 | 0.719212 |           0 |     0 |
|    1 | 0.576355 |           0 |     1 |
|    2 | 0.458128 |           0 |     2 |
|    3 | 0.251232 |           0 |     3 |


TODO better eval (forward one token at a time)
TODO run eval using transformers

nope can't even replicate wth
this is with -1
|      | eval/acc | eval/cot_em | epoch |
| ---: | -------: | ----------: | ----: |
|    0 | 0.719212 |           0 |     0 |
|    1 | 0.561576 |           0 |     1 |
|    2 | 0.463054 |           0 |     2 |
|    3 | 0.295567 |           0 |     3 |


OK I can't even replciate, probobly 16bit training is the problem!?
Maybe I should use 0.5b and 32bit

|      | eval/acc | eval/cot_em | epoch | minutes |
| ---: | -------: | ----------: | ----: | ------: |
|    0 | 0.246305 |           0 |     2 | 20.8777 |
|    1 | 0.142857 |           0 |     3 | 175.502 |


so even 0.5b 32b weight, 16b training it doesn't work. lets see after the long train...
0.26
0.18
0.08

Wait it did start working!
    {'project': 'coconut', 'save_path': 'outputs/', 'name': 'gsm-qwen', 'only_eval': False, 'coconut': True, 'cot': False, 'no_thoughts': False, 'no_cot': False, 'c_thought': 2, 'epochs_per_stage': 1, 'max_latent_stage': 3, 'pad_latent_to_max': True, 'replacement_method': '-1', 'save_only_improve': True, 'uniform_prob': 0.0, 'model_id': 'plaguss/Qwen2.5-0.5B-Math-Shepherd-PRM-0.2', 'load_model_path': None, 'seed': 0, 'resume': 0, 'bf16': True, 'bf16_weight': False, 'train_path': 'data/gsm_train.json', 'val_path': 'data/gsm_valid.json', 'reset_optimizer': False, 'batch_size_training': 10, 'max_size': 10000, 'debug': False, 'gradient_accumulation_steps': 4, 'num_epochs': 5, 'lr': 0.0001, 'weight_decay': 0.01}
    |      |  eval/acc | eval/cot_em | epoch | minutes |
    | ---: | --------: | ----------: | ----: | ------: |
    |    0 |  0.267857 |           0 |     0 | 9.56295 |
    |    1 |  0.196429 |           0 |     1 | 13.2542 |
    |    2 | 0.0863095 |           0 |     2 | 14.6686 |
    |    3 | 0.0714286 |           0 |     3 | 32.8783 |
    wandb: 🚀 View run gsm-qwen_20250201-071510 at: https://wandb.ai/wassname/coconut/runs/v49wpqas


ok now with 0.5b and 32bit it seems to work eventually hmm

TODO
- test 16but but only linear
- 16 bit but larger batc hsize
- larger model on h100


# Results: gsm-qwen_20250201-122443
{'project': 'coconut', 'save_path': 'outputs/', 'name': 'gsm-qwen', 'only_eval': False, 'coconut': True, 'cot': False, 'no_thoughts': False, 'no_cot': False, 'c_thought': 2, 'epochs_per_stage': 1, 'max_latent_stage': 3, 'pad_latent_to_max': True, 'replacement_method': '-1', 'save_only_improve': True, 'uniform_prob': 0.0, 'model_id': 'plaguss/Qwen2.5-0.5B-Math-Shepherd-PRM-0.2', 'load_model_path': None, 'seed': 0, 'resume': 0, 'bf16': True, 'bf16_weight': False, 'train_path': 'data/gsm_train.json', 'val_path': 'data/gsm_valid.json', 'reset_optimizer': False, 'batch_size_training': 10, 'max_size': 10000, 'debug': False, 'gradient_accumulation_steps': 4, 'num_epochs': 20, 'lr': 0.0001, 'weight_decay': 0.01}
|      |                     eval/acc |                           eval/cot_em | epoch | minutes |
| ---: | ---------------------------: | ------------------------------------: | ----: | ------: |
|    0 |                         0.25 |                                     0 |   nan |     nan |
|    1 |                         0.25 |                                     0 |     0 | 10.4013 |
|    2 |                     0.205357 |                                     0 |   nan |     nan |
|    3 |                     0.205357 |                                     0 |     1 | 16.4774 |
|    4 |                    0.0684524 |                                     0 |   nan |     nan |
|    5 |                    0.0684524 |                                     0 |     2 | 18.9123 |
|    6 |                    0.0535714 |                                     0 |   nan |     nan |
|    7 |                    0.0684524 |                                     0 |   nan |     nan |
|    8 |                    0.0327381 |                                     0 |   nan |     nan |
|    9 |                    0.0416667 |                                     0 |     3 | 63.4671 |
|  18% | ███████████████████████████▎ | 754/4250 [1:03:28<4:54:17,  5.05s/it] |
wandb: 🚀 View run gsm-qwen_20250201-122443 at: https://wandb.ai/wassname/coconut/runs/al3d68tu


Hmm just try to replicate?
Hmm just try with fp32?


Note is the original should get 40% with gpt2
used almost 400k samples, and 50 epochs!!

also they did not use bf16
they did not reset the optimiser. One big run of 25 epochs for CoT. Then 25 for tink
no scheduler?


So the RL experiment used a prompt and a tiny training. Why did that need 8k, but this needs 400k*50!!!. Maybe learning new fundemental behavious takes a long itme. Maybe the prompt helps?
- try with prompt to help initial exploration
- don't reset optimiser!!? how

no no schedule?
might help if I use chatml to tokenize?


### prompts?

https://github.com/Jiayi-Pan/TinyZero/blob/8a623926012ff785f2dc6f3639a821465eed07c4/examples/data_preprocess/countdown.py#L65

"""<|im_start|>system\nFirst think about the reasoning process in the mind and then provides the user with the answer.<|im_end|>\n<|im_start|>user\n Using the numbers {numbers}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in << and >> tags. And return the final answer after ### , for example <<(1 + 2) / 3 = 1>>\n## 1<|im_end|>\n<|im_start|>assistant\nLet me solve this step by step.\n"""


prefix = f"""Reason about the math question then provide your answer. Show your work in <<< and >>> tags or think in <|start-latent|><|latent|><|end-latent|> tags. And return the final answer after ###, for example <<1+2=3>>\n### 3.

prefix = f"""Reason about the math question in <<< and >>> tags or think silently in <|start-latent|><|latent|><|end-latent|> tags. And return the final answer after ###, for example <<1+2=3>>\n### 3.

prefix = f"""Reason and solve the following math question in this format <|start-latent|><|latent|><|end-latent|><<1+2=3>>\n### 3"""

Without prompt or chatml

| model_id          | loss 0.5 | loss 500 | loss 0.5 epochs |
| ----------------- | -------- | -------- | --------------- |
| Qwen 0.5b         | 0.6      | 0.29     | 0.18            |
| 0.5b Math-PRM-0.2 | 0.97     | 0.28     | 0.18            |
| Coder-0.5B        | 0.6      | 0.3      | 0.19            |

0.5b Math-PRM-0.2
with prompt
0steps 3.5
step100, loss 0.5
step 400, 0.2


but acc was only 0.02, wth


![seems to help with loer loss](files/image.png)


# trying with smol model, but god it's so basd and slow

I reckon I will need 1.5b+ to even work, so I should just rent some H100 or two
- USD $2.00 per hour. 3.25 aud
# 
ok revisit 1. gpt2 2. seq-vcr 3. wich alyer 4 batch size




# 2025-05-06 19:26:30

Ok I think that if I add special positional encodings the model might learn faster that it's in another mode

ok with smol2 it's 5:28 mins train, 3:06mins to test the first epoch

eval_1 12mins


TODO can we resume from a stage is saved or passed?
outputs/gsm-smol_20250508-155834/checkpoint_0

OK I'm running, on smol, with the proper number of epochs. Overnight
So it shold do as well as gpt2 if not better that is. GSM9k, 42.9 with CoT, 21.6 without thought, 34.1 coconut. In fact this should do better as it has better pretrasining than gpt2 with the same size


meant to do 25 epochs of CoT First
 
# from the github

https://github.com/facebookresearch/coconut/issues/11

> Hi, thanks for the question.

> We haven't got time to carefully tune the hyper-parameters for larger models like Llama. Generally it's better to use a smaller lr (e.g. 1e-5) and fewer epochs per stage to avoid overfitting.

> We'd like to note that, since these larger models have been pre-trained extensively on language space rather than latent space, you may find coconut at a disadvantage in comparison with language CoT. Future work will explore pre-training in latent space to better unlock the potential of latent reasoning.

https://github.com/facebookresearch/coconut/issues/3

> Hi, thanks for the question. I've confirmed that the exact code can reproduce the reported number. We used 4 GPUs to train the model, which means the effective batch size is 32 * 4 = 128. In the wandb log you shared, the effective batch size seems to be 32?

so changes I need to make


- bs: 128
- ls: 1e-5 with smol2 for speed
- 25 to 50 epochs
- should be around 40% with CoT
- max size 1000000000 not (it's 3012 samples at 32 batch, 45mins)



in the smol2 train they used 3e-4 !
https://github.com/huggingface/alignment-handbook/blob/main/recipes/smollm2/sft/config.yaml
presumably with linear (defualt) but maybe cosine
https://github.com/huggingface/nanotron/blob/c737f00f01e65bc44e7624695351da7ed756ec31/examples/doremi/configs/config_280m_llama.yaml#L69
weight_decay: 0.01


ah butthen unsloth uses linear, and 5e-5 hmm
but we are not continued pretrain we are CoT train
https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Mistral_(7B)-Text_Completion.ipynb#scrollTo=95_Nn-89DhsL
another is cosine 5e-5 weight decay 0 huh



https://github.com/huggingface/alignment-handbook/blob/main/recipes/smollm2/sft/config_smol.yaml
wow smol has 1e-3??
but this would just be at the start of all training since it's cosine so not so good for me hmm

their recipy for fine tune is also 1e-3

note they use trl sft
and they just load args on top


ok I think we could have 
1e-3 with cosine
or 3e-4 with linear


hmm maybe I need to remove qwen
maybe I need to apply chat template?'


hmm I shoudl check the lods

# 2025-05-10 07:04:36

For some reason it doesn't seem to learn in bfloat16.
With 9 epochs of CoT training smol2 model only has 0.6% acc. Weird. I'ts learnt some CoT but it's all wrong.
This is using 1e-3 and no weight decay as reccmended by the model makers.
I might have to go back to Qwen 0.6, and then even 16b and 8b, and hope they work or V100



- Qwen
- omegaconfig again?
  - save config
- either smaller epochs for testing or ... yeah that's easier
- I don't know the right lr... 1e-4 after 20k did not work. incoherent. loss of 1.5. 
maybe warmup too



without sys prompt but with template I get 0.17 loss after 334it. incoherent. 0% CoT match
and with, 1.2% CoT match


yet it made almost no diff in the loss. strange


fixme... why is it only generating 2 tokens when asked for 100? something is weird
this is in eval



did an overnight one 19 epochs
outputs/qwen3-0.6b_20250510-205601/checkpoint_18 
https://wandb.ai/wassname/coconut/runs/4iaa9wbb?nw=nwuserwassname
loss went up a lot, hmm
but it still had that 2 tocken gen. But I can eval all the checkpoints to check and fix
but it did work with only 84% mem alloc!!
seems like grad norm was good as it spiked up, so was warmup'
I could add warmup for adding <start_latent> and adding latent too!
it got a lot slower at epoch 14
2025-05-11 06:11:02.866 | INFO     | coconut.eval:evaluate:110 - Correct=4, CoT_correct=6, Total=500. eval_18                                                                   
2025-05-11 06:11:02.866 | INFO     | coconut.eval:evaluate:111 - Accuracy on val:  4 / 500 =  0.8000%                                                                           
2025-05-11 06:11:02.867 | INFO     | coconut.eval:evaluate:112 - CoT match on val: 6 / 500 =  1.2000%   

this whole thing is token effecient...but not compuete effecient so who cares? or at least during train, what abput durng inf


TODO:
- fix eval. On thos saved models outputs/qwen3-0.6b_20250510-205601/checkpoint_18 


# 2025-05-13 11:34:21

Ok fixed eval. it didn't learn fast, didn't get up to 40%
![alt text](img/mjc_research_journal-1747113708667-image.png)
now try lr=1e-4 (10x)



# 2025-05-15 09:20:41

Ok it got up to 40% on epoch 2. Now I can run expeirments
- try cosine lr on each epoch
- eval/los goes up so might be too high


hmm mine can learn well for one token but not two, interesting
maybe I should rename no schedule stage of stage=0
t

ry lower lr or longer epochst
ry to debug coesne learning in scr
atch nb
try float 32


So now I'd like to experim,ent with a single epoch of each type of hidden states, and see the loss/acc after. Also 32b, vs8b grad, 16b weights

Loss, Acc, Ratio for

| method       | loss | acc  | ratio |
| ------------ | ---- | ---- | ----- |
| none lr=6e-3 | 9.4  | 0.   | 1.002 |
| none lr=6e-4 | 0.5  | 0.16 | 0.94  |
| none lr=6e-5 | 0.65 | 0.19 | 0.951 |
| hs[-1]       | 22   | 0    | 0.926 |

ok none whould work! lets turn of or fix lr and seq_vcr
lr frp, 6e-3 to 6e-4
16% aac. 94$ ratio
0.5 loss

now try 6e-5
19.2% acc
0.951 ratio
0.65 loss


The weird thing is that the initial eal is not good

- hs[-1]
- supr[0.5:-1]
- hs[-2]
- supr[0.5:]+hs[-1]
- supr[0.5:]+ie

and
- hs: b16_w
- hs: 8b grad
- hs: 32b

Start again as I had it messed up


|  method             |   eval/acc |   eval/cot_em |   epoch |   stage |   eval/ratios |   train/minutes |   train/loss |   eval/loss | commnt|
|---          :       |-----------:|--------------:|--------:|-  -----:|  ------------:|----------------:| ------------:|-------------|   -----|
|  load               |      0.449 |         0.107 |      -1 |      -1 |         0.933 |       nan       |     nan      |   nan       |        |
|  1vr,lr=1e-6        |      0.454 |         0.112 |       1 |      -1 |         0.934 |           2.291 |        3.595 |       3.277 |        |
|  1vr,lr=1e-5        |      0.438 |         0.111 |       1 |      -1 |         0.934 |           2.262 |        3.185 |       2.912 |        |
|  1vr,le=1e-4        |      0.375 |         0.100 |       1 |      -1 |         0.912 |           2.311 |        0.848 |       0.966 |        |
|  0sqr,1e-4          |     0.4796 |        0.1115 |       1 |      -1 |        0.9277 |          1.9772 |    0.0611213 |       0.222 |        |
|  vcr2?,1e-4         |     0.1152 |        0.0297 |       1 |      -1 |        0.9324 |           2.262 |     0.190808 |      0.3211 |  |
|  vcr2,1e-5          |     0.4201 |        0.119  |       1 |      -1 |        0.9334 |          2.2748 |     0.118035 |      0.2439 |        |
|  vcrv2 1e-4         |     0.461  |        0.1264 |       1 |      -1 |        0.9229 |          2.3131 |    0.0891597 |      0.2365 | huh this was good       |
|  vr2,opt8b          |     0.4535 |        0.1264 |       1 |      -1 |        0.9224 |          2.2454 |    0.0888447 |       0.236 | no diff |
|vc2,1e-4,bf16w       |     0.4052 |        0.0855 |       1 |      -1 |        0.9077 |          2.2486 |    0.0798283 |      0.2391 |this hurt peft |

python scripts/run.py EpochSingle --opt-8b --bff6_weight --lr=1e-5

when I did it in 32bit it took twice as long, started better, and got worse?? 
python scripts/run.py EpochSingle --no-bf16  --batch-size-training=32 --gradient-accumulation-steps=4 --lr=1e-6

|         | eval/acc | eval/cot_em | epoch | stage | eval/ratios | train/minutes | train/loss | eval/loss |
| ------: | -------: | ----------: | ----: | ----: | ----------: | ------------: | ---------: | --------: |
|       0 |    0.461 |       0.119 |    -1 |    -1 |      0.9321 |           nan |        nan |       nan |
| lr 1e-4 |   0.3011 |       0.119 |     1 |    -1 |      0.9079 |        7.1669 |  0.0911841 |    0.2372 |
| 1r 1e-5 |   0.4572 |       0.145 |     1 |    -1 |      0.9246 |        7.1446 |  0.0896801 |     0.238 |
|    1e-6 |   0.4275 |      0.1264 |     1 |    -1 |      0.9314 |          7.13 |     0.1144 |    0.2468 |
for vcr2 I lowered the lambda for seq-vcr by 1/100 and they still go down but so dos the normal ar loss


|  bf16_weight 1e-5|     0.3978 |        0.1264 |       1 |      -1 |        0.9315 |          2.2409 |     0.100122 |      0.2364 |
|   bf16_weight1e-3 |     0.0074 |        0.0409 |       1 |      -1 |        0.926  |          2.3096 |     0.209068 |      0.3418 |
python scripts/run.py EpochSingle --opt-8b --bf16_weight --lr=1e-5

python scripts/run.py EpochSingle --weight_decay=0 --grad_clip=0 
|  1ithout wd |     0.461  |        0.1264 |       1 |      -1 |        0.9229 |          2.2232 |    0.0891597 |      0.2365 |

62% meme with bf16w and opt8b
77% neither?
82% opt8


python scripts/run.py GsmQwen_H100

next try one epoch of training, but with differen't methods


# 2025-05-18 02:05:26 long run

So learning
- is maintain some of the acc
- but it slowed exponentially as more latent tokens were added! so this is token effecient in test but not in train... which is not attractive
- my methods do seem to work as it performs better with onyl a few samples
![](img/ksnip_20250518-095710.png)

https://wandb.ai/wassname/coconut/runs/xvwpx0dj

```sh
python scripts/run.py EpochSingleLatent  --replacement-method="hs[-1]"
# /workspace/coconut/wandb/run-20250518_022740-nuq8u2c8
python scripts/run.py EpochSingleLatent  --replacement-method="hs[-2]"
python scripts/run.py EpochSingleLatent  --replacement-method="supressed[0.5:]"
python scripts/run.py EpochSingleLatent  --replacement-method="ie+supressed[0.5:]"
python scripts/run.py EpochSingleLatent  --replacement-method="hs[-2]+supressed[0.5:]"
python scripts/run.py EpochSingleLatent  --replacement-method="supressed[0.75:]"
python scripts/run.py EpochSingleLatent  --replacement-method="supressed[0.25:]"
python scripts/run.py EpochSingleLatent  --replacement-method="hs[-3]"
python scripts/run.py EpochSingleLatent  --replacement-method="supressed[0.9:]"
python scripts/run.py EpochSingleLatent  --replacement-method="hs[-4]"
```

|        | eval/acc | eval/cot_em | epoch | stage | eval/ratios | train/minutes | train/loss | eval/loss |
| -----: | -------: | ----------: | ----: | ----: | ----------: | ------------: | ---------: | --------: |
|      0 |        0 |      0.0074 |    -1 |     3 |      0.7401 |           nan |        nan |       nan |
| hs[-1] |    0.052 |      0.0074 |     3 |     3 |      0.6023 |       18.8816 |     0.7113 |    0.7133 |

ah that was stage 3 due to a calc error, lets try stage 1

6 mins instead of 20


If we train for 1 epoch which replacement method is best/

 Config: {'project': 'coconut', 'save_path': 'outputs/', 'name': 'gsm-qwen-0.6bh100', 'model_id': 'suayptalha/Qwen3-0.6B-Math-Expert', 'only_eval': False, 'load_model_path': 'outputs/qwen3-0.6b_20250514-194730/checkpoint_2', 'resume_epochs': 3, 'replacement_method': 'hs[-2]+supressed[0.5:]', 'use_position_ids': True, 'bf16': True, 'bf16_weight': False, 'opt_8b': False, 'cot_epochs': 2, 'epochs_per_stage': 1, 'max_latent_stage': 3, 'num_epochs': 4, 'batch_size_training': 48, 'gradient_accumulation_steps': 3, 'lr': 0.0001, 'weight_decay': 0.01, 'grad_clip': 10.0, 'scheduler': 'cosine', 'debug': False, 'seed': 42, 'reset_optimizer': True, 'loss_seq_vcr': True, 'max_size': 8000, 'c_thought': 2, 'pad_latent_to_max': True, 'uniform_prob': 0.0, 'train_path': 'data/gsm_train.json', 'val_path': 'data/gsm_valid.json', 'system_prompt': ''}

|                        | eval/acc | eval/cot_em | epoch | stage | eval/ratios | train/minutes | train/loss | eval/loss |
| ---------------------: | -------: | ----------: | ----: | ----: | ----------: | ------------: | ---------: | --------: |
|       supressed[0.75:] |   0.3383 |      0.0074 |     3 |     1 |      0.9259 |       10.1804 |   0.421989 |    0.4588 |
|       supressed[0.90:] |   0.2379 |      0.0112 |     3 |     1 |      0.9264 |        10.126 |   0.364225 |    0.4098 |
|                 hs[-4] |   0.2342 |      0.0112 |     3 |     1 |      0.9199 |        8.4421 |   0.352921 |    0.3992 |
|                 hs[-3] |   0.2268 |      0.0112 |     3 |     1 |      0.9202 |        8.3769 |   0.341455 |    0.3993 |
|        supressed[0.5:] |    0.223 |      0.0112 |     3 |     1 |       0.924 |       10.2664 |   0.537695 |    0.5329 |
|     ie+supressed[0.5:] |   0.2156 |      0.0112 |     3 |     1 |       0.924 |       10.3073 |   0.540861 |    0.5356 |
|                 hs[-2] |   0.1896 |      0.0149 |     3 |     1 |      0.9175 |        8.4899 |   0.341802 |    0.3987 |
| hs[-2]+supressed[0.5:] |   0.1784 |      0.0074 |     3 |     1 |      0.9456 |       10.3819 |   0.733456 |     0.582 |
|                 hs[-1] |   0.1747 |      0.0112 |     3 |     1 |      0.9452 |        8.7211 |   0.420122 |    0.4459 |
|       supressed[0.25:] |   0.1487 |           0 |     3 |     1 |      0.9287 |       10.6798 |   0.613252 |     0.587 |



python scripts/run.py GsmQwen_H100



| TRM | 0.168 | 0.04 | 3 | 1 | 0.94 | 26 | ? | ? |

# 2025-10-11 21:11:43

    # TRM-Style Detached Recursions Implementation

    ## Summary

    Minimal implementation of TRM-style training where early recursive passes are detached (no gradients), and only the last N passes backpropagate gradients. This forces the model to learn to clean up its own accumulated errors.


    ## How It Works

    **Example**: If `max_n_latents=4` and `n_detached_recursions=2`:
    - **Pass 0,1**: Run with `torch.no_grad()` 
    - Model does "blind" recursions
    - Accumulates its own errors/junk
    - No gradient computation (faster, less memory)
    
    - **Pass 2,3**: Run with gradients enabled
    - Model sees accumulated errors from passes 0,1
    - Learns to clean up and be robust to its own mistakes
    - Gradients flow back to update weights

    ## Key Insight from TRM Paper

    > "After many of its own steps, it will be filled with its own junk, it will have to learn to clean it up! And that cleaning step might be the element needed to make it stable and convergent!"

    This is like training a denoising autoencoder on the model's own outputs - it learns an implicit error correction dynamic.

    ## Testing

    ### Using Config Classes (Recommended)

    Run the TRM experiment:
    ```bash
    uv run python scripts/run.py TRMTest
    ```

    ## Expected Results

    - **Memory**: Slightly lower during detached passes (no gradient storage)
    - **Speed**: Slightly faster overall (fewer backward passes)
    - **Stability**: Potentially more stable training (error correction mechanism)
    - **Accuracy**: Should match or exceed baseline if hypothesis is correct

    ## Next Steps

    1. Run full experiment comparing n_detached_recursions=0,1,2,3
    2. Monitor loss curves and accuracy
    3. If promising, try with larger models on H100
    4. Consider adding EMA (Exponential Moving Average) for extra stability


# 2025-10-12 10:52:50

Where are we at?


I do the first few epochs and save a checkpoint, since it doesn't differ.
This lets me do follow up experiments more quickly.

# 2025-10-14 18:12:23

# TRM-Style Recursive Reasoning for Small LLMs


## Core Idea

Apply Tiny Recursive Model (TRM) principles to pretrained LLMs using two learned components:
1. **Transcoder**: Bridges output hidden states → input embeddings (the format mismatch problem)
2. **Recurser**: Tiny 2-layer transformer that iteratively refines hidden states

Only backprop through full LLM once at the end, avoiding Coconut's 3-day training bottleneck.

Refs:

- docs/trm_paper.md
- docs/trm_reference_code/models/layers.py
- docs/trm_reference_code/models/recursive_reasoning/trm.py
- docs/trm_reference_code/pretrain.py
- coconut/trm_layers.py


so it's TRM but I'm modifying the idea to apply to LLM's coconut style, so the idea is TRM but it's applied to the <latent> tokens in coconut, and just like TRM we detach all but the last 2 recursions. This lets us use a quantised LLM and just learn a TRM that takes in the output hidden state `hs.detach()` from the LLM processing every token up to <start-latent> then TRM works on the latents, then the network decodes the final hidden state to tokens `hs->output`

There one additional problem which is that LLM's are not used to receiving their own hidden states as input, so we need a small transcoder network to convert the hidden state to the embedding space using the output_head. This is like the "format mismatch" problem in coconut, but we solve it explicitly with a small network. I thought that output_head would naturally learn the transcode, and the TRM would natrually learn to work in the output embedding space

we should reuse the coconut code as much as possible, so read README.md and justfile, and coconut.py

Here's my attempt at modifying the TRM psudocode from the paper to add the LLM wrapper. Again this is my modification of TRM not TRM's original psudocode

```py
# where output_head converts zH to input embeddings
# where x are output hidden states from LLM
# hs are embeddings 
# where the llm is 4bit and frozen

def hrm(z, x, n=2, T=2): # hierarchical reasoning
    zH, zL = z
    with torch.no_grad():
        for i in range(nT - 2):
            zL = L_net(zL, zH, x)
            if (i + 1) % T == 0:
                zH = H_net(zH, zL)
    # 1-step grad
    zL = L_net(zL, zH, x)
    zH = H_net(zH, zL)
    return (zH, zL), output_head(zH), Q_head(zH)

def ACT_halt(q, y_hat, y_true):
    target_halt = (y_hat == y_true)
    loss = 0.5*binary_cross_entropy(q[0], target_halt)
    return loss

def ACT_continue(q, last_step):
    if last_step:
        target_continue = sigmoid(q[0])
    else:
        target_continue = sigmoid(max(q[0], q[1]))
    loss = 0.5*binary_cross_entropy(q[1], target_continue)
    return loss

# Deep Supervision
for x_input, y_true in train_dataloader:
    z = z_init
    for step in range(N_sup): # deep supervision
        with torch.no_grad():
            # LLM converts input tokens to output hidden states
            x_hs = LLM(x_input).hidden_states[-1]
        z, embed_pred, q = hrm(z, x_hs)
        y_pred = LLM(embed_pred) # new
        loss = loss_fn(y_pred, y_true)
        # Adaptive computational time (ACT) using Q-learning
        loss += ACT_halt(q, y_pred, y_true)
        _, _, q_next = hrm(z, x_hs) # extra forward pass
        loss += ACT_continue(q_next, step == N_sup - 1)
        z = z.detach()
        loss.backward()
        opt.step()
        opt.zero_grad()
        if q[0] > q[1]: # early-stopping
            break
```
Figure 2: Pseudocode of Hierarchical Reasoning Models (HRMs).


So we use a frozen 8bit or 4bit llm for the encoder, out output head is now a transcoder that converts hidden states to embeddings,and we decode the final hidden state with the llm rollout from <think> to full output chain of thought.

ok but for TRM modules look at layers.py and trm.py ema.py, losses.py how much do we need to carry over? and you designed your own... why? I don't think you shoul;d

we are looking for minimal modification to this repo's coconut, we can probobly modify the config, coconut.py forward, bring in trm/*.py as needed,

coconut/
├── __init__.py
├── coconut.py          # Modify: add use_trm mode
├── configs.py          # Modify: add TRM config
├── trm_layers.py       # NEW: copy from TRM reference
└── trm_adapter.py      # NEW: minimal TRM wrapper for LLM

Minimal Modification Strategy
What to Copy from TRM Reference Code:
layers.py → coconut/trm_layers.py

Already have Attention, SwiGLU, rms_norm, etc.
Use these building blocks directly
Core TRM Components (simplified from trm.py):

TinyRecursiveReasoningModel_ACTV1Block - the 2-layer transformer
TinyRecursiveReasoningModel_ACTV1ReasoningModule - wrapper for recursion
Skip ACT/Q-learning initially (too complex)

Looking at the progress so far and your plan, here's a comprehensive task list:

## TRM Integration Task List

### Core Implementation
- [x] Copy TRM layers over as needed
  - [x] Create trm_layers.py with core building blocks (Attention, SwiGLU, RMSNorm)
  - [x] Create trm_adapter.py with TRMBlock, TRMRecurser, TRMTranscoder, CoconutTRM
- [x] Modify config
  - [x] Add TRM config fields to configs.py (use_trm, load_in_4bit/8bit, trm_num_layers, etc.)
- [ ] Modify coconut.py forward and loss
  - [ ] Add `CoconutTRM` initialization in `__init__` when `use_trm=True`
  - [ ] Add `_forward_trm()` method for TRM path
  - [ ] Modify `forward()` to route to TRM path when `use_trm=True`
  - [ ] Handle frozen LLM loading (4bit/8bit via bitsandbytes)
- [ ] Create TRM config class
  - [ ] Add `GSMQwenTRM` config in configs.py with use_trm=True, load_in_4bit=True, n_detached_recursions=2
- [ ] Update run.py to handle TRM config name

### Testing & Debugging
- [ ] Try running with `uv run python scripts/run.py TRM`

Next immediate action: Modify coconut.py to add TRM integration.

fyi as you go, I do like jaxtyping and einops, so if you are rewriting any they are nice and as a bonux jaxtyping for pytorch lets us check dims e.g.
```
# https://github.com/patrick-kidger/jaxtyping?tab=readme-ov-file
from jaxtyping import Float, Int
x: Float[Tensor, 'batch seq hs'] = torch.rand(...
# https://github.com/beartype/beartype
from beartype.claw import beartype_this_package       # <-- hype comes
beartype_this_package()                               # <-- hype goes
```

# 2025-10-15 18:28:19
```
Question → Frozen LLM Encoder → Context Hidden States
                                        ↓
                              TRM Recursive Reasoning
                              (detached + gradient passes)
                                        ↓
                                   Transcoder
                                        ↓
                              Latent Embeddings
                                        ↓
                           Frozen LLM Decoder → Answer
```


# 2025-10-16 08:40:35

    **What we implemented:**

    1. **Dual-network HRM architecture** (`L_net` + `H_net`) matching the paper's Figure 2 pseudocode
    2. **Frozen quantized LLM** (4bit) with only `lm_head` trainable for gradient flow
    3. **Internal HRM recursions** with detached early steps (n_detached=2) and final steps with gradients
    4. **Memory-efficient forward pass**: LLM encode (once) → HRM recursions (cheap) → LLM decode (once)

    **Key architectural decision:**

    The paper's full deep supervision (N_sup outer loop with backprop after each step) would require 4x LLM forward passes = OOM. Instead, we do:
    - LLM encode question (frozen, detached): ~6GB
    - HRM internal recursions (tiny model, <1GB): cheap
    - LLM decode (frozen backbone, trainable lm_head): ~6GB
    - Total: ~13GB fits in 24GB

    **What's working:**
    - Code runs without syntax errors
    - Model loads correctly with 4bit quantization
    - HRM adapter initialized with dual networks
    - Training started (got to evaluation phase)

    **Remaining limitation:**
    The deep supervision loop from the paper isn't implemented due to memory constraints. The HRM still does multi-step refinement internally (via n_detached recursions), just without the outer supervision loop. This is a reasonable tradeoff for consumer GPUs.

    Made changes.

# 2025-10-16 14:48:59

Ok it's not learning. But I don't have deep supervision, which might be expensive in this setup. Perhaps I need it.

How can I get it, with the benefits, with some approximation that doesn't have one LLM rollout per supervision step?

# Base acc

For model_id: str = "outputs/qwen3-0.6b_20250514-194730/checkpoint_2"

which is the one that's trained to produce CoT math solutions in the format

Extracted llm Output: `300` (=? 300) ✅.
ideal_CoT = '<<4-2=2>>
        <<2/.5=4>>
        <<12/4=3>>
        <<100*3=300>>'.
Answer = '300' .

Acc is 44.9%

    ██████████████████████████████| 19/19 [02:36<00:00,  8.25s/it]
    2025-10-16 17:42:40.466 | INFO     | coconut.eval:evaluate:112 - Correct=150, CoT_correct=4, Total=303. eval_2_start                             
    2025-10-16 17:42:40.471 | INFO     | coconut.eval:evaluate:113 - Accuracy on val:  150 / 303 =  49.5050%                                         
    2025-10-16 17:42:40.477 | INFO     | coconut.eval:evaluate:114 - CoT match on val: 4 / 303 =  1.3201% 


This run
epoch1:

    loss: 0.4
    eval_loss 0.5
    full eval: 47.85
    CoT acc = 1.32

epoch2:
- loss 0.35
- eval_acc 0.49
- full acc: 0.48.5
- CoT acc 1.32
 nll_ans/nll_corrupted_ans = 0.9002
 
loss 0.54
loss 0.48
test acc 0.5
full acc: 49.83
CoT acc 1.32
nll_ans/nll_corrupted_ans = 0.6478

# Results: trm-qwen3-0.6b_20251017-075826

    {'project': 'coconut', 'save_path': 'outputs/', 'name': 'trm-qwen3-0.6b', 'model_id': 'outputs/qwen3-0.6b_20250514-194730/checkpoint_2', 'only_eval': False, 'load_model_path': '', 'resume_epochs': 2, 'replacement_method': 'supressed[0.75:]', 'use_position_ids': True, 'bf16': True, 'bf16_weight': False, 'opt_8b': False, 'cot_epochs': 0, 'epochs_per_stage': 2, 'max_latent_stage': 3, 'num_epochs': 6, 'batch_size_training': 16, 'gradient_accumulation_steps': 8, 'lr': 0.0001, 'weight_decay': 0.5, 'grad_clip': 10.0, 'scheduler': 'cosine', 'debug': False, 'seed': 42, 'reset_optimizer': False, 'loss_seq_vcr': False, 'n_detached_recursions': 2, 'use_trm': True, 'load_in_4bit': True, 'load_in_8bit': False, 'trm_n_sup': 4, 'trm_num_layers': 2, 'trm_num_heads': 8, 'trm_expansion': 2.67, 'max_size': 9000, 'c_thought': 2, 'pad_latent_to_max': True, 'uniform_prob': 0.0, 'train_path': 'data/gsm_train.json', 'val_path': 'data/gsm_valid.json', 'system_prompt': '', 'latent_token_id': None, 'eval_first_epoch': False, 'n_gradient_recursions': 2}

|      | eval/acc | eval/cot_em | eval/ratios | epoch | stage | train/minutes | train/loss | eval/loss |
| ---: | -------: | ----------: | ----------: | ----: | ----: | ------------: | ---------: | --------: |
|    0 |   0.4785 |      0.0132 |      0.9074 |     2 |     1 |        14.615 |   0.403016 |    0.5406 |
|    1 |   0.4851 |      0.0132 |      0.9002 |     3 |     1 |       14.4587 |   0.345151 |    0.4847 |
|    2 |   0.4983 |      0.0132 |      0.6478 |     4 |     2 |       16.2101 |   0.483461 |     0.623 |
|    3 |    0.495 |      0.0132 |      0.6454 |     5 |     2 |       15.1292 |   0.420522 |    0.6093 |



# Results: trm-qwen3-0.6b_20251017-154402
{'project': 'coconut', 'save_path': 'outputs/', 'name': 'trm-qwen3-0.6b', 'model_id': 'outputs/qwen3-0.6b_20250514-194730/checkpoint_2', 'only_eval': False, 'load_model_path': '', 'resume_epochs': 8, 'replacement_method': 'supressed[0.75:]', 'use_position_ids': True, 'bf16': True, 'bf16_weight': False, 'opt_8b': False, 'cot_epochs': 0, 'epochs_per_stage': 8, 'max_latent_stage': 3, 'num_epochs': 50, 'batch_size_training': 16, 'gradient_accumulation_steps': 8, 'lr': 0.0001, 'weight_decay': 0.5, 'grad_clip': 10.0, 'scheduler': 'cosine', 'debug': False, 'seed': 42, 'reset_optimizer': False, 'loss_seq_vcr': False, 'n_detached_recursions': 2, 'use_trm': True, 'load_in_4bit': True, 'load_in_8bit': False, 'trm_n_sup': 4, 'trm_num_layers': 2, 'trm_num_heads': 8, 'trm_expansion': 2.67, 'max_size': 20000, 'c_thought': 2, 'pad_latent_to_max': True, 'uniform_prob': 0.0, 'train_path': 'data/gsm_train.json', 'val_path': 'data/gsm_valid.json', 'system_prompt': '', 'latent_token_id': None, 'eval_first_epoch': False, 'n_gradient_recursions': 2}
|      | eval/acc | eval/cot_em | eval/ratios | epoch | stage | train/minutes | train/loss | eval/loss |
| ---: | -------: | ----------: | ----------: | ----: | ----: | ------------: | ---------: | --------: |
|    0 |    0.486 |       0.016 |      0.9312 |     8 |     1 |       21.5086 |   0.561535 |    0.6984 |
|    1 |    0.492 |       0.016 |      0.9315 |     9 |     1 |       22.7353 |    0.44593 |    0.5652 |
|    2 |    0.492 |       0.016 |      0.9317 |    10 |     1 |       22.6433 |   0.392704 |    0.5235 |
|    3 |    0.488 |       0.016 |      0.9322 |    11 |     1 |        21.784 |   0.326752 |    0.5187 |
|    4 |    0.484 |       0.016 |      0.9312 |    12 |     1 |       21.7898 |    0.32877 |    0.5095 |
|    5 |    0.492 |       0.016 |      0.9334 |    13 |     1 |       21.8261 |   0.437547 |    0.5079 |
|    6 |     0.48 |       0.016 |      0.9312 |    14 |     1 |        21.833 |   0.366916 |    0.5013 |
|    7 |    0.494 |       0.016 |      0.9305 |    15 |     1 |       22.3296 |   0.344434 |    0.5006 |
|    8 |    0.484 |       0.018 |      0.6817 |    16 |     2 |       22.2727 |   0.514602 |    0.6541 |
|    9 |     0.48 |       0.018 |      0.6934 |    17 |     2 |       22.1381 |   0.479511 |    0.6479 |
|   10 |    0.482 |       0.018 |      0.7048 |    18 |     2 |       18.5789 |   0.534687 |    0.6368 |
|   11 |    0.486 |       0.018 |      0.7079 |    19 |     2 |       18.2682 |   0.470743 |    0.6315 |
|   12 |    0.484 |       0.018 |      0.7107 |    20 |     2 |       18.3157 |   0.507592 |    0.6268 |
|   13 |    0.492 |       0.018 |      0.7115 |    21 |     2 |       18.4808 |   0.431141 |    0.6293 |
|   14 |    0.484 |       0.018 |      0.7142 |    22 |     2 |       18.2401 |   0.411453 |    0.6298 |
|   15 |    0.486 |       0.018 |      0.7046 |    23 |     2 |       18.3428 |   0.504539 |    0.6339 |
|   16 |    0.492 |       0.018 |      0.5844 |    24 |     3 |        20.962 |   0.445394 |    0.7045 |
|   17 |    0.496 |       0.018 |      0.5847 |    25 |     3 |       20.5289 |   0.487632 |    0.7121 |
|   18 |      0.5 |       0.018 |      0.5905 |    26 |     3 |       20.5692 |   0.497739 |    0.7127 |
|   19 |    0.502 |       0.018 |      0.5861 |    27 |     3 |       20.4389 |   0.549286 |    0.7074 |
|   20 |    0.502 |       0.018 |      0.5908 |    28 |     3 |       20.3751 |   0.458084 |    0.7296 |
|   21 |    0.502 |       0.018 |      0.5873 |    29 |     3 |       20.5997 |   0.623392 |    0.7353 |
|   22 |    0.502 |       0.018 |      0.5839 |    30 |     3 |       20.6925 |   0.545111 |    0.7474 |
|   23 |      0.5 |       0.018 |      0.5799 |    31 |     3 |       20.5603 |   0.535096 |    0.7552 |
|   24 |    0.496 |       0.018 |      0.5852 |    32 |     3 |       21.0831 |    0.45221 |    0.7714 |
|   25 |    0.496 |       0.018 |      0.5836 |    33 |     3 |       21.5744 |   0.540627 |    0.7576 |
|   26 |    0.494 |       0.018 |      0.5829 |    34 |     3 |       21.3658 |   0.581768 |    0.7979 |
|   27 |    0.506 |       0.018 |      0.5863 |    35 |     3 |       21.5139 |     0.6375 |    0.8009 |
|   28 |    0.496 |       0.018 |        0.58 |    36 |     3 |       21.5572 |   0.475648 |    0.7865 |
|   29 |    0.496 |       0.018 |       0.579 |    37 |     3 |       21.7109 |   0.527833 |    0.8089 |
|   30 |    0.498 |       0.018 |      0.5907 |    38 |     3 |       21.7482 |   0.581697 |    0.7583 |
|   31 |      0.5 |       0.018 |      0.5853 |    39 |     3 |       21.5036 |   0.644376 |    0.7743 |
|   32 |     0.49 |       0.018 |      0.5833 |    40 |     3 |       21.5312 |   0.539744 |    0.7824 |
|   33 |    0.498 |       0.018 |       0.583 |    41 |     3 |       21.5114 |   0.592342 |    0.7168 |
|   34 |    0.496 |       0.018 |      0.5925 |    42 |     3 |       21.4841 |   0.543578 |    0.7813 |
|   35 |    0.496 |       0.018 |      0.5856 |    43 |     3 |        21.531 |   0.604821 |    0.7597 |
|   36 |    0.498 |       0.018 |      0.5825 |    44 |     3 |       21.5864 |   0.612349 |    0.7686 |
|   37 |      0.5 |       0.018 |      0.5826 |    45 |     3 |       21.1743 |   0.544882 |    0.7106 |
|   38 |    0.496 |       0.018 |      0.5931 |    46 |     3 |       21.5019 |   0.617497 |    0.6941 |
|   39 |    0.504 |       0.018 |       0.599 |    47 |     3 |       21.1994 |   0.611732 |    0.7067 |
|   40 |    0.498 |       0.018 |      0.5882 |    48 |     3 |       21.4256 |   0.530537 |    0.7298 |
|   41 |      0.5 |       0.018 |      0.5853 |    49 |     3 |       22.1675 |   0.510084 |    0.7242 |
 14h 44m 17s
wandb: 
wandb: 🚀 View run trm-qwen3-0.6b_20251017-154402 at: 
wandb: Find logs at: wandb/run-20251017_154406-cekkxhs8/logs

Hm I wonder if a higher lr, or no scheduelr

Hm so it overfits too
https://wandb.ai/wassname/coconut/runs/cekkxhs8?nw=nwuserwassname
so it's stable but it overfits




Ratios (nll_ans/nll_corrupted_ans) drop sharply post-stage 1, hinting mode collapse to safe (non-latent) outputs. 


TODO load
- [ ] 2025-10-18 06:28:10.828 | INFO     | coconut.load_model:save_model:114 - saving model outputs/trm-qwen3-0.6b_20251017-154402/checkpoint_49/pytorch_model.safetensors
- [ ] fix gen, then run eval.
- [ ] add train/ratios?


# 2025-10-18 15:37:52

So I fixed generate so that eval now uses TRM, and I got an initial drop, but then it went up. Ratios went up, when it should go down.


- [x] fix generate and eval
- [x] I also change it use hidden state n-4



## Results: trm-qwen3-0.6b_20251018-141208
{'project': 'coconut', 'save_path': 'outputs/', 'name': 'trm-qwen3-0.6b', 'model_id': 'outputs/qwen3-0.6b_20250514-194730/checkpoint_2', 'only_eval': False, 'load_model_path': '', 'resume_epochs': 8, 'replacement_method': 'supressed[0.75:]', 'use_position_ids': True, 'bf16': True, 'bf16_weight': False, 'opt_8b': False, 'cot_epochs': 0, 'epochs_per_stage': 8, 'max_latent_stage': 3, 'num_epochs': 12, 'batch_size_training': 16, 'gradient_accumulation_steps': 8, 'lr': 0.0001, 'weight_decay': 0.1, 'grad_clip': 10.0, 'scheduler': 'linear', 'debug': False, 'seed': 42, 'reset_optimizer': False, 'loss_seq_vcr': False, 'n_detached_recursions': 2, 'use_trm': True, 'load_in_4bit': True, 'load_in_8bit': False, 'trm_n_sup': 4, 'trm_num_layers': 2, 'trm_num_heads': 8, 'trm_expansion': 2.67, 'max_size': 20000, 'c_thought': 2, 'pad_latent_to_max': True, 'uniform_prob': 0.0, 'train_path': 'data/gsm_train.json', 'val_path': 'data/gsm_valid.json', 'system_prompt': '', 'latent_token_id': None, 'eval_first_epoch': False, 'n_gradient_recursions': 2}
|      | eval/acc | eval/cot_em | eval/ratios | epoch | stage | train/minutes | train/loss | eval/loss |
| ---: | -------: | ----------: | ----------: | ----: | ----: | ------------: | ---------: | --------: |
|    0 |    0.304 |       0.022 |      0.9298 |     8 |     1 |       20.7967 |   0.479528 |    0.5213 |
|    1 |    0.328 |       0.014 |      0.9301 |     9 |     1 |       20.3774 |   0.361955 |    0.4921 |
|    2 |    0.326 |       0.014 |       0.932 |    10 |     1 |       21.0332 |   0.295458 |    0.4851 |
|    3 |    0.288 |       0.012 |      0.9358 |    11 |     1 |       21.0486 |   0.258795 |    0.4792 |


- [x] use 1 latent per stage: c_thought->1


@/README.md @/coconut  @/coconut/trm_layers.py  hopefully you can see what I'm doing from the psudocode in the readme

anyway I'm thinking that

right now each call of TRM taken in hidden[n-3] and output input_embedding. It must recurse on hs. but also convert hs and z into input embedding

would it not be better to have it convert z and input_embed to new input embed. It can still take in hs for context thought as it's usefull?


The otehr thing is... should I make TRM a true adapter? Like LoRA, how much more complex would that be... it would have to hook into a layer and modify the hidden state... so no longer would it return input_embed (less info) but instead a modified hidden[n-3] which is a much more information rich place to put modifications, and is better supported by LORA type papers, and the out_head has a much easier job modifying hidden rather than converting hidden to input embed

changes
- [x] learn addition to embed, not embed
- [x] use last hidden state not mean
- [x] 1 latent thought not 2 per stage
- [ ] linear schedule

ok now it starts at
- eval/acc 0.304 -> 0.318! -> 0.25 ?
- eval/cot_em 0.022 -> 0.014 -> 1
- eval/ratios 0.9298 -> 0.9313 ! -> 0.9281
- loss 0.47 -> ?
- 20min to 10min (due to 1 latent per stage)

# 2025-10-18 16:17:55

### Step-by-Step Implementation Steps

    If this plan sounds good, here's how I'd implement in ACT mode:

    1. Update trm_adapter.py: Change TRMTranscoder to output hidden_size (not embedding size if different). Return trm_delta instead of input_embed_diff.

    2. In coconut.py forward loop:

    - After initial outputs = model.forward(...), get original_hs = hidden_states[-4]
    - Compute trm_delta = trm(original_hs, zL_prev, zH_prev)[0]
    - modified_hs = original_hs + trm_delta
    - Then, custom re_forward_from_layer(model, modified_hs, kv_cache_up_to_-4, position_ids, etc.) to get updated outputs/logits/KV.
    - Proceed to next latent with updated KV.

    3. Test: Add logging to verify modified hs affects outputs correctly.



> To make it adapter-like: Simplest way is to keep multi-pass but change injection point. Compute TRM delta for hs[-4], then re-forward from layer -4 with added delta, using KV cache up to that point. This avoids full hooks but requires splitting the LLM forward.

Hm I wonder if that's easy, or use TraceDict from Baukit. Or reuse PEFT type hooks.

> Currently, it processes detached hidden_states[-4] to produce an additive diff for the input embedding at latent positions. You want it to instead recurse on hidden states and output a modified hidden[n-3] (i.e., hidden_states[-4]), which is richer for modifications and better aligns with adapter literature.

Do you think so, or is input_embedding just as rich. I mean it goes directly into the residual stream anyway... so we might be able to generate it at any stage

# 2025-10-18 19:41:11

# Results: trm-qwen3-0.6b_20251018-155617
{'project': 'coconut', 'save_path': 'outputs/', 'name': 'trm-qwen3-0.6b', 'model_id': 'outputs/qwen3-0.6b_20250514-194730/checkpoint_2', 'only_eval': False, 'load_model_path': '', 'resume_epochs': 8, 'replacement_method': 'supressed[0.75:]', 'use_position_ids': True, 'bf16': True, 'bf16_weight': False, 'opt_8b': False, 'cot_epochs': 0, 'epochs_per_stage': 8, 'max_latent_stage': 3, 'num_epochs': 16, 'batch_size_training': 16, 'gradient_accumulation_steps': 8, 'lr': 0.0001, 'weight_decay': 0.1, 'grad_clip': 10.0, 'scheduler': 'linear', 'debug': False, 'seed': 42, 'reset_optimizer': False, 'loss_seq_vcr': False, 'n_detached_recursions': 2, 'use_trm': True, 'load_in_4bit': True, 'load_in_8bit': False, 'trm_n_sup': 4, 'trm_num_layers': 2, 'trm_num_heads': 8, 'trm_expansion': 2.67, 'max_size': 20000, 'c_thought': 1, 'pad_latent_to_max': True, 'uniform_prob': 0.0, 'train_path': 'data/gsm_train.json', 'val_path': 'data/gsm_valid.json', 'system_prompt': '', 'latent_token_id': None, 'eval_first_epoch': False, 'n_gradient_recursions': 2}
|      | eval/acc | eval/cot_em | eval/ratios | epoch | stage | train/minutes | train/loss | eval/loss |
| ---: | -------: | ----------: | ----------: | ----: | ----: | ------------: | ---------: | --------: |
|    0 |    0.318 |       0.014 |      0.9313 |     8 |     1 |       13.5332 |   0.532149 |    0.5204 |
|    1 |    0.272 |       0.012 |      0.9305 |     9 |     1 |       13.6789 |   0.353447 |    0.4924 |
|    2 |    0.252 |        0.01 |      0.9281 |    10 |     1 |       14.4978 |   0.296711 |    0.4819 |
|    3 |    0.254 |        0.01 |      0.9247 |    11 |     1 |       14.8083 |   0.272362 |    0.4771 |
|    4 |     0.25 |        0.01 |       0.928 |    12 |     1 |        14.531 |   0.292608 |    0.4744 |
|    5 |    0.218 |       0.008 |      0.9256 |    13 |     1 |        14.801 |   0.371305 |    0.4723 |
|    6 |    0.226 |        0.01 |      0.9247 |    14 |     1 |        14.551 |   0.324713 |    0.4692 |
|    7 |    0.202 |       0.012 |      0.9293 |    15 |     1 |       13.2352 |   0.308156 |     0.468 |

Well that seems disapointing, the loss went down, it did not overfit, but the acc went down.

# 2025-10-18 19:41:14

now try with persistant steering

big loss of 7 to start with

# Results: trm-qwen3-0.6b_20251018-201519
{'project': 'coconut', 'save_path': 'outputs/', 'name': 'trm-qwen3-0.6b', 'model_id': 'outputs/qwen3-0.6b_20250514-194730/checkpoint_2', 'only_eval': False, 'load_model_path': '', 'resume_epochs': 8, 'replacement_method': 'supressed[0.75:]', 'use_position_ids': True, 'bf16': True, 'bf16_weight': False, 'opt_8b': False, 'cot_epochs': 0, 'epochs_per_stage': 8, 'max_latent_stage': 3, 'num_epochs': 16, 'batch_size_training': 16, 'gradient_accumulation_steps': 8, 'lr': 0.0001, 'weight_decay': 0.1, 'grad_clip': 10.0, 'scheduler': 'linear', 'debug': False, 'seed': 42, 'reset_optimizer': False, 'loss_seq_vcr': False, 'n_detached_recursions': 2, 'use_trm': True, 'load_in_4bit': True, 'load_in_8bit': False, 'trm_n_sup': 4, 'trm_num_layers': 2, 'trm_num_heads': 8, 'trm_expansion': 2.67, 'trm_persistent_steering': True, 'max_size': 20000, 'c_thought': 1, 'pad_latent_to_max': True, 'uniform_prob': 0.0, 'train_path': 'data/gsm_train.json', 'val_path': 'data/gsm_valid.json', 'system_prompt': '', 'latent_token_id': None, 'eval_first_epoch': False, 'n_gradient_recursions': 2}
|      | eval/acc | eval/cot_em | eval/ratios | epoch | stage | train/minutes | train/loss | eval/loss |
| ---: | -------: | ----------: | ----------: | ----: | ----: | ------------: | ---------: | --------: |
|    0 |    0.322 |       0.046 |      0.9715 |     8 |     1 |       14.4724 |    5.85434 |    6.1491 |
|    1 |     0.29 |       0.036 |      0.9655 |     9 |     1 |       14.4789 |    5.55748 |    5.9305 |
|    2 |     0.27 |       0.038 |      0.9726 |    10 |     1 |       14.5137 |    5.37045 |    5.7498 |
|    3 |    0.258 |        0.03 |      0.9827 |    11 |     1 |       14.5729 |    5.19475 |    5.5971 |
|    4 |    0.258 |       0.034 |      0.9927 |    12 |     1 |       14.5285 |    5.16986 |    5.4678 |
|    5 |    0.242 |       0.026 |      0.9813 |    13 |     1 |       14.5436 |    5.21671 |    5.3621 |
|    6 |    0.236 |       0.026 |      0.9954 |    14 |     1 |        14.612 |    4.68249 |    5.2943 |
|    7 |    0.252 |       0.026 |      1.0032 |    15 |     1 |       14.5252 |    4.74156 |    5.2109 |


wih trm_persistent_steering, loss starts at 6
without starts at 1


1 epoch of perisistent steering was 
    2025-10-19 07:04:49.891 | INFO     | coconut.eval:evaluate:112 - Correct=159, CoT_correct=6, Total=500. eval_8                                       
    2025-10-19 07:04:49.892 | INFO     | coconut.eval:evaluate:113 - Accuracy on val:  159 / 500 =  31.8000%                                             
    2025-10-19 07:04:49.893 | INFO     | coconut.eval:evaluate:114 - CoT match on val: 6 / 500 =  1.2000%                                                
    process_dataset: cot_latent_1 (num_proc=12): 100%|█████████████████████████████████████████████████████████| 500/500 [00:01<00:00, 493.06 examples/s]
    PPX: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████| 32/32 [00:13<00:00,  2.33it/s]
    2025-10-19 07:05:04.735 | INFO     | coconut.eval:get_answer_preference:390 - ratio nll_ans/nll_corrupted_ans = 0.9283   
    loss 0.44
    Full llm output: `['<<', '1', '2', '*', '1', '2', '=', '1', '4', '4', '>>\n', '<<', '1', '0', '0', '/', '1', '2', '=', '8', '.', '3', '3', '>>\n', '<<', '8', '.', '3', '3', '*', '1', '4', '4', '=', '1', '2', '0', '0', '.', '3', '2', '>>\n', '###', ' ', '1', '2', '0', '0', '.', '3', '2', '\n', '<|im_end|>', '\n', '<|im_end|>', '\n', '<|im_end|>', '\n', '<|im_end|>', '\n', '<|im_end|>', '\n', '<|im_end|>', '\n']`. 
    Extracted llm Output: `1200.32` (=? 300) ❌.
    ideal_CoT = '<<4-2=2>>
            <<2/.5=4>>
            <<12/4=3>>
            <<100*3=300>>'.
    Answer = '300' .


With persistent steering off, after 1 epoch:
    Full llm output: `['<<', '1', '0', '0', '/', '1', '2', '=', '8', '.', '3', '3', '>>\n', '###', ' ', '8', '.', '3', '3', '\n', '<|im_end|>', '\n', '<|im_end|>', '\n', '<|im_end|>', '\n', '<|im_end|>', '\n', '<|im_end|>', '\n', '<|im_end|>', '\n', '<|im_end|>', '\n', '<|im_end|>', '\n', '<|im_end|>', '\n', '<|im_end|>', '\n', '<|im_end|>', '\n', '<|im_end|>', '\n', '<|im_end|>', '\n', '<|im_end|>', '\n', '<|im_end|>', '\n', '<|im_end|>', '\n', '<|im_end|>', '\n', '<|im_end|>', '\n', '###', ' ', '8', '.', '3', '3', '\n', '<|im_end|>']`. 
    Extracted llm Output: `8.33` (=? 300) ❌.
    ideal_CoT = '<<4-2=2>>
            <<2/.5=4>>
            <<12/4=3>>
            <<100*3=300>>'.
    Answer = '300' .


    Test accuracy: 0.25. eval_8: 100%|███████████████████████████████████████████████████████████████████████████████████| 32/32 [01:55<00:00,  3.62s/it]
    2025-10-19 07:19:29.188 | INFO     | coconut.eval:evaluate:112 - Correct=124, CoT_correct=6, Total=500. eval_8                                       
    2025-10-19 07:19:29.189 | INFO     | coconut.eval:evaluate:113 - Accuracy on val:  124 / 500 =  24.8000%                                             
    2025-10-19 07:19:29.190 | INFO     | coconut.eval:evaluate:114 - CoT match on val: 6 / 500 =  1.2000%                                                
    process_dataset: cot_latent_1 (num_proc=12): 100%|█████████████████████████████████████████████████████████| 500/500 [00:01<00:00, 473.72 examples/s]
    2025-10-19 07:19:43.406 | INFO     | coconut.eval:get_answer_preference:390 - ratio nll_ans/nll_corrupted_ans = 0.9249       

# 2025-10-19 10:14:50

Trying longer run with lower lr

# Results: trm-qwen3-0.6b_20251019-070624
{'project': 'coconut', 'save_path': 'outputs/', 'name': 'trm-qwen3-0.6b', 'model_id': 'outputs/qwen3-0.6b_20250514-194730/checkpoint_2', 'only_eval': False, 'load_model_path': '', 'resume_epochs': 8, 'replacement_method': 'supressed[0.75:]', 'use_position_ids': True, 'bf16': True, 'bf16_weight': False, 'opt_8b': False, 'cot_epochs': 0, 'epochs_per_stage': 8, 'max_latent_stage': 3, 'num_epochs': 16, 'batch_size_training': 16, 'gradient_accumulation_steps': 8, 'lr': 0.0001, 'weight_decay': 0.1, 'grad_clip': 10.0, 'scheduler': 'linear', 'debug': False, 'seed': 42, 'reset_optimizer': False, 'loss_seq_vcr': False, 'n_detached_recursions': 2, 'use_trm': True, 'load_in_4bit': True, 'load_in_8bit': False, 'trm_n_sup': 4, 'trm_num_layers': 2, 'trm_num_heads': 8, 'trm_expansion': 2.67, 'trm_persistent_steering': False, 'max_size': 20000, 'c_thought': 1, 'pad_latent_to_max': True, 'uniform_prob': 0.0, 'train_path': 'data/gsm_train.json', 'val_path': 'data/gsm_valid.json', 'system_prompt': '', 'latent_token_id': None, 'eval_first_epoch': False, 'n_gradient_recursions': 2}
|      | eval/acc | eval/cot_em | eval/ratios | epoch | stage | train/minutes | train/loss | eval/loss |
| ---: | -------: | ----------: | ----------: | ----: | ----: | ------------: | ---------: | --------: |
|    0 |    0.248 |       0.012 |      0.9249 |     8 |     1 |       13.2342 |   0.471396 |    0.4976 |
|    1 |    0.194 |       0.012 |      0.9226 |     9 |     1 |       13.2039 |   0.331766 |     0.481 |
|    2 |    0.242 |        0.01 |       0.924 |    10 |     1 |         13.18 |   0.314844 |    0.4776 |
|    3 |    0.228 |       0.014 |      0.9251 |    11 |     1 |       13.0798 |   0.413031 |    0.4737 |
|    4 |    0.256 |       0.012 |       0.924 |    12 |     1 |       13.0448 |   0.287812 |    0.4749 |
|    5 |    0.212 |        0.01 |      0.9254 |    13 |     1 |       13.0954 |   0.295502 |    0.4734 |
|    6 |    0.214 |       0.012 |      0.9234 |    14 |     1 |       12.9535 |   0.295184 |    0.4739 |
|    7 |    0.206 |        0.01 |      0.9283 |    15 |     1 |       12.9635 |    0.36557 |     0.473 |

# 2025-10-19 10:59:15 original papert configs

https://github.com/SamsungSAILMontreal/TinyRecursiveModels/tree/e7b68717f0a6c4cbb4ce6fbef787b14f42083bd9/config/arch

The other configs are: 
- trm.yaml (with mlp false) TRM-MLP (87.4% test accuracy with 5M params), 
- hrm.yaml for the original HRM (55% acc), 
- transformers_baseline.yaml for baseline (0% acc), 
- trm_hier6.yaml for a multi-scale z variant (lower acc), 
- trm_singlez.yaml for single z (71.9% acc).


And here is the actuall TRM forward https://github.com/SamsungSAILMontreal/TinyRecursiveModels/blob/e7b68717f0a6c4cbb4ce6fbef787b14f42083bd9/models/recursive_reasoning/trm.py#L196


# 2025-10-19 15:50:44

Oh good I was using the wrong psudocode, I keep mixing up TRM and HRM. OK
updated readme, updated code
- [ ] no Q head
- [ ] now use cirriculum instead of deep supervision
- [ ] init transcoder from SVD basis, V from `We` as a prior for embedding space

runnning, trm_persistent_steering=False, initial loss 1






2025-10-19 16:10:42.822 | INFO     | coconut.eval:evaluate:112 - Correct=185, CoT_correct=8, Total=500. eval_8                                                                         
2025-10-19 16:10:42.828 | INFO     | coconut.eval:evaluate:113 - Accuracy on val:  185 / 500 =  37.0000%                                                                               
2025-10-19 16:10:42.834 | INFO     | coconut.eval:evaluate:114 - CoT match on val: 8 / 500 =  1.6000%    
loss=0.43


# 2025-10-19 18:39:02


# Results: trm-qwen3-0.6b_20251019-154941
{'project': 'coconut', 'save_path': 'outputs/', 'name': 'trm-qwen3-0.6b', 'model_id': 'outputs/qwen3-0.6b_20250514-194730/checkpoint_2', 'only_eval': False, 'load_model_path': '', 'resume_epochs': 8, 'replacement_method': 'supressed[0.75:]', 'use_position_ids': True, 'bf16': True, 'bf16_weight': False, 'opt_8b': False, 'cot_epochs': 0, 'epochs_per_stage': 8, 'max_latent_stage': 3, 'num_epochs': 16, 'batch_size_training': 16, 'gradient_accumulation_steps': 8, 'lr': 5e-05, 'weight_decay': 0.1, 'grad_clip': 10.0, 'scheduler': 'linear', 'debug': False, 'seed': 42, 'reset_optimizer': False, 'loss_seq_vcr': False, 'n_detached_recursions': 2, 'use_trm': True, 'load_in_4bit': True, 'load_in_8bit': False, 'max_size': 20000, 'c_thought': 1, 'pad_latent_to_max': True, 'uniform_prob': 0.0, 'train_path': 'data/gsm_train.json', 'val_path': 'data/gsm_valid.json', 'system_prompt': '', 'latent_token_id': None, 'eval_first_epoch': False, 'trm_n_sup': 16, 'trm_h_cycles': 3, 'trm_l_cycles': 6, 'trm_l_layers': 2, 'trm_hidden_size': None, 'trm_num_heads': 8, 'trm_expansion': 2.67, 'trm_transcoder_layers': 1, 'trm_persistent_steering': False}
|      | eval/acc | eval/cot_em | eval/ratios | epoch | stage | train/minutes | train/loss | eval/loss |
| ---: | -------: | ----------: | ----------: | ----: | ----: | ------------: | ---------: | --------: |
|    0 |     0.37 |       0.016 |      0.9266 |     8 |     1 |       21.4294 |   0.421252 |     0.549 |
|    1 |    0.328 |       0.014 |      0.9298 |     9 |     1 |       20.8754 |   0.459505 |    0.5293 |
|    2 |    0.312 |       0.014 |      0.9311 |    10 |     1 |       20.8222 |   0.349655 |    0.5216 |
|    3 |    0.306 |       0.012 |      0.9315 |    11 |     1 |       21.0609 |   0.366589 |     0.518 |
|    4 |    0.296 |       0.012 |      0.9316 |    12 |     1 |       21.3691 |   0.426163 |    0.5139 |
|    5 |    0.268 |       0.012 |      0.9317 |    13 |     1 |       21.0347 |   0.403238 |    0.5109 |
|    6 |     0.27 |       0.012 |      0.9315 |    14 |     1 |       20.9655 |    0.45056 |    0.5067 |
|    7 |    0.268 |       0.014 |      0.9309 |    15 |     1 |       21.0347 |   0.387177 |    0.5044 |
wandb: 


# 2025-10-20 06:08:06

# Results: trm-qwen3-0.6b_20251019-204459
{'project': 'coconut', 'save_path': 'outputs/', 'name': 'trm-qwen3-0.6b', 'model_id': 'outputs/qwen3-0.6b_20250514-194730/checkpoint_2', 'only_eval': False, 'load_model_path': '', 'resume_epochs': 8, 'replacement_method': 'supressed[0.75:]', 'use_position_ids': True, 'bf16': True, 'bf16_weight': False, 'opt_8b': False, 'cot_epochs': 0, 'epochs_per_stage': 8, 'max_latent_stage': 3, 'num_epochs': 30, 'batch_size_training': 16, 'gradient_accumulation_steps': 8, 'lr': 5e-05, 'weight_decay': 0.1, 'grad_clip': 10.0, 'scheduler': 'linear', 'debug': False, 'seed': 42, 'reset_optimizer': False, 'loss_seq_vcr': False, 'n_detached_recursions': 2, 'use_trm': True, 'load_in_4bit': True, 'load_in_8bit': False, 'max_size': 20000, 'c_thought': 1, 'pad_latent_to_max': True, 'uniform_prob': 0.0, 'train_path': 'data/gsm_train.json', 'val_path': 'data/gsm_valid.json', 'system_prompt': '', 'latent_token_id': None, 'eval_first_epoch': False, 'trm_n_sup': 16, 'trm_h_cycles': 3, 'trm_l_cycles': 6, 'trm_l_layers': 2, 'trm_hidden_size': None, 'trm_num_heads': 8, 'trm_expansion': 2.67, 'trm_transcoder_layers': 1, 'trm_persistent_steering': True}
|      | eval/acc | eval/cot_em | eval/ratios | epoch | stage | train/minutes | train/loss | eval/loss |
| ---: | -------: | ----------: | ----------: | ----: | ----: | ------------: | ---------: | --------: |
|    0 |    0.434 |       0.028 |      0.9543 |     8 |     1 |       13.6735 |   0.693636 |    0.8054 |
|    1 |    0.462 |        0.03 |      0.9468 |     9 |     1 |       13.8391 |   0.688966 |    0.7145 |
|    2 |     0.45 |        0.03 |      0.9433 |    10 |     1 |       13.8354 |   0.547561 |    0.6614 |
|    3 |    0.442 |       0.018 |      0.9422 |    11 |     1 |       13.9457 |   0.557111 |    0.6342 |
|    4 |     0.44 |       0.018 |       0.943 |    12 |     1 |        13.687 |   0.572101 |    0.6175 |
|    5 |    0.426 |       0.016 |      0.9444 |    13 |     1 |       13.7035 |    0.60038 |    0.6071 |
|    6 |    0.432 |       0.018 |      0.9439 |    14 |     1 |        13.702 |   0.536857 |    0.6008 |
|    7 |     0.43 |       0.016 |      0.9434 |    15 |     1 |       13.6208 |   0.575433 |    0.5953 |
|    8 |     0.38 |       0.024 |      0.6976 |    16 |     2 |       20.1121 |   0.635938 |    0.7869 |
|    9 |    0.348 |        0.02 |      0.6987 |    17 |     2 |       20.1906 |   0.838113 |    0.7786 |
|   10 |     0.33 |       0.022 |      0.7001 |    18 |     2 |       20.1416 |   0.700116 |    0.7749 |
|   11 |    0.324 |       0.022 |      0.7014 |    19 |     2 |       19.9205 |   0.733533 |    0.7708 |
|   12 |    0.306 |       0.024 |       0.702 |    20 |     2 |       20.2084 |    0.68219 |    0.7683 |
|   13 |     0.28 |       0.022 |      0.7041 |    21 |     2 |       20.1077 |   0.633149 |    0.7647 |
|   14 |     0.27 |       0.026 |      0.7052 |    22 |     2 |       19.8475 |   0.630443 |    0.7625 |
|   15 |     0.25 |        0.02 |      0.7071 |    23 |     2 |       20.0469 |   0.663379 |      0.76 |
|   16 |    0.074 |       0.018 |      0.5485 |    24 |     3 |       16.7198 |   0.774709 |    0.8572 |
|   17 |    0.046 |       0.014 |      0.5498 |    25 |     3 |       16.6994 |   0.865078 |    0.8458 |
|   18 |    0.038 |       0.014 |      0.5532 |    26 |     3 |       16.7742 |   0.838326 |    0.8389 |
|   19 |    0.026 |       0.014 |      0.5543 |    27 |     3 |       16.6039 |   0.789819 |    0.8333 |
|   20 |    0.026 |       0.014 |      0.5558 |    28 |     3 |       16.7808 |   0.769315 |    0.8266 |
|   21 |    0.022 |       0.014 |      0.5584 |    29 |     3 |       16.6993 |   0.669265 |    0.8245 | l |


6 hour run

hmm ok the results are in,
- loss doesn't really go down, 
- but acc does, 
- but ratios went down which is good!!! confusing. that's the perplexity of the good answer vs the bad... I guess the ratio got better but the absolute answer got less likely but not as fast as the corrupted answer

I guess I should add intermentate things too? Maybe load the checkpoints to prototype?

q:

- does each resursion make the loss go down... this is hard to measure I think.
- does the transcoder project to embedding space or blow up or down
- does the absoute perplexity go down, need to log this?


oh but note the stages, the ratios took a jump down with an extra stage. and acc a jump down. maybe I need longer to do it

well loss did trend down, but then bump up again with stages. so that supports the hypothesis that it was learnign with stages

I could also consider adding in the ema deep supervision proxy along with the existing cirriculu

could be the lr is too low too



exp
- without svd init
- without persistent steering
  - monitor diff_content_ratio it shoudl get bigger, and not go to zero


# 2025-10-21 03:58:52

I'm going to create a tiny todo list (marking the table generation as in-progress), then output the requested Markdown table here. After that I'll mark the todo as completed. 

Created 1 todo

## Experiment summary (markdown table)

| Eval                     | Accuracy (val)   | CoT match     | Eval perplexity | Eval loss (approx) | Ratio nll_ans/nll_corrupted_ans | Stage |
| ----                     | -------------:   | --------:     | --------------: | -----------------: | ------------------------------: | ----: |
| eval_8                   | 20.40% (102/500) | 2.00% (10/500)|          1.9209 |             0.6528 |                          0.9402 |     1 |
| eval_9                   | 23.80% (119/500) | 1.20% (6/500) |          1.8317 |             0.6053 |                          0.9431 |     1 |
| eval_10                  | 22.20% (111/500) | 1.40% (7/500) |          1.7980 |             0.5867 |                          0.9409 |     1 |
| eval_11                  | 22.80% (114/500) | 1.20% (6/500) |          1.7798 |             0.5765 |                          0.9376 |     1 |
| eval_12                  | 23.80% (119/500) | 1.40% (7/500) |          1.7686 |             0.5702 |                          0.9380 |     1 |
| eval_13                  | 21.80% (109/500) | 1.40% (7/500) |          1.7595 |             0.5650 |                          0.9364 |     1 |
| eval_14                  | 23.00% (115/500) | 1.40% (7/500) |          1.7537 |             0.5617 |                          0.9369 |     1 |
| eval_15                  | 23.00% (115/500) | 1.40% (7/500) |          1.7484 |             0.5587 |                          0.9359 |     1 |
| eval_16                  |  16.80% (84/500) | 1.00% (5/500) |          2.0950 |             0.7396 |                          0.7027 |     2 |
| eval_17                  |  13.00% (65/500) | 1.20% (6/500) |          2.0813 |             0.7330 |                          0.7017 |     2 |
| eval_18                  |  12.00% (60/500) | 1.40% (7/500) |          2.0664 |             0.7258 |                          0.7014 |     2 |
| eval_19                  |  10.80% (54/500) | 1.60% (8/500) |          2.0542 |             0.7199 |                          0.7023 |     2 |
| eval_20                  |   9.00% (45/500) | 1.40% (7/500) |          2.0367 |             0.7113 |                          0.6999 |     2 |
| eval_21                  |   7.40% (37/500) | 1.40% (7/500) |          2.0196 |             0.7029 |                          0.7039 |     2 |
| eval_22                  |   7.00% (35/500) | 1.20% (6/500) |          2.0072 |             0.6967 |                          0.7032 |     2 |
| eval_23                  |   7.60% (38/500) | 1.20% (6/500) |          1.9953 |             0.6908 |                          0.7035 |   2→3 |
| eval_24                  |   2.80% (14/500) | 1.20% (6/500) |          2.1882 |             0.7831 |                          0.5702 |     3 |
| eval_25                  |   2.60% (13/500) | 1.20% (6/500) |          2.1730 |             0.7761 |                          0.5798 |     3 |
| eval_26                  |   2.20% (11/500) | 1.20% (6/500) |          2.1651 |             0.7725 |                          0.5876 |     3 |
| eval_27                  |   2.60% (13/500) | 1.20% (6/500) |          2.1503 |             0.7656 |                          0.5885 |     3 |
| eval_28                  |   2.40% (12/500) | 1.20% (6/500) |          2.1442 |             0.7628 |                          0.5906 |     3 |
| eval_29                  |   2.80% (14/500) | 1.20% (6/500) |          2.1353 |             0.7586 |                          0.5936 |     3 |
| eval_30                  |   3.00% (15/500) | 1.40% (7/500) |          2.1342 |             0.7581 |                          0.5970 |     3 |




# 2025-10-21 06:01:10

Why does the loss go down but the accuracy gets worse? Why the disconnect. It happens within stages. I've looked for differences in train and eval/generate but they seem fine now.

    Here's how this would work:

    The Task: The TRM's only learning signal comes from the final cross-entropy loss after its generated embedding difference (diff_to_hs) is added and the frozen LLM generates text.
    The Exploit: The TRM discoversthat it can minimize loss by outputting a diff_to_hs that nudges the final embedding towards a "safe," high-probability region of the LLM's vocabulary space. This could be a vector that corresponds to generating common numbers, short answers, or sequences that are syntactically plausible but numerically wrong. The frozen LLM sees an embedding that "looks good" and confidently outputs a low-loss (but incorrect) token stream.
    Accuracy vs. Loss Divergence:
    Loss goes down because the model gets better and better at producing these "safe," plausible-looking embeddings. The LLM's confidence in the generated tokens increases, reducing the negative log-likelihood.
    Accuracy goes down because this exploit is entirely divorced from the actual mathematical reasoning required by GSM8k. The model isn't solving the problem; it's just learning to produce outputs that feel right to the LLM. persistent_steering would accelerate this process, as the TRM continually refines its state towards this "low-loss" basin of attraction over multiple steps.

    Revised Diagnosis: Latent Space Collapse

    The core problem is that the TRM is not learning to solve the math problems. Instead, under the pressure of a simple cross-entropy loss, it has learned to generate a latent vector (diff_to_hs) that nudges the final embedding into a "safe" region that the frozen LLM decodes with high confidence, regardless of factual accuracy. This explains the paradoxical metrics:

Another idea

    __What I Meant__: Yes, both apply the *last* diff— in train, to the final inputs_embeds before last model.forward (coconut.py line 283); in generate, to each new token's embed (lines 373, 394). The difference is context: Train applies it to the remaining input sequence (including ground-truth answer tokens), optimizing loss with known continuations. Generate applies it to *generated* tokens, so if the last diff accumulates noise (from multiple recursions), it could amplify errors in auto-regressive generation (e.g., steering new tokens off-course, collapsing acc).

Possible I guess.


    Consolidated Action Plan
    Validate Generated Outputs (Immediate)

    Load a checkpoint from early training (before divergence) and a late checkpoint
    Generate responses for the same set of problems
    Compare:
    Coherence of reasoning steps
    Numerical accuracy
    Output diversity
    Look for patterns like repeated numbers, safe answers, or degraded reasoning quality
    Implement Regularization (Next Step)

    Add your proposed margin loss:
    # Calculate logprobs for shifted input
    shift_logprobs = logprobs[..., :-1, :]  # Remove last token
    shift_targets = input_ids[..., 1:]       # Shifted targets

    # Calculate NLL for shifted sequence
    nll_shifted = F.cross_entropy(shift_logprobs.view(-1, shift_logprobs.size(-1)), 
                                shift_targets.view(-1), 
                                reduction='none')

    # Margin loss
    margin_loss = torch.mean(F.relu(10 - nll_shifted)**4)
    total_loss = ce_loss + 0.1 * margin_loss  # Weighted addition
    This penalizes the model when it becomes too confident in incorrect predictions

and

    If the hypothesis is correct:

    Stronger transcoder: Should allow the TRM to make more meaningful modifications to the embedding space, improving the coupling between latent reasoning and output quality
    Margin loss: Should prevent the model from becoming overconfident in incorrect outputs, forcing it to maintain plausibility of the full reasoning chain


next experiment:
- [x] cosine scheduler
- [x] stronger transcoder
- [x] margin loss, on whole input nll

```
uv run scripts/run.py TRM 
