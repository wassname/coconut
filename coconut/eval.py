import torch
from tqdm.auto import tqdm
from loguru import logger
import re
from torch.nn import CrossEntropyLoss
from tqdm.auto import tqdm
import numpy as np

def indent(s):
    return s.replace("\n", "\n\t")

def crop(s, maxl=30):
    s = (s
         .replace('<|endoftext|>', '')
         .replace('<|im_end|>', '')
    )
    if len(s) > maxl:
        return s[:maxl] + "..."
    return s

@torch.no_grad()
def evaluate(dataloader, model, tokenizer, ds, max_new_tokens=64, device='cuda', name="", dtype=torch.float32, quick=False, verbose=1):


    # get original answer
    question_val = ds["question"]
    answers_val = [
        d.replace(",", "").strip() for d in ds["answer"]
    ]
    cot_val = ["\n".join(d) for d in ds["steps"]]

    # val generation accuracy
    total_length = len(dataloader)
    if quick:
        total_length = 3 * dataloader.batch_size

    pbar = tqdm(
        colour="green", desc=f"Test Accuracy {name}", total=total_length, dynamic_ncols=True
    )
    logger.info(f"Starting evaluation {name}")
    cor, cor_cot, total = 0, 0, 0
    model.eval()
    for batch_n, batch in enumerate(dataloader):
        if quick and batch_n > 3:
            break
        
        idx = batch["idx"]
        batch = {
            k: v.to(device)
            for k, v in batch.items()
            if v != None and k not in ["idx", "position_ids"]
        }


        with torch.autocast(device_type=device, dtype=dtype):
            outputs = model.generate(
                **batch,
                use_cache=False,
                max_new_tokens=max_new_tokens,
                min_new_tokens=max_new_tokens,
                early_stopping=False,
                pad_token_id=tokenizer.pad_token_id,
                # pad_size='left',
            )

        for i in range(len(outputs)):
            test_idx = idx[i].item()

            # split into question and answer
            q_toks = batch["input_ids"][i]
            a_toks = outputs[i][q_toks.size(0):]
            q_s = tokenizer.decode(q_toks, skip_special_tokens=False)
            ans_tok_list = tokenizer.decode(a_toks, skip_special_tokens=False)
            ans_tok_list = tokenizer.batch_decode(a_toks, skip_special_tokens=False)
            llm_text_output = tokenizer.decode(a_toks, skip_special_tokens=True)

            # TODO use regexp to find numbers group, can be float, after #
            # llm_answer_output = llm_text_output.split("#")[-1]
            llm_answer_output = re.match(
                r".*#+\s*([0-9\.]+).*", llm_text_output
            )
            llm_answer_output = llm_answer_output.group(1) if llm_answer_output else None
            if llm_answer_output is None:
                 llm_answer_output = llm_text_output.split("#")[-1].replace(",", "").replace("<|im_end|>", "").strip()
            llm_cot_output = (
                ("\n".join(llm_text_output.split("\n")[1:])).split("#")[0].strip()
            )

            total += 1
            answer = answers_val[test_idx]
            answer_cot = cot_val[test_idx]
            question = question_val[test_idx]
            cor += llm_answer_output == answer
            cor_cot += llm_cot_output == answer_cot

            if (batch_n < verbose) and (i < 1):
                correct = '✅' if llm_answer_output==answer else '❌'
                logger.debug(
                    f"""Q #{test_idx}: Question: `{indent(question)}`.
Full question: `{indent(crop(q_s, maxl=2900))}`.
Full llm output: `{ans_tok_list}`. 
Extracted llm Output: `{crop(llm_answer_output)}` (=? {answer}) {correct}.
ideal_CoT = '{indent(answer_cot)}'.
Answer = '{answer}' .
""")                


        pbar.update(1)
        pbar.set_description(f"Test accuracy: {round(float(cor / total), 2)}. {name}")

    pbar.close()
    logger.info(f"Correct={cor}, CoT_correct={cor_cot}, Total={total}. {name}")
    logger.info(f"Accuracy on val:  {cor} / {total} = {cor / total: .4%}")
    logger.info(
        f"CoT match on val: {cor_cot} / {total} = {cor_cot / total: .4%}"
    )

    return {"eval/acc": cor / total, "eval/cot_em": cor_cot / total}




@torch.no_grad()
def get_answer_perplexity(
    model,
    tokenizer,
    valid_gen_dataloader,
    device='cuda',
    dtype=torch.float32,
    verbose=False
):
    """
    If I forward through the answer, then get the perplexity over the answer tokens, I can a more sensitive measure
    """

    model.eval()
    with torch.no_grad():

        token_preans = tokenizer.convert_tokens_to_ids("###")
        nlls, counts = 0, 0
        loss_fn = CrossEntropyLoss(reduction="none")

        for batch_n, batch in enumerate(tqdm(valid_gen_dataloader, desc="PPX", colour="green", dynamic_ncols=True)):

            if verbose and batch_n <1:
                i = tokenizer.decode(batch['input_ids'][0], skip_special_tokens=False)
                logger.info(f"Input: {i}")
            
            batch = {
                k: v.to(device)
                for k, v in batch.items()
                if v != None and k not in ["idx", "position_ids"]
            }
            with torch.autocast(device_type=device, dtype=dtype):
                output = model.forward(
                    **batch)
            logprobs = output.logits.log_softmax(-1)

            B = logprobs.shape[0]
            for i in range(B):
                # find tokens after token_preans
                input_ids_i = batch['input_ids'][i].cpu()
                a=(input_ids_i==token_preans).float()
                i_ans = a.argmax()+2
                i_end = (input_ids_i[i_ans:]==tokenizer.eos_token_id).float().argmax() + i_ans
                if i_end == i_ans:
                    i_end = -1
                if i_ans == 0:
                    raise ValueError(
                        f"Answer token {token_preans} not found in input_ids {input_ids_i} make sure you used get_cot_latent_dataset() to generate the dataset"
                    )
                
                ans_mask = batch['attention_mask'][i, i_ans: i_end].cpu()
                ans_tokens = input_ids_i[i_ans:i_end]
                ans_logits = logprobs[i, i_ans:i_end].cpu()

                if verbose and (batch_n < 1) and i < 1:

                    parts =[ 
                        input_ids_i[:i_ans],
                        input_ids_i[i_ans+1:i_end],
                        input_ids_i[i_end:],
                    ]
                    for j, part in enumerate(parts):
                        part_s = tokenizer.batch_decode(part, skip_special_tokens=False)
                        logger.debug(f"part {j}: `{part_s}`")
                    
                    g0 = ans_tokens
                    g0s = tokenizer.decode(g0)
                    g = g0 * ans_mask
                    g = g[g != 0]
                    g1s = tokenizer.decode(g)
                    logger.debug(f"g (unmask): `{g0s}`, ans (masked): `{g1s}`")
                    # TODO make a good QC thing here. I want to see where we sliced the string. where we masked it.

                    # logger.info(f"

                # calc ppx
                # Compute loss on shifted sequences
                shift_logits = ans_logits[:-1].contiguous()
                shift_labels = ans_tokens[ 1:].contiguous()
                shift_masks = ans_mask[ 1:].contiguous()


                # Calculate NLL only for targeted tokens
                loss = loss_fn(shift_logits, shift_labels)
                # print(f"loss: loss.shape: {loss.shape}")
                # print(f"shift_labels:shape: {shift_labels.shape}")
                # print(f"shift_logits: shape: {shift_logits.shape}")
                # print(f"shift_masks: e: {shift_masks.shape} {shift_masks.sum()}")
                masked_loss = (loss * shift_masks).sum()
                token_count = shift_masks.sum()

                # Accumulate results
                nlls += masked_loss.item()
                counts += token_count.item()

    # Return corpus-level perplexity
    ppx =  np.exp(nlls / counts) if counts > 0 else float('inf')
    return {
        "eval/ppx": ppx,
        "eval/ppx_count": counts,
        "eval/ppx_nlls": nlls,
    }


import random

def corrupt_answer(ans_tokens, tokenizer):
    num2tok = dict([(i,tokenizer.convert_tokens_to_ids(str(i))) for i in range(10)])
    numtoks = list(num2tok.values())[1:] # not zero, as then we would get leading zeros
    # corrupt the answer by randomly swapping digits
    ans_tokens = ans_tokens.clone()

    # number swap dict:
    num_perm = dict(zip(numtoks, numtoks[::-1]))
    # print('num_perm', num_perm)
    # print(0, tokenizer.decode(ans_tokens))
    for i in range(len(ans_tokens)):
        t = ans_tokens[i].cpu().item()
        if t in num_perm:
            ans_tokens[i] = num_perm[t]
    # print(1, tokenizer.decode(ans_tokens))
    return ans_tokens




def corrupt_batch_answers(input_ids, tokenizer):
    input_ids = input_ids.clone()
    token_preans = tokenizer.convert_tokens_to_ids("###")
    for i in range(len(input_ids)):
        input_ids_i = input_ids[i]
        a=(input_ids_i==token_preans).float()
        i_ans = a.argmax()-10 # we want to get the last CoT too as it states the answer
        # print(0, input_ids[i][i_ans:i_ans+5])
        input_ids_i[i_ans:] = corrupt_answer(input_ids_i[i_ans:], tokenizer)
        input_ids[i] = input_ids_i
        # print(1, input_ids[i][i_ans:i_ans+5])
    return input_ids



def calc_ans_nll(batch, model, tokenizer, device, dtype, verbose=False):
    """
    Perpexlity is not the best measure so here we prefer the likelihood of the answer over a corrupted answer
    """
    nlls, counts = 0, 0
    loss_fn = CrossEntropyLoss(reduction="none")
    token_preans = tokenizer.convert_tokens_to_ids("###")

    with torch.autocast(device_type=device, dtype=dtype):
        output = model.forward(
            **batch)
    logprobs = output.logits.log_softmax(-1)

    B = logprobs.shape[0]
    for i in range(B):
        # find tokens after token_preans
        input_ids_i = batch['input_ids'][i].cpu()

        a=(input_ids_i==token_preans).float()
        if a.max() == 0:
            logger.warning(f"Answer token {token_preans} not found in input_ids {input_ids_i} make sure you used get_cot_latent_dataset() to generate the dataset")
            continue
        
        # get the last instance of '###'
        idx_ans_start = a.flip(0).argmax()
        idx_ans_start = len(a)-idx_ans_start
        idx_ans_start += 1 # skip [' '] that is after ###

        # find the end of the answer, denoted by eos_token
        idx_ans_end = (input_ids_i[idx_ans_start:]==tokenizer.eos_token_id).float().argmax() + idx_ans_start
        if idx_ans_end == idx_ans_start:
            idx_ans_end = -1
        
        if idx_ans_start == 0:
            raise ValueError(
                f"Answer token {token_preans} not found in input_ids {input_ids_i} make sure you used get_cot_latent_dataset() to generate the dataset"
            )
        
        ans_mask = batch['attention_mask'][i, idx_ans_start: idx_ans_end].cpu()
        ans_tokens = input_ids_i[idx_ans_start:idx_ans_end]
        ans_logits = logprobs[i, idx_ans_start:idx_ans_end].cpu()

        # if verbose and (i < 1):

        #     parts =[ 
        #         input_ids_i[:idx_ans_start],
        #         input_ids_i[idx_ans_start:idx_ans_end],
        #         input_ids_i[idx_ans_end:],
        #     ]
        #     for j, part in enumerate(parts):
        #         part_s = tokenizer.batch_decode(part, skip_special_tokens=False)
        #         logger.debug(f"part {j}: `{part_s}`")


        #     ans_s_premask = tokenizer.batch_decode(ans_tokens, skip_special_tokens=False)
        #     g = ans_tokens * ans_mask
        #     g = g[g != 0]
        #     ans_s = tokenizer.batch_decode(g, skip_special_tokens=False)
        #     logger.debug(f"extracted ans: `{ans_s_premask}` -masked-> `{ans_s}`")

        # calc ppx
        # Compute loss on shifted sequences
        shift_logits = ans_logits[:-1].contiguous()
        shift_labels = ans_tokens[ 1:].contiguous()
        shift_masks = ans_mask[ 1:].contiguous()


        # Calculate NLL only for targeted tokens
        loss = loss_fn(shift_logits, shift_labels)
        masked_loss = (loss * shift_masks).sum()
        token_count = shift_masks.sum()

        # Accumulate results
        nlls += masked_loss.item()
        counts += token_count.item()
    if counts == 0:
        logger.warning("No tokens found for answer")
        return 0    
    ratio = nlls/(counts+.001)
    return ratio

@torch.no_grad()
def get_answer_preference(
    model,
    tokenizer,
    valid_gen_dataloader,
    device='cuda',
    dtype=torch.float32,
    verbose=False
):
    """
    If I forward through the answer, then get the perplexity over the answer tokens, I can a more sensitive measure
    """

    # Perplexity is not that great a measure since it might be measuring formatting rather than answer. Ideally we measure a chosen vs alternative answer

    model.eval()
    with torch.no_grad():

        ratios = []
        nll_chos = []
        nll_refs = []

        for batch_n, batch in enumerate(valid_gen_dataloader):

            if verbose and batch_n <1:
                i = tokenizer.decode(batch['input_ids'][0], skip_special_tokens=False)
                logger.info(f"Input: {i}")
            
            batch = {
                k: v.to(device)
                for k, v in batch.items()
                if v != None and k not in ["idx", "position_ids"]
            }

            batch2 = {k: v.clone() for k, v in batch.items()}
            random.seed(batch_n)
            batch2['input_ids'] = corrupt_batch_answers(batch['input_ids'], tokenizer)

            nll_cho = calc_ans_nll(batch, model, tokenizer, device, dtype, verbose=batch_n < 1)
            nll_ref = calc_ans_nll(batch2, model, tokenizer, device, dtype, verbose=batch_n < 1)
            ratio = nll_cho - nll_ref
            ratios.append(ratio)
            nll_chos.append(nll_cho)
            nll_refs.append(nll_ref)



    # Return corpus-level perplexity
    ratios =  np.exp(ratios).mean()
    logger.info(f"ratio nll_ans/nll_corrupted_ans = {ratios:2.4f}")
    return {
        "eval/ratios": ratios,
        "eval/nll_chos_avg": np.mean(nll_chos),
        "eval/nll_refs_avg": np.mean(nll_refs),
    }
