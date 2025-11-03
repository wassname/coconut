import torch
from tqdm.auto import tqdm
from loguru import logger
import re
from torch.nn import CrossEntropyLoss
from tqdm.auto import tqdm
import numpy as np
import random

def match_token_indices(tokens: torch.Tensor, tokenizer, regex_pattern: str= r'#+\s*([+-]?\d{1,3}(?:,\d{3})*\.?\d*)'):
    """
    Find start and end indices (0-based, end exclusive) of the minimal token span
    where the decoded text contains a regex match of maximum length.
    Returns (start, end) or (None, None) if no match.
    Assumes tokens is 1D tensor/list of token IDs.
    """
    tokens_list = tokens.tolist() if isinstance(tokens, torch.Tensor) else tokens
    max_match_len = 0
    candidate_end = len(tokens_list)

    regex_pattern = re.compile(regex_pattern)
    
    # Forward pass: find end of longest match (preferring rightmost if ties)
    for i in range(len(tokens_list)):
        curr_str = tokenizer.decode(tokens_list[:i+1])
        match = regex_pattern.search(curr_str)
        if match and len(match.group(0)) > max_match_len:
            max_match_len = len(match.group(0))
            candidate_end = i + 1
    
    if max_match_len == 0:
        return None, None
    
    # Backward pass: find leftmost start for that max length match
    max_match_len = 0
    for j in range(candidate_end):
        curr_str = tokenizer.decode(tokens_list[j:candidate_end])
        match = regex_pattern.search(curr_str)
        if (not match) or len(match.group(0)) < max_match_len:
            candidate_start = j - 1
            break
        elif match:
            candidate_start = j
            max_match_len = len(match.group(0))

    return candidate_start, candidate_end

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


from transformers.generation.stopping_criteria import StoppingCriteria, EosTokenCriteria

class GSM8KStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> torch.BoolTensor:
        # stop if we see two ###
        outs = self.tokenizer.batch_decode(input_ids, skip_special_tokens=False)
        stop = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        for i, out in enumerate(outs):
            if out.count('###') >= 2:
                stop[i] = True
        return stop


@torch.no_grad()
def evaluate(dataloader, model, tokenizer, ds, max_new_tokens=16, device='cuda', name="", dtype=torch.float32, quick=False, verbose=1, best_of_n=1):
    # TODO enable best of 4 like in qwen paper


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
                # use_cache=True,
                max_new_tokens=max_new_tokens,
                min_new_tokens=6,
                # early_stopping=True,
                pad_token_id=tokenizer.pad_token_id,
                num_return_sequences=best_of_n,
                num_beams=best_of_n,
                return_dict_in_generate=True,
                output_scores=True,
                stopping_criteria=[
                    GSM8KStoppingCriteria(tokenizer),
                    EosTokenCriteria(tokenizer.eos_token_id)
                ],
                # padding_side='left',
            )

        # FIXME handle multiple return sequences in COCONUT

        for i in range(len(outputs)):
            test_idx = idx[i].item()

            # split into question and answer
            q_toks = batch["input_ids"][i]
            a_toks = outputs.sequences[i][q_toks.size(0):]
            q_s = tokenizer.decode(q_toks, skip_special_tokens=False)
            ans_tok_list = tokenizer.decode(a_toks, skip_special_tokens=False)
            ans_tok_list = tokenizer.batch_decode(a_toks, skip_special_tokens=False)
            llm_text_output = tokenizer.decode(a_toks, skip_special_tokens=True)

            # Use token-span matching to find precise answer after #
            # see https://github.com/QwenLM/Qwen/blob/b5529b8958ba806c633570e1f64aaa38b6dbe3aa/eval/evaluate_chat_gsm8k.py#L49
            # _PAT_LAST_DIGIT = re.compile(r'#+\s*([+-]?\d{1,3}(?:,\d{3})*\.?\d*)')
            start, end = match_token_indices(a_toks, tokenizer, r'#+\s*([+-]?\d{1,3}(?:,\d{3})*\.?\d*)')
            if start is not None:
                answer_span = a_toks[start:end]
                decoded_span = tokenizer.decode(answer_span, skip_special_tokens=True)
                match = re.search(r'#+\s*(\d+\.?\d*)', decoded_span)
                llm_answer_output = match.group(1).strip() if match else None
                # CoT is everything before the answer span
                cot_tokens = a_toks[:start]
                llm_cot_output = tokenizer.decode(cot_tokens, skip_special_tokens=True).strip()
            else:
                # No match: answer fails, CoT is full output
                llm_answer_output = ''
                llm_cot_output = llm_text_output.strip()

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
        # FIXME should be first? as the mode repeats itself, but after the question as we might have it in system prompt
        idx_ans_start = a.flip(0).argmax()
        idx_ans_start = len(a)-idx_ans_start
        idx_ans_start += 1 # skip [' '] that is after ###

        remaining_tokens = input_ids_i[idx_ans_start:]
        rel_start, rel_end = match_token_indices(remaining_tokens, tokenizer, r'\d+\.?\d*')
        if rel_start is None:
            raise ValueError(f"No number match found in answer tokens after ###: {tokenizer.decode(remaining_tokens)}")
        idx_ans_end = idx_ans_start + rel_end

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
