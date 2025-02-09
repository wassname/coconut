import torch
from tqdm.auto import tqdm
from loguru import logger
import re

def indent(s):
    return s.replace("\n", "\n\t")

def crop(s, maxl=30):
    tokens = ['<|endoftext|>', '<|im_end|>']
    for token in tokens:
        while s.startswith(token):
            s = s[len(token):]
    if len(s) > maxl:
        return "..." + s[-maxl:]
    return s

def extract_first_number2(text):
    # updated regex to capture the first number after '###'
    match = re.search(r'###\s*(\d+\.?\d*)', text)
    if match:
        return match.group(1)
    return ''

@torch.no_grad()
def evaluate(dataloader, model, tokenizer, ds, max_new_tokens=64, device='cuda', name="", dtype=torch.float32, quick=False):


    # get original answer
    question_val = ds["question"]
    answers_val = [
        d.replace(",", "").strip() for d in ds["answer"]
    ]
    cot_val = ["\n".join(d) for d in ds["steps"]]

    # val generation accuracy
    total_length = len(dataloader)

    pbar = tqdm(
        colour="green", desc=f"Test Accuracy {name}", total=total_length, dynamic_ncols=True
    )
    logger.info(f"Starting evaluation {name}")
    cor, cor_cot, total = 0, 0, 0
    batch_size = dataloader.batch_size
    with torch.no_grad():
        model.eval()
        for batch_n, batch in enumerate(dataloader):
            if quick and batch_n*batch_size > 32:
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
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.eos_token_id,
                )

            llm_text_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=False)

            for i in range(len(llm_text_outputs)):
                test_idx = idx[i].item()
                llm_text_output = llm_text_outputs[i]

                # llm_answer_output = llm_text_output.split("#")[-1].replace(",", "").strip()
                llm_answer_output = extract_first_number2(llm_text_output)
                llm_cot_output = (
                    ("\n".join(llm_text_output.split("\n")[1:])).split("#")[0].strip()
                )

                total += 1
                answer = answers_val[test_idx]
                answer_cot = cot_val[test_idx]
                question = question_val[test_idx]
                cor += llm_answer_output == answer
                cor_cot += llm_cot_output == answer_cot

                if (batch_n*batch_size+i)<3:
                    correct = '✅' if llm_answer_output==answer else '❌'
                    logger.info(
                        f"""Q #{test_idx}: Answer = '{answer}' ideal_CoT = '{indent(answer_cot)},'.
    Question: `{indent(question)}`.
    Extracted llm Output: `{crop(llm_answer_output)}` (=? {answer}) {correct}.
    Full llm output: `{indent(crop(llm_text_output, 3000))}`. 
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
