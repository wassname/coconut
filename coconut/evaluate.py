import torch
from tqdm.auto import tqdm
from loguru import logger



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
    with torch.no_grad():
        model.eval()
        for idx, batch in enumerate(dataloader):
            if quick and idx > 3:
                break
            test_idx = batch["idx"][0]

            batch = {
                k: v.to(device)
                for k, v in batch.items()
                if v != None and k not in ["idx", "position_ids"]
            }
            assert len(batch["input_ids"]) == 1
            answer = answers_val[test_idx.item()]
            answer_cot = cot_val[test_idx.item()]
            question = question_val[test_idx.item()]

            total += 1

            with torch.autocast(device_type=device, dtype=dtype):
                outputs = model.generate(
                    **batch,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.eos_token_id,
                )

            def indent(s):
                return s.replace("\n", "\n\t")

            llm_text_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
            llm_answer_output = llm_text_output.split("#")[-1].replace(",", "").strip()
            def crop(s, max=30):
                if len(s) > max:
                    return s[:max] + "..."
                return s
            llm_cot_output = (
                ("\n".join(llm_text_output.split("\n")[1:])).split("#")[0].strip()
            )

            if idx < 3:
                correct = '✅' if llm_answer_output==answer else '❌'
                logger.info(
                    f"""Q #{test_idx}: Answer = '{answer}' ideal_CoT = '{indent(answer_cot)},'.
Question: `{indent(question)}`.
Extracted llm Output: `{crop(llm_answer_output)}` (=? {answer}) {correct}.
Full llm output: `{indent(tokenizer.decode(outputs[0]))}`. 
""")
                

            cor += llm_answer_output == answer
            cor_cot += llm_cot_output == answer_cot

            pbar.update(1)
            pbar.set_description(f"Test accuracy: {round(float(cor / total), 2)}. {name}")

        pbar.close()
        logger.info(f"Correct={cor}, CoT_correct={cor_cot}, Total={total}. {name}")
        logger.info(f"Accuracy on validation set:  {cor} / {total} = {cor / total}")
        logger.info(
            f"CoT match on validation set: {cor_cot} / {total} = {cor_cot / total}"
        )

    return {"eval/acc": cor / total, "eval/cot_em": cor_cot / total}
