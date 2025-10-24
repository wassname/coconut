import torch

def gen_sample(model, tokenizer, verbose=True):
    # try differen't lengthso f latent
    for l in range(0, 10, 2):
        latent_tokens = '<|start-latent|>' + '<|latent|>' * l + '<|end-latent|>'
        s=[
        {'role':'user', 'content':'What is two plus two but wrong and french?'+latent_tokens},]
        if verbose:
            print(f'--- Generating with {l} latent tokens ---')
        yield gen(s, model, tokenizer, tokenizer_kwargs=dict(add_generation_prompt=True, verbose=verbose))

def gen(s, model, tokenizer, tokenizer_kwargs={}, generate_kwargs={}, verbose=True):
    p = next(iter(model.parameters()))
    dtype = p.dtype
    device = p.device
    if isinstance(s, str):
        inputs = tokenizer.apply_chat_template(
            [{'role': 'user', 'content': s}],    return_tensors='pt',
            truncation=True, padding=True, max_length=128, return_dict=True, **tokenizer_kwargs
        ).to(device)
    elif isinstance(s, list):
        inputs = tokenizer.apply_chat_template(
            s,    return_tensors='pt',
            truncation=True, padding=True, max_length=128, return_dict=True, **tokenizer_kwargs
        ).to(device)
    else:
        raise ValueError('s should be str or list')

    with torch.autocast(device_type='cuda', dtype=dtype):
        inputs = {k: v.to(device=device) for k, v in inputs.items()}
        out = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            # input_embedings=inputs["input_embeddings"],
            max_new_tokens=64,
            min_new_tokens=16,
            # do_sample=True,
            # top_p=0.9,
            # temperature=0.7,
            do_sample=False,
            **generate_kwargs
        )

    s = tokenizer.batch_decode(out, skip_special_tokens=False)[0]
    if verbose:
        print('---input---')
        print(s)
        print('---output---')
    return s
