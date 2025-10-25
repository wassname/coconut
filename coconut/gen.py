import torch

def gen_sample(model, tokenizer, verbose=True, **kwargs):
    # try different lengths of latent
    for l in [0, 1, 2]:
        latent_tokens = '<|start-latent|>' + '<|latent|>' * l + '<|end-latent|>'
        s=[
        {'role':'user', 'content':'What is two plus two but wrong and french?'+latent_tokens},]
        if verbose:
            print(f'--- Generating with {l} latent tokens ---')
        yield gen(s, model, tokenizer, tokenizer_kwargs=dict(add_generation_prompt=True), verbose=verbose, **kwargs)

def gen(s, model, tokenizer, min_new_tokens=4, max_new_tokens=16, do_sample=False, tokenizer_kwargs={}, generate_kwargs={}, verbose=True):
    p = next(iter(model.parameters()))
    dtype = p.dtype
    device = p.device
    if isinstance(s, str):
        inputs = tokenizer.apply_chat_template(
            [{'role': 'user', 'content': s}],    return_tensors='pt',
            return_dict=True, **tokenizer_kwargs
        ).to(device)
    elif isinstance(s, list):
        last_role = s[-1].get('role')
        if last_role == 'assistant':
            tokenizer_kwargs['add_generation_prompt'] = False
            tokenizer_kwargs['continue_final_message'] = True
        elif last_role == 'user':
            tokenizer_kwargs['add_generation_prompt'] = True
            tokenizer_kwargs['continue_final_message'] = False
        inputs = tokenizer.apply_chat_template(
            s,    return_tensors='pt',
            return_dict=True, **tokenizer_kwargs
        ).to(device)
    else:
        raise ValueError('s should be str or list')

    with torch.autocast(device_type='cuda', dtype=dtype):
        inputs = {k: v.to(device=device) for k, v in inputs.items()}
        out = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            do_sample=do_sample,
            **generate_kwargs
        )

    # TODO seperate out input vs output
    n = inputs["input_ids"].shape[1]
    out = out[:, n:]  # only return generated tokens

    s_input = tokenizer.batch_decode(inputs["input_ids"], skip_special_tokens=False)[0]
    s_output = tokenizer.batch_decode(out, skip_special_tokens=False)[0]
    if verbose:
        print('---input---')
        print(s_input)
        print('---output---')
        print(s_output)
    return s_output
