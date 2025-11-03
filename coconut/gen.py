import torch
from loguru import logger
from coconut.coconut import Coconut, recursion_context
from coconut.adapters import set_adapter
# from coconut.configs import bot_token, latent_token, eot_token

def gen_sample2(model: Coconut, tokenizer, verbose=True, latents=[None, 0, 1, 2], **kwargs):

    latent_token  = model.config.latent_token
    bot_token     = model.config.bot_token
    eot_token     = model.config.eot_token

    adapters = [model.model.active_adapter, None]

    # try different lengths of latent
    for adapter in adapters:
        outs = []
        with set_adapter(model.model, adapter):
            for i, n_latents in enumerate(latents):
                if n_latents is None:
                    latent_tokens = ''
                else:
                    latent_tokens = bot_token + latent_token * n_latents + eot_token
                s=[
                    {'role': 'system', 'content': ''}, # FIXME use config system prompt
                    {'role':'user', 'content':'What is two plus two but wrong and french?'},
                    {'role':'assistant', 'content': latent_tokens}
                ]
                if i==0 and verbose:
                    logger.info(f'--- Generating adapter=({adapter}) {n_latents} latent tokens ---')
                if n_latents is None:
                    with set_adapter(model.model, None):
                        out = gen(s, model, tokenizer, tokenizer_kwargs=dict(add_generation_prompt=True), verbose=False, **kwargs)
                else:
                    out = gen(s, model, tokenizer, tokenizer_kwargs=dict(add_generation_prompt=True), verbose=False, **kwargs)
                outs.append(latent_tokens+out)
        if verbose:
            sout = f"Input: {s[0]['content']}\n"
            sout += '\n'.join([f'--- Generated with adapter={adapter} and {ll} latent tokens ---\n{out}' for ll, out in zip(latents, outs)])
            logger.info(sout)
    return outs

# def gen_sample(model, tokenizer, verbose=True, **kwargs):
#     latent_token  = model.config.latent_token
#     bot_token     = model.config.bot_token
#     eot_token     = model.config.eot_token
#     # try different lengths of latent
#     for l in [0, 1, 2]:
#         latent_tokens = bot_token + latent_token * l + eot_token
#         s=[
#         {'role':'user', 'content':'What is two plus two but wrong and french?'},
#         {'role':'assistant', 'content':latent_tokens}
#         ]
#         if verbose:
#             logger.info(f'--- Generating with {l} latent tokens ---')
#         yield gen(s, model, tokenizer, tokenizer_kwargs=dict(add_generation_prompt=True), verbose=verbose, **kwargs)

def gen(s, model, tokenizer, min_new_tokens=4, max_new_tokens=16, do_sample=False, tokenizer_kwargs={}, generate_kwargs={}, verbose=True):
    p = next(iter(model.parameters()))
    dtype = p.dtype
    device = p.device
    if isinstance(s, str):
        inputs = tokenizer.apply_chat_template(
            [{'role': 'user', 'content': s}],    return_tensors='pt',
            return_dict=True, **tokenizer_kwargs
        ).to(device)
    
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

    with torch.autocast(device_type='cuda', dtype=dtype):
        with recursion_context(model, inputs["input_ids"], {}, tokenizer.convert_tokens_to_ids('...')):
            inputs = {k: v.to(device=device) for k, v in inputs.items()}
            out = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                do_sample=do_sample,
                **generate_kwargs
            )

    n = inputs["input_ids"].shape[1]
    out = out[:, n:]  # only return generated tokens

    s_input = tokenizer.batch_decode(inputs["input_ids"], skip_special_tokens=False)[0]
    s_output = tokenizer.batch_decode(out, skip_special_tokens=False)[0]
    if verbose:
        ss = f'''---input---
{s_input}
---output---
{s_output}'''
        logger.info(ss)
    return s_output
