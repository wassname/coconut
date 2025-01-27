from transformers import Trainer
from optimi import AdamW
from transformers import TrainingArguments, PreTrainedModel
from typing import Any, Optional, Tuple
from torch import nn
import torch

class TrainerOptimi(Trainer):

	@staticmethod
	def get_optimizer_cls_and_kwargs(
		args: TrainingArguments, model: Optional[PreTrainedModel] = None
	) -> Tuple[Any, Any]:
		optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(args, model)
		allowed_kwargs = {'lr', 'betas', 'weight_decay', 'eps', 'decouple_lr', 'max_lr', 'kahan_sum', 'foreach', 'gradient_release'}
		default_kwargs = {'kahan_sum': True}
		optimizer_kwargs = {k: v for k, v in optimizer_kwargs.items() if k in allowed_kwargs}
		optimizer_kwargs = {**default_kwargs, **optimizer_kwargs}
		return AdamW, optimizer_kwargs
        

def convert_to_bfloat16(module):
	for child in module.children():
		if isinstance(child, (nn.Linear, nn.Conv2d)):
			child.to(torch.bfloat16)
		else:
			convert_to_bfloat16(child)
