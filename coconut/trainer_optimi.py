from transformers import Trainer
from optimi import AdamW
from transformers import TrainingArguments, PreTrainedModel
from typing import Any, Optional, Tuple


class TrainerOptimi(Trainer):

    @staticmethod
    def get_optimizer_cls_and_kwargs(
        args: TrainingArguments, model: Optional[PreTrainedModel] = None
    ) -> Tuple[Any, Any]:
        optimizer_cls, optimizer_kwargs = super().get_optimizer_cls_and_kwargs(args, model)
        return AdamW, optimizer_kwargs
        