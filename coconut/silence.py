import warnings
from pydantic._internal._generate_schema import UnsupportedFieldAttributeWarning

warnings.filterwarnings("ignore", category=UnsupportedFieldAttributeWarning)

# suppress transformers/data/data_collator.py UserWarning about slow tensor creation
warnings.filterwarnings(
    "ignore",
    message=r"Creating a tensor from a list of numpy\.ndarrays is extremely slow.*",
    category=UserWarning,
)


# # can be emitted when generating schemas for dataclasses used by tyro.
# warnings.filterwarnings(
#     "ignore",
#     message="The 'repr' attribute with value False was provided to the `Field()` function",
# )
# warnings.filterwarnings(
#     "ignore",
#     message="The 'frozen' attribute with value True was provided to the `Field()` function",
# )


# /media/wassname/SGIronWolf/projects5/2025/fbai_coconut/.venv/lib/python3.10/site-packages/transformers/data/data_collator.py:740: UserWarning: Creating a tensor from a list of numpy.ndarrays is extremely slow. Please consider converting the list to a single numpy.ndarray with numpy.array() before converting to a tensor. (Triggered internally at /pytorch/torch/csrc/utils/tensor_new.cpp:253.)
#   batch["labels"] = torch.tensor(batch["labels"], dtype=torch.int64)
