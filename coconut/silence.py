import warnings

# can be emitted when generating schemas for dataclasses used by tyro.
warnings.filterwarnings(
    "ignore",
    message="The 'repr' attribute with value False was provided to the `Field()` function",
)
warnings.filterwarnings(
    "ignore",
    message="The 'frozen' attribute with value True was provided to the `Field()` function",
)
