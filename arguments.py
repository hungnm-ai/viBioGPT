from typing import *
from dataclasses import dataclass, field
from transformers import TrainingArguments


@dataclass
class TokenizerArgs:
    _model_name_or_path: Optional[str] = field(default="Viet-Mistral/Vistral-7B-Chat",
                                               metadata={"help": "Tokenizer model name on HuggingFace"})
    padding_side: Optional[str] = field(default="left",
                                        metadata={"help": "Setting padding side is left or right"})


@dataclass
class ModelArgs:
    model_name_or_path: Optional[str] = field(
        # default="bkai-foundation-models/vietnamese-llama2-7b-120GB",
        default="Viet-Mistral/Vistral-7B-Chat",
        metadata={"help": "Model name or path to pretrained model"})
    lora: Optional[bool] = field(default=True,
                                 metadata={"help": "Use lora to train"})
    qlora: Optional[bool] = field(default=True,
                                  metadata={"help": "Use qlora to train"})
    flash_attention: Optional[bool] = field(default=True,
                                            metadata={"help": "Use flash_attention to train"})
    model_max_length: Optional[int] = field(default=1024,
                                            metadata={
                                                "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."})


@dataclass
class DataArgs:
    train_path: str = field(default="",
                            metadata={"help": "Path to the training data. Use comma for multi input"})
    valid_path: str = field(default=None,
                            metadata={"help": "Path to the evaluation data. Use comma for multi input"})


@dataclass
class TrainingArguments(TrainingArguments):
    per_device_train_batch_size: int = field(default=8,
                                             metadata={
                                                 "help": "The batch size per GPU/XPU/TPU/MPS/NPU core/CPU for training"})
    per_device_eval_batch_size: int = field(default=8,
                                            metadata={
                                                "help": "The batch size per GPU/XPU/TPU/MPS/NPU core/CPU for evaluation"})
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    packing: bool = field(default=False,
                          metadata={"help": "Whether use packing or not"})
