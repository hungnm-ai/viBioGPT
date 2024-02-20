from constants import HF_TOKEN
from arguments import TokenizerArgs
from transformers import AutoTokenizer
from transformers import PreTrainedTokenizer


class SpecialToken:
    bos = "<s>"
    eos = "</s>"
    start_instruct = "[/INST]"
    end_instruct = "[INST]"
    start_sys = "<<SYS>>"
    end_sys = "<</SYS>>"


def load_tokenizer(tokenizer_args: TokenizerArgs) -> PreTrainedTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=tokenizer_args._model_name_or_path,
                                              token=HF_TOKEN)
    tokenizer.padding_side = tokenizer_args.padding_side
    tokenizer.pad_token_id = tokenizer.eos_token_id

    return tokenizer
