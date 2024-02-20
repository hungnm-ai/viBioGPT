import torch
import wandb
import random
import bitsandbytes as bnb
from constants import WANDB_KEY
from tokenizer import load_tokenizer
from dataset import BioDataset, DataReader
from sklearn.model_selection import train_test_split
from data_collators import DataCollatorForCompletionLM
from arguments import ModelArgs, TokenizerArgs, TrainingArguments, DataArgs
from peft import get_peft_model, prepare_model_for_kbit_training, LoraConfig
from transformers import (Trainer, HfArgumentParser, BitsAndBytesConfig,
                          PreTrainedTokenizer, AutoModelForCausalLM)

SEED = 100

wandb.login(key=WANDB_KEY)
run = wandb.init(
    project='bioGPT-instruct',
    job_type="training",
    anonymous="allow"
)


def set_seed(seed: int = SEED):
    """Set random to ensure result reproducible"""
    random.seed(seed)
    torch.manual_seed(seed)


def find_all_linear_names(model):
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, bnb.nn.Linear4bit) or isinstance(module, torch.nn.Linear):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"Trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")


def load_model(model_args: ModelArgs, tokenizer: PreTrainedTokenizer):
    device = {"": 0} if torch.cuda.is_available() else "cpu"
    quantization_config = None
    if model_args.qlora:
        compute_dtype = getattr(torch, "bfloat16")
        quantization_config = BitsAndBytesConfig(load_in_4bit=True,
                                                 bnb_4bit_quant_type="nf4",
                                                 bnb_4bit_use_double_quant=True,
                                                 bnb_4bit_compute_dtype=compute_dtype)
    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path,
                                                 device_map=device,
                                                 trust_remote_code=True,
                                                 quantization_config=quantization_config,
                                                 use_flash_attention_2=model_args.flash_attention
                                                 )

    model.resize_token_embeddings(len(tokenizer))

    model.config.pad_token_id = tokenizer.pad_token_id
    model.gradient_checkpointing_enable()

    modules = find_all_linear_names(model)

    if model_args.lora:
        if model_args.qlora:
            print("Using QLora to train...")
            model = prepare_model_for_kbit_training(model)
        else:
            print("Using Lora to train...")

        print("Modules is adopted: {}".format(modules))
        # Adding the adopter to the layer
        peft_config = LoraConfig(
            r=16,  # dimension of the updated matrices
            lora_alpha=64,  # parameter for scaling
            # ['k_proj', 'q_proj', 'v_proj', 'o_proj',
            # "gate_proj", "down_proj", "up_proj"]
            target_modules=modules,
            lora_dropout=0.1,  # dropout probability for layers
            bias="none",
            task_type="CAUSAL_LM",
            modules_to_save=["lm_head", "embed_tokens"],
        )
        model = get_peft_model(model, peft_config)

    """Because KV cache is useless during training(Finetune), It only works for inference.
    For a Generative Language model.
    For a training iteration, all result are computed parallel with casual mask and teacher-forcing,
    which means all the key and value for different input token are computed in one time.
    https://stackoverflow.com/questions/76633335/why-does-hugging-face-falcon-model-use-mode-config-use-cache-false-why-wouldn
    """
    model.config.use_cache = False

    print_trainable_parameters(model)

    return model


def train():
    arg_parser = HfArgumentParser((ModelArgs, DataArgs, TrainingArguments, TokenizerArgs))
    model_args, data_args, training_args, tokenizer_args = arg_parser.parse_args_into_dataclasses()

    train_reader = DataReader(data_args.train_path)
    train_data = train_reader.load_data()
    if data_args.valid_path:
        train_reader = DataReader(data_args.valid_path)
        valid_data = train_reader.load_data()

    else:
        train_data, valid_data = train_test_split(train_data,
                                                  test_size=0.1,
                                                  random_state=SEED,
                                                  shuffle=True)

    tokenizer_args._model_name_or_path = model_args.model_name_or_path
    tokenizer = load_tokenizer(tokenizer_args)

    print("Number of training examples: {}".format(len(train_data)))
    print("Number of valid examples: {}".format(len(valid_data)))
    train_dataset = BioDataset(examples=train_data,
                               tokenizer=tokenizer)

    valid_dataset = BioDataset(examples=valid_data,
                               tokenizer=tokenizer)

    model = load_model(model_args, tokenizer)
    data_collator = DataCollatorForCompletionLM(mlm=False, tokenizer=tokenizer)
    trainer = Trainer(model=model,
                      args=training_args,
                      train_dataset=train_dataset,
                      eval_dataset=valid_dataset,
                      data_collator=data_collator
                      )

    trainer.train()


if __name__ == '__main__':
    train()
