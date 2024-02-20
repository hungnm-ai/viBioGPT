import typer
import torch
from peft import PeftModel
from constants import HF_TOKEN
from tokenizer import SpecialToken
from prompt.qa_prompt import QAPrompt
from transformers import (AutoTokenizer,
                          BitsAndBytesConfig,
                          AutoModelForCausalLM,
                          PreTrainedTokenizer,
                          MistralForCausalLM,
                          StoppingCriteria,
                          StoppingCriteriaList)

app = typer.Typer()

model_name = "Viet-Mistral/Vistral-7B-Chat"
adapter = "hungnm/viBioGPT-7B-instruct-qlora-adapter"


class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = [stop for stop in stops]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        # print(input_ids.device)
        for stop in self.stops:
            stop = stop.to(input_ids.device)
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True

        return False


def load_tokenizer():
    """load and config tokenizer"""

    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              token=HF_TOKEN)
    tokenizer.padding_side = "left"
    tokenizer.pad_token_id = tokenizer.eos_token_id

    return tokenizer


@app.command()
def merge_adapter(save_dir: str = "./models/merge_adapter"):
    """Merge adapter into pretrained model and save model"""
    # Load the pretrained model
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Load and activate the adapter on top of the base model
    model = PeftModel.from_pretrained(model, adapter)

    # Merge the adapter with the base model
    model = model.merge_and_unload()

    tokenizer = load_tokenizer()

    # Save the merged model in a director in the safetensors format
    model.save_pretrained(save_dir,
                          safe_serialization=True)
    tokenizer.save_pretrained(save_dir)


def load_adapter_merged(save_dir: str = typer.Option(default="./models/merge_adapter")):
    """Load adapter merged with pretrained model"""
    compute_dtype = getattr(torch, "bfloat16")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(save_dir,
                                                 quantization_config=bnb_config,
                                                 device_map={"": 0},
                                                 use_flash_attention_2=True
                                                 )

    return model


def load_adapter():
    """
    Load adapter on top of pretrained model (Vistral-7B-Chat)
    Note: Using the same loading hyperparameters used for fine-tuning
    """
    compute_dtype = getattr(torch, "bfloat16")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                 quantization_config=bnb_config,
                                                 device_map={"": 0},
                                                 use_flash_attention_2=True
                                                 )
    model = PeftModel.from_pretrained(model, adapter)

    return model


@app.command()
def text_generate(question: str,
                  model=None,
                  tokenizer=None):
    question = question.strip()
    if model is None:
        model = load_adapter()
    if tokenizer is None:
        tokenizer = load_tokenizer()

    stop_words = [SpecialToken.eos, SpecialToken.end_instruct]
    stop_words_ids = [tokenizer(stop_word, return_tensors='pt',
                                add_special_tokens=False)['input_ids'].squeeze(1)
                      for stop_word in stop_words]
    stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

    prompt = QAPrompt()
    instruction_str = prompt.build_prompt_instruction(question=question, tokenizer=tokenizer)
    token_ids = tokenizer([instruction_str], return_tensors="pt")["input_ids"]
    token_ids = token_ids.to(model.device)
    outputs = model.generate(input_ids=token_ids,
                             max_new_tokens=768,
                             do_sample=True,
                             temperature=0.001,
                             top_p=0.95,
                             top_k=40,
                             repetition_penalty=1.2,
                             stopping_criteria=stopping_criteria
                             )
    all_token_ids = outputs[0].tolist()
    output_token_ids = all_token_ids[token_ids.shape[-1]:]
    output = tokenizer.decode(output_token_ids)

    print(f"User: {question}\n")
    print(f"AI-Doctor: {output}")

    return output


if __name__ == '__main__':
    app()
