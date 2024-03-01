import torch
from peft import PeftModel
from fastapi import FastAPI
from api.models import HistoryChat
from tokenizer import load_tokenizer
from arguments import TokenizerArgs
from fastapi.responses import JSONResponse
from constants import SYS_PROMPT, HF_TOKEN
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

app = FastAPI()
origins = ["*"]
app.add_middleware(
    CORSMiddleware, allow_origins=origins,
    allow_credentials=True, allow_methods=['*'], allow_headers=['*'])

model_name = "Viet-Mistral/Vistral-7B-Chat"
adapter = "hungnm/viBioGPT-7B-instruct-qlora-adapter"

compute_dtype = getattr(torch, "bfloat16")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(model_name,
                                             quantization_config=bnb_config,
                                             device_map="auto",
                                             token=HF_TOKEN
                                             )
model = PeftModel.from_pretrained(model, adapter)

tokenizer = load_tokenizer(tokenizer_args=TokenizerArgs())

system_prompt = SYS_PROMPT


@app.get("/ping")
def ping():
    return {"message": "Pong"}


@app.post("/qa")
def qa(question: str, messages: HistoryChat):
    conversation = [
        {
            "role": "system",
            "content": system_prompt}
    ]
    history = [message.dict() for message in messages.messages]
    conversation.extend(history)
    conversation.append({
        "role": "user",
        "content": question
    })
    instruction_str = tokenizer.apply_chat_template(conversation=conversation,
                                                    tokenize=False)
    token_ids = tokenizer([instruction_str], return_tensors="pt")["input_ids"]
    token_ids = token_ids.to(model.device)
    outputs = model.generate(input_ids=token_ids,
                             max_new_tokens=768,
                             do_sample=True,
                             temperature=0.001,
                             top_p=0.95,
                             top_k=40,
                             repetition_penalty=1.2)
    all_token_ids = outputs[0].tolist()
    output_token_ids = all_token_ids[token_ids.shape[-1]:]
    output = tokenizer.decode(output_token_ids)

    print(output)

    return JSONResponse(content={"status": True, "data": conversation})
