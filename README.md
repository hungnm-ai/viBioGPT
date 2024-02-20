## viBioGPT: A Vietnamese Large Language Model for Biomedical Question Answering

----

**viBioGPT-7B-instruct** is a Vietnamese Large Language Model (LLM) fine-tuned for the task of Question Answering within
the medical and healthcare domain. This model uses
pre-trained [Vistral-Chat-7B](https://huggingface.co/Viet-Mistral/Vistral-7B-Chat), then QLora technique
to fine-tune.

### Table of Contents

---

* [Using viBioGPT with Transformer](#using-vibiogpt-with-transformer)
    * [Run on your device](#run-on-your-device)
    * [Run on Google colab](#run-on-google-colab)
* [Training Data](#training-data)
* [Training](#training)

## Using viBioGPT with Transformer

### Model Download

Our model has been fine-tuned based on the pre-trained model
model [Vistral-Chat-7B](https://huggingface.co/Viet-Mistral/Vistral-7B-Chat)

| Size | Hugging Face Model                                                                                            | Base Model                                                             |
|------|---------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------|
| 7B   | [hungnm/viBioGPT-7B-instruct-qlora-adapter](https://huggingface.co/hungnm/viBioGPT-7B-instruct-qlora-adapter) | [Vistral-Chat-7B](https://huggingface.co/Viet-Mistral/Vistral-7B-Chat) |

### Run on your device

Create environment with conda

```shell
conda create -n biogpt python=3.10
```

Activate environment

```shell
conda activate biogpt
```

Install dependencies

```shell
pip install peft==0.7.1 bitsandbytes==0.41.3.post2 transformers==4.36.2 torch==2.1.2 typer==0.9.0
```

Install Flash Attention 2

```shell
pip install flash-attn==2.3.3 --no-build-isolation
````

Example usage

_Note: replace your huggingface token_

```python
import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

HF_TOKEN = "<your_hf_token>"
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
                                             device_map={"": 0},
                                             token=HF_TOKEN
                                             )
model = PeftModel.from_pretrained(model, adapter)

# load and config tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name,
                                          token=HF_TOKEN)
tokenizer.padding_side = "left"
tokenizer.pad_token_id = tokenizer.eos_token_id

system_prompt = ("Bạn là một trợ lý ảo AI trong lĩnh vực Y học, Sức Khỏe. Tên của bạn là AI-Doctor. "
                 "Nhiệm vụ của bạn là trả lời các thắc mắc hoặc các câu hỏi về Y học, Sức khỏe.")

question = "tôi có một ít nhân sâm nhưng đang bị viêm dạ dày. Vậy tôi có nên ăn nhân sâm ko?"
conversation = [
    {
        "role": "system",
        "content": system_prompt},
    {
        "role": "user",
        "content": question
    }]
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

```

Output:

```text
Chào anh!
Nhân sâm được biết đến như loại thảo dược quý hiếm và rất tốt cho sức khoẻ con người tuy nhiên không phải ai cũng dùng được nó đặc biệt với những bệnh nhân đau dạ dày thì càng cần thận trọng khi sử dụng vì nếu lạm dụng sẽ gây ra nhiều tác hại nghiêm trọng tới hệ tiêu hoá nói chung và tình trạng đau dạ dày nói riêng .
Vì vậy trước tiên anh hãy điều trị dứt điểm căn bênh này rồi mới nghĩ tới việc bổ sung thêm dinh dưỡng từ nhân sâm nhé ! 
Chúc anh mau khỏi bệnh ạ!
```

Or Inference with script

```shell
python inference text-generate "tôi có một ít nhân sâm nhưng đang bị viêm dạ dày. Vậy tôi có nên ăn nhân sâm ko?"
```

### Run on Google colab

[Notebook](https://colab.research.google.com/drive/1yo53qWNo6bsfBNjp0IgLORQG0Howx30o?usp=drive_link)

## Training Data

Dataset collected from [edoctor](https://edoctor.io/hoi-dap)
and [vinmec](https://www.vinmec.com/vi/tin-tuc/hoi-dap-bac-si/).

* Size: After merging data from these two sources, obtained 9335 QA pairs.
* Language: Vietnamese

Data example:

```json

{
  "question": "Chào bác sĩ,\nRăng cháu hiện tại có mủ ở dưới lợi nhưng khi đau cháu sẽ không ngủ được (quá đau). Tuy nhiên chỉ vài ngày là hết mà thỉnh thoảng nó lại bị đau. Chị cháu bảo là trước chị cháu cũng bị như vậy chỉ là đau răng tuổi dậy thì thôi. Bác sĩ cho cháu hỏi đau răng kèm có mủ dưới lợi là bệnh gì? Cháu có cần đi chữa trị không? Cháu cảm ơn.",
  "answer": "Chào bạn,\nĐể trả lời câu hỏi trên, bác sĩ xin giải đáp như sau:\nRăng bạn hiện tại có mủ dưới lợi gây đau nhức nhiều. Bạn có thể đến phòng khám răng hàm mặt bệnh viện để được thăm khám, chụp phim và tư vấn cho bạn được chính xác\nTrân trọng!"
}

```

## Training

Because we use pretrained [Viet-Mistral/Vistral-7B-Chat](https://huggingface.co/Viet-Mistral/Vistral-7B-Chat), ensure
that you granted access to that model.

Then, you have to create .env file and set your **HF_TOKEN**, **WANDB_KEY**

```shell
HF_TOKEN=<YOUR_HF_TOKEN>
WANDB_KEY=<YOUR_WANDB_KEY>
```

To training model you can run command:

```shell
sh train.sh
```

or run:

```shell
python -m training \
    --model_name_or_path Viet-Mistral/Vistral-7B-Chat \
    --train_path ./data/qa_pairs_edoctor.json,./data/qa_vinmec.json \
    --lora True \
    --qlora True \
    --bf16 True \
    --output_dir models/bioGPT-instruct \
    --num_train_epochs 2 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 3 \
    --eval_accumulation_steps 1 \
    --evaluation_strategy "epoch" \
    --eval_steps 40 \
    --save_strategy "epoch" \
    --save_steps 100 \
    --save_total_limit 3 \
    --learning_rate 1.2e-5 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 1024 \
    --gradient_checkpointing True \
    --packing False \
    --report_to "wandb"
```

## Citation

If you find our project helpful, please star our repo and cite our work. Thanks!

```bibtex
@misc{viBioGPT,
      title={Vietnamese Medical QA: Question Answering dataset for medical in Vietnamese},
      author={Hung Nguyen},
      howpublished={\url{https://github.com/hungnm-ai/viBioGPT}},
      year={2024},
}
```